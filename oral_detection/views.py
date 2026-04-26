"""
ScanMyMouth AI — REST API Views

POST   /api/analyze/             Upload image → AI analysis
GET    /api/analyze/<uuid>/pdf/  Download PDF clinical report
GET    /api/history/             Last 50 records
GET    /api/history/<uuid>/      Single record
GET    /api/health/              Liveness probe
GET    /api/stats/               Aggregate statistics
"""
import base64
import logging
from datetime import datetime, timezone

from django.core.files.base import ContentFile
from django.http import HttpResponse
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt

from .models import AnalysisRecord
from .serializers import AnalysisRequestSerializer, AnalysisRecordSerializer

logger = logging.getLogger(__name__)

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        from .ai_engine import OralCancerDetector
        _detector = OralCancerDetector.get_instance()
    return _detector


# ── POST /api/analyze/ ────────────────────────────────────────────────
@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def analyze_image(request):
    serializer = AnalysisRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(
            {'error': 'Invalid input', 'details': serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    image_file  = serializer.validated_data['image']
    image_bytes = image_file.read()

    try:
        result = _get_detector().predict(image_bytes)
    except Exception as exc:
        logger.exception("AI inference error")
        return Response(
            {'error': 'AI processing failed', 'details': str(exc)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    record = AnalysisRecord(
        detection_result   = result['detection_result'],
        risk_level         = result['risk_level'],
        confidence_score   = result['confidence_score'],
        severity_score     = result['severity_score'],
        cancer_probability = result['cancer_probability'],
        ai_description     = result['ai_description'],
        lesion_analysis    = result['lesion_analysis'],
        risk_factors       = result['risk_factors'],
        recommendations    = result['recommendations'],
        processing_time_ms = result['processing_time_ms'],
        model_version      = result['model_version'],
    )

    image_file.seek(0)
    record.original_image.save(
        f"upload_{record.id}.jpg", ContentFile(image_bytes), save=False
    )

    if result.get('gradcam_image'):
        try:
            gc_bytes = base64.b64decode(result['gradcam_image'])
            record.gradcam_image.save(
                f"gradcam_{record.id}.jpg", ContentFile(gc_bytes), save=False
            )
        except Exception as exc:
            logger.warning(f"Could not save Grad-CAM: {exc}")

    record.save()

    orig_url    = request.build_absolute_uri(record.original_image.url) if record.original_image else None
    gradcam_url = request.build_absolute_uri(record.gradcam_image.url)  if record.gradcam_image  else None

    return Response({
        **result,
        'analysis_id':        str(record.id),
        'original_image_url': orig_url,
        'gradcam_image_url':  gradcam_url,
        'created_at':         record.created_at.isoformat(),
    }, status=status.HTTP_200_OK)


# ── GET /api/analyze/<uuid>/pdf/ ─────────────────────────────────────
@api_view(['GET'])
def download_pdf_report(request, pk):
    """
    Generate and stream a clinical PDF report for a stored analysis.
    Includes original image, Grad-CAM heatmap, scores, lesion analysis,
    risk factors, recommendations, and medical disclaimer.
    """
    try:
        record = AnalysisRecord.objects.get(pk=pk)
    except AnalysisRecord.DoesNotExist:
        return Response({'error': 'Record not found'}, status=status.HTTP_404_NOT_FOUND)

    # Build result dict from stored record fields
    result = {
        'analysis_id':        str(record.id),
        'detection_result':   record.detection_result,
        'risk_level':         record.risk_level,
        'severity_label':     dict(AnalysisRecord.RISK_LEVELS).get(record.risk_level, record.risk_level),
        'confidence_score':   record.confidence_score,
        'severity_score':     record.severity_score,
        'cancer_probability': record.cancer_probability,
        'normal_probability': round(100.0 - record.cancer_probability, 2),
        'urgency':            _get_urgency(record.risk_level),
        'ai_description':     record.ai_description,
        'risk_factors':       record.risk_factors   or [],
        'recommendations':    record.recommendations or [],
        'lesion_analysis':    record.lesion_analysis or {},
        'processing_time_ms': record.processing_time_ms,
        'model_version':      record.model_version,
        'created_at':         record.created_at.isoformat(),
        'gradcam_image':      None,
    }

    # Read Grad-CAM image as base64 from saved file
    if record.gradcam_image:
        try:
            with record.gradcam_image.open('rb') as f:
                result['gradcam_image'] = base64.b64encode(f.read()).decode()
        except Exception as e:
            logger.warning(f"Could not read gradcam for PDF: {e}")

    # Read original image bytes
    original_image_bytes = None
    if record.original_image:
        try:
            with record.original_image.open('rb') as f:
                original_image_bytes = f.read()
        except Exception as e:
            logger.warning(f"Could not read original image for PDF: {e}")

    # Generate PDF
    try:
        from .report_generator import generate_pdf_report
        pdf_bytes = generate_pdf_report(result, original_image_bytes)
    except ImportError:
        return Response(
            {'error': 'reportlab not installed. Run: pip install reportlab'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    except Exception as exc:
        logger.exception("PDF generation failed")
        return Response(
            {'error': 'PDF generation failed', 'details': str(exc)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    filename = f"OralGuard_Report_{str(record.id)[:8]}.pdf"
    response = HttpResponse(pdf_bytes, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    response['Content-Length']      = len(pdf_bytes)
    return response


def _get_urgency(risk_level):
    """Map risk_level to urgency string."""
    return {
        'critical': 'Immediate',
        'high':     'Urgent',
        'moderate': 'Soon',
        'low':      'Routine',
    }.get((risk_level or '').lower(), 'Routine')


# ── GET /api/history/ ─────────────────────────────────────────────────
@api_view(['GET'])
def analysis_history(request):
    records    = AnalysisRecord.objects.all()[:50]
    serializer = AnalysisRecordSerializer(records, many=True, context={'request': request})
    return Response({'count': len(serializer.data), 'results': serializer.data})


# ── GET /api/history/<uuid>/ ─────────────────────────────────────────
@api_view(['GET'])
def analysis_detail(request, pk):
    try:
        record = AnalysisRecord.objects.get(pk=pk)
    except AnalysisRecord.DoesNotExist:
        return Response({'error': 'Record not found'}, status=status.HTTP_404_NOT_FOUND)
    return Response(AnalysisRecordSerializer(record, context={'request': request}).data)


# ── GET /api/health/ ─────────────────────────────────────────────────
@api_view(['GET'])
def health_check(request):
    try:
        _get_detector()
        model_status = 'loaded'
    except Exception:
        model_status = 'unavailable'

    return Response({
        'status':    'ok',
        'model':     model_status,
        'service':   'OralGuard AI',
        'version':   '1.0.0',
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })


# ── GET /api/stats/ ──────────────────────────────────────────────────
@api_view(['GET'])
def statistics(request):
    from django.db.models import Count, Avg

    total = AnalysisRecord.objects.count()
    if total == 0:
        return Response({'total': 0, 'message': 'No analyses recorded yet.'})

    return Response({
        'total_analyses':        total,
        'by_detection_result':   list(AnalysisRecord.objects.values('detection_result').annotate(count=Count('id'))),
        'by_risk_level':         list(AnalysisRecord.objects.values('risk_level').annotate(count=Count('id'))),
        'average_confidence':    round(AnalysisRecord.objects.aggregate(a=Avg('confidence_score'))['a'] or 0, 2),
        'average_processing_ms': round(AnalysisRecord.objects.aggregate(a=Avg('processing_time_ms'))['a'] or 0, 2),
    })