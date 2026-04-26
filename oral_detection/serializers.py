"""
ScanMyMouth AI — DRF Serializers
"""
from rest_framework import serializers
from .models import AnalysisRecord


class AnalysisRequestSerializer(serializers.Serializer):
    """Validates incoming image upload."""
    image = serializers.ImageField(required=True)

    def validate_image(self, value):
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("Image must be under 10 MB.")
        allowed_types = ['image/jpeg', 'image/png', 'image/webp', 'image/heic']
        if hasattr(value, 'content_type') and value.content_type not in allowed_types:
            raise serializers.ValidationError(
                "Only JPG, PNG, WEBP, and HEIC images are accepted."
            )
        return value


class LesionAnalysisSerializer(serializers.Serializer):
    location = serializers.CharField()
    size     = serializers.CharField()
    color    = serializers.CharField()
    border   = serializers.CharField()
    surface  = serializers.CharField()


class AnalysisResultSerializer(serializers.Serializer):
    analysis_id        = serializers.UUIDField()
    detection_result   = serializers.CharField()
    risk_level         = serializers.CharField()
    severity_label     = serializers.CharField()
    confidence_score   = serializers.FloatField()
    severity_score     = serializers.FloatField()
    cancer_probability = serializers.FloatField()
    normal_probability = serializers.FloatField()
    urgency            = serializers.CharField()
    ai_description     = serializers.CharField()
    risk_factors       = serializers.ListField(child=serializers.CharField())
    recommendations    = serializers.ListField(child=serializers.CharField())
    lesion_analysis    = LesionAnalysisSerializer()
    gradcam_image      = serializers.CharField(allow_null=True, required=False)
    original_image_url = serializers.CharField(allow_null=True, required=False)
    gradcam_image_url  = serializers.CharField(allow_null=True, required=False)
    processing_time_ms = serializers.IntegerField()
    model_version      = serializers.CharField()
    created_at         = serializers.DateTimeField()


class AnalysisRecordSerializer(serializers.ModelSerializer):
    # FIXED: Return absolute URLs for image fields
    original_image_url = serializers.SerializerMethodField()
    gradcam_image_url  = serializers.SerializerMethodField()

    class Meta:
        model  = AnalysisRecord
        fields = '__all__'

    def get_original_image_url(self, obj):
        request = self.context.get('request')
        if obj.original_image and request:
            return request.build_absolute_uri(obj.original_image.url)
        return None

    def get_gradcam_image_url(self, obj):
        request = self.context.get('request')
        if obj.gradcam_image and request:
            return request.build_absolute_uri(obj.gradcam_image.url)
        return None