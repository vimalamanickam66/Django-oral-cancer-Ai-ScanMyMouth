from django.contrib import admin
from .models import AnalysisRecord


@admin.register(AnalysisRecord)
class AnalysisRecordAdmin(admin.ModelAdmin):
    list_display    = ['id', 'detection_result', 'risk_level',
                       'confidence_score', 'processing_time_ms', 'created_at']
    list_filter     = ['detection_result', 'risk_level']
    search_fields   = ['id', 'ai_description']
    readonly_fields = ['id', 'created_at', 'updated_at',
                       'gradcam_image', 'processing_time_ms']
    ordering        = ['-created_at']