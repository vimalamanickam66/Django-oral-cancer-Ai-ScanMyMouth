"""
ScanMyMouth AI — Database Models
"""
import uuid
from django.db import models


class AnalysisRecord(models.Model):
    RISK_LEVELS = [
        ('low',      'Low Risk'),
        ('moderate', 'Moderate Risk'),
        ('high',     'High Risk'),
        ('critical', 'Critical'),
    ]
    DETECTION_RESULTS = [
        ('negative', 'Negative'),
        ('positive', 'Positive'),
    ]

    id               = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at       = models.DateTimeField(auto_now_add=True)
    updated_at       = models.DateTimeField(auto_now=True)

    original_image   = models.ImageField(upload_to='uploads/', blank=True, null=True)
    gradcam_image    = models.ImageField(upload_to='results/', blank=True, null=True)

    detection_result   = models.CharField(max_length=20, choices=DETECTION_RESULTS, blank=True)
    risk_level         = models.CharField(max_length=20, choices=RISK_LEVELS, blank=True)
    confidence_score   = models.FloatField(default=0.0)
    severity_score     = models.FloatField(default=0.0)
    cancer_probability = models.FloatField(default=0.0)

    ai_description  = models.TextField(blank=True)
    lesion_analysis = models.JSONField(default=dict)
    risk_factors    = models.JSONField(default=list)
    recommendations = models.JSONField(default=list)

    processing_time_ms = models.IntegerField(default=0)
    model_version      = models.CharField(max_length=50, default='ScanMyMouth-MobileNetV2-v1.0')

    class Meta:
        ordering            = ['-created_at']
        verbose_name        = 'Analysis Record'
        verbose_name_plural = 'Analysis Records'

    def __str__(self):
        return f"[{self.detection_result.upper()}] {self.risk_level} — {self.id}"