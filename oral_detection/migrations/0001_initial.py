from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='AnalysisRecord',
            fields=[
                ('id',                models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('created_at',        models.DateTimeField(auto_now_add=True)),
                ('updated_at',        models.DateTimeField(auto_now=True)),
                ('original_image',    models.ImageField(blank=True, null=True, upload_to='uploads/')),
                ('gradcam_image',     models.ImageField(blank=True, null=True, upload_to='results/')),
                ('detection_result',  models.CharField(blank=True, choices=[('negative','Negative'),('positive','Positive')], max_length=20)),
                ('risk_level',        models.CharField(blank=True, choices=[('low','Low Risk'),('moderate','Moderate Risk'),('high','High Risk'),('critical','Critical')], max_length=20)),
                ('confidence_score',  models.FloatField(default=0.0)),
                ('severity_score',    models.FloatField(default=0.0)),
                ('cancer_probability',models.FloatField(default=0.0)),
                ('ai_description',    models.TextField(blank=True)),
                ('lesion_analysis',   models.JSONField(default=dict)),
                ('risk_factors',      models.JSONField(default=list)),
                ('recommendations',   models.JSONField(default=list)),
                ('processing_time_ms',models.IntegerField(default=0)),
                ('model_version',     models.CharField(default='ScanMyMouth-CNN-v1.0', max_length=50)),
            ],
            options={'ordering': ['-created_at'], 'verbose_name': 'Analysis Record', 'verbose_name_plural': 'Analysis Records'},
        ),
    ]
