"""
ScanMyMouth AI — Project-level URLs (scanmymouth/urls.py)
FIXED: Added routes for index.html and results.html pages
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),

    # API routes
    path('api/', include('oral_detection.urls')),

    # Frontend pages
    # FIXED: /results/ must exist because index.html redirects there via JS
    path('',         TemplateView.as_view(template_name='index.html'),   name='home'),
    path('results/', TemplateView.as_view(template_name='results.html'), name='results'),
]

# Serve uploaded media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)