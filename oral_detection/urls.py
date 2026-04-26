from django.urls import path
from . import views

urlpatterns = [
    # Analysis
    path('analyze/',               views.analyze_image,       name='analyze'),
    path('analyze/<uuid:pk>/pdf/', views.download_pdf_report, name='download-pdf'),

    # History
    path('history/',               views.analysis_history,    name='history'),
    path('history/<uuid:pk>/',     views.analysis_detail,     name='analysis-detail'),

    # Utilities
    path('health/',                views.health_check,        name='health'),
    path('stats/',                 views.statistics,          name='stats'),
]