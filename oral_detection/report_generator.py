def generate_pdf_report(result: dict, original_image_bytes: bytes = None) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    )
    import io
    from datetime import datetime

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=25,
        leftMargin=25,
        topMargin=30,
        bottomMargin=25
    )

    styles = getSampleStyleSheet()

    # ✅ CLEAN STYLES
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=10
    )

    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=9,
        leading=12
    )

    small_style = ParagraphStyle(
        'Small',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey
    )

    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=6
    )

    story = []

    # ================= HEADER =================
    story.append(Paragraph("OralGuard AI — Clinical Report", title_style))

    story.append(Paragraph(
        f"Analysis ID: {result.get('analysis_id','N/A')}<br/>"
        f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}",
        small_style
    ))

    story.append(Spacer(1, 10))

    # ================= RESULT =================
    is_positive = result.get('detection_result') == 'positive'

    result_text = "POSITIVE — Cancer Risk Detected" if is_positive else "NEGATIVE — Low Risk"

    story.append(Paragraph("<b>Detection Result</b>", heading_style))
    story.append(Paragraph(result_text, normal_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph(result.get("ai_description", ""), normal_style))
    story.append(Spacer(1, 12))

    # ================= SCORES =================
    story.append(Paragraph("<b>Scores</b>", heading_style))

    score_data = [
        ["Severity", "Confidence", "Cancer Prob", "Normal Prob"],
        [
            f"{result.get('severity_score',0)}%",
            f"{result.get('confidence_score',0)}%",
            f"{result.get('cancer_probability',0)}%",
            f"{result.get('normal_probability',0)}%",
        ]
    ]

    table = Table(score_data, colWidths=[120,120,120,120])

    table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))

    story.append(table)
    story.append(Spacer(1, 15))

    # ================= IMAGES =================
    story.append(Paragraph("<b>Images</b>", heading_style))

    if original_image_bytes:
        img = Image(io.BytesIO(original_image_bytes), width=200, height=150)
        story.append(img)
        story.append(Spacer(1, 10))

    # ================= LESION =================
    story.append(Paragraph("<b>Lesion Analysis</b>", heading_style))

    lesion = result.get("lesion_analysis", {})

    lesion_data = [
        ["Parameter", "Value"],
        ["Location", lesion.get("location","-")],
        ["Size", lesion.get("size","-")],
        ["Color", lesion.get("color","-")],
        ["Border", lesion.get("border","-")],
        ["Surface", lesion.get("surface","-")],
    ]

    lesion_table = Table(lesion_data, colWidths=[150, 300])

    lesion_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
    ]))

    story.append(lesion_table)
    story.append(Spacer(1, 15))

    # ================= RECOMMENDATIONS =================
    story.append(Paragraph("<b>Recommendations</b>", heading_style))

    for i, rec in enumerate(result.get("recommendations", []), 1):
        story.append(Paragraph(f"{i}. {rec}", normal_style))
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 20))

    # ================= FOOTER =================
    story.append(Paragraph(
        "Disclaimer: This is an AI-generated report. Consult a doctor.",
        small_style
    ))

    doc.build(story)

    pdf = buffer.getvalue()
    buffer.close()
    return pdf