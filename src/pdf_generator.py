import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(battery_id, condition, cycle, soh, rul, health_score, health_band, anomaly_status, mitigation_action):
    """
    Generates a beautiful, high-quality PDF diagnostic report using ReportLab.
    Returns bytes of the PDF file.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1abc9c'),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#7f8c8d'),
        spaceAfter=25
    )
    
    heading_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceBefore=15,
        spaceAfter=10
    )
    
    normal_style = ParagraphStyle(
        'Normal_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2c3e50')
    )
    
    # Title
    story.append(Paragraph("BATTERY HEALTH DIAGNOSTICS REPORT", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Battery Analytics Platform", subtitle_style))
    story.append(Spacer(1, 10))
    
    # General Info & Predict Table
    story.append(Paragraph("1. System Configuration & Prognostics Summary", heading_style))
    
    data = [
        [Paragraph("<b>Battery ID</b>", normal_style), Paragraph(str(battery_id), normal_style),
         Paragraph("<b>Operational Profile</b>", normal_style), Paragraph(str(condition), normal_style)],
        [Paragraph("<b>Current Cycle</b>", normal_style), Paragraph(str(cycle), normal_style),
         Paragraph("<b>State of Health (SOH)</b>", normal_style), Paragraph(f"{soh:.2f}%", normal_style)],
        [Paragraph("<b>Estimated RUL</b>", normal_style), Paragraph(f"{int(rul)} Cycles", normal_style),
         Paragraph("<b>Composite Health Score</b>", normal_style), Paragraph(f"{health_score}/100", normal_style)]
    ]
    
    t = Table(data, colWidths=[100, 150, 130, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Anomaly Status Card
    story.append(Paragraph("2. Safety & Anomaly Center Diagnostics", heading_style))
    
    band_colors = {
        'Optimal': colors.HexColor('#2ecc71'),
        'Degraded': colors.HexColor('#f1c40f'),
        'Critical': colors.HexColor('#e74c3c'),
        'Normal': colors.HexColor('#2ecc71'),
        'Warning': colors.HexColor('#f39c12')
    }
    
    status_color = band_colors.get(health_band, colors.HexColor('#3498db'))
    
    status_data = [
        [
            Paragraph(f"<font color='white'><b>HEALTH STATUS BAND: {health_band.upper()}</b></font>", ParagraphStyle('StatusHeader', parent=normal_style, fontSize=12, leading=16)),
            Paragraph(f"<font color='white'><b>ANOMALY BAND: {anomaly_status.upper()}</b></font>", ParagraphStyle('StatusHeader2', parent=normal_style, fontSize=12, leading=16))
        ],
        [
            Paragraph(f"<b>Recommended Action / Mitigation Plan:</b><br/>{mitigation_action}", ParagraphStyle('Mitigation', parent=normal_style, textColor=colors.black)),
            Paragraph(f"<b>Active Safe Operations Protocol:</b><br/>Continuous state tracking active. Charge current limits enforced.", ParagraphStyle('Protocol', parent=normal_style, textColor=colors.black))
        ]
    ]
    
    st_t = Table(status_data, colWidths=[265, 265])
    st_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), status_color),
        ('BACKGROUND', (0,1), (1,1), colors.HexColor('#fdfefe')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    
    story.append(st_t)
    story.append(Spacer(1, 20))
    
    # Disclaimer
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Disclaimer:</b> This diagnostic report is generated using statistical and deep learning models trained on the NASA randomized battery usage dataset. Final deployment safety checks should be conducted in accordance with cell manufacturer datasheets.", subtitle_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
