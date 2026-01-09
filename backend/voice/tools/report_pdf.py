"""PDF report generation for poker session stats.

Generates session statistics reports including VPIP charts
using matplotlib and reportlab.

Usage:
    from backend.voice.tools.report_pdf import generate_session_pdf

    pdf_path = generate_session_pdf(vpip_data)
"""
import io
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List

# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)

# Charts
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt


def generate_vpip_chart(vpip_data: Dict[str, Any]) -> bytes:
    """Generate VPIP pie chart as PNG bytes.

    Args:
        vpip_data: Dict with vpip_percentage key

    Returns:
        PNG image as bytes
    """
    vpip_pct = vpip_data.get("vpip_percentage", 0)
    fold_pct = 100 - vpip_pct

    # Handle edge cases
    if vpip_pct <= 0:
        vpip_pct = 0.1  # Show tiny slice
        fold_pct = 99.9
    elif vpip_pct >= 100:
        vpip_pct = 99.9
        fold_pct = 0.1

    fig, ax = plt.subplots(figsize=(4, 4))

    # WHAM Vision color scheme
    colors_list = ["#4CAF50", "#f44336"]  # Green for VPIP, Red for Fold

    wedges, texts, autotexts = ax.pie(
        [vpip_pct, fold_pct],
        labels=[f"VPIP {vpip_pct:.1f}%", f"Fold {fold_pct:.1f}%"],
        colors=colors_list,
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.05, 0),  # Slightly explode VPIP slice
    )

    # Style the text
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('white')

    ax.set_title("VPIP Distribution", fontsize=12, fontweight='bold')

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_session_pdf(
    vpip_data: Dict[str, Any],
    hand_history: Optional[List[Dict]] = None,
    output_path: Optional[str] = None,
    session_name: Optional[str] = None,
) -> str:
    """Generate session stats PDF report.

    Args:
        vpip_data: VPIP stats from PokerOCR.get_vpip_stats()
            Expected keys: vpip_percentage, hands_played, vpip_hands,
                          leak_analysis, recommendation
        hand_history: Optional list of hand history dicts
        output_path: Optional output path (auto-generates if None)
        session_name: Optional session name for report title

    Returns:
        Path to generated PDF file
    """
    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"vpip_report_{timestamp}.pdf"
        )

    # Create document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#1976D2')
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#333333')
    )

    # Title
    title = session_name or "WHAM Vision - Session Report"
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 20))

    # VPIP Stats Table
    story.append(Paragraph("Session Statistics", heading_style))

    vpip_pct = vpip_data.get("vpip_percentage", 0)
    hands_played = vpip_data.get("hands_played", 0)
    vpip_hands = vpip_data.get("vpip_hands", 0)

    stats_data = [
        ["Metric", "Value"],
        ["VPIP %", f"{vpip_pct:.1f}%"],
        ["Hands Played", str(hands_played)],
        ["Hands with VPIP", str(vpip_hands)],
        ["Hands Folded", str(hands_played - vpip_hands)],
    ]

    stats_table = Table(stats_data, colWidths=[200, 150])
    stats_table.setStyle(TableStyle([
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor('#1976D2')),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("TOPPADDING", (0, 0), (-1, 0), 12),
        # Data rows
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
        ("TOPPADDING", (0, 1), (-1, -1), 8),
        # Alignment
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        # Grid
        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0')),
        # Alternating row colors
        ("BACKGROUND", (0, 2), (-1, 2), colors.white),
        ("BACKGROUND", (0, 4), (-1, 4), colors.white),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 30))

    # VPIP Chart
    if hands_played > 0:
        story.append(Paragraph("VPIP Distribution", heading_style))

        chart_bytes = generate_vpip_chart(vpip_data)
        chart_path = os.path.join(tempfile.gettempdir(), "vpip_chart_temp.png")
        with open(chart_path, "wb") as f:
            f.write(chart_bytes)

        story.append(Image(chart_path, width=3*inch, height=3*inch))
        story.append(Spacer(1, 20))

    # Leak Analysis
    leak_analysis = vpip_data.get("leak_analysis")
    if leak_analysis:
        story.append(Paragraph("Leak Analysis", heading_style))
        story.append(Paragraph(leak_analysis, styles["Normal"]))
        story.append(Spacer(1, 10))

    # Recommendation
    recommendation = vpip_data.get("recommendation")
    if recommendation:
        story.append(Paragraph("Recommendation", heading_style))

        # Style recommendation box
        rec_style = ParagraphStyle(
            'Recommendation',
            parent=styles['Normal'],
            fontSize=11,
            backColor=colors.HexColor('#E8F5E9'),
            borderColor=colors.HexColor('#4CAF50'),
            borderWidth=1,
            borderPadding=10,
            spaceBefore=10,
            spaceAfter=10,
        )
        story.append(Paragraph(recommendation, rec_style))

    # Hand History (if provided)
    if hand_history and len(hand_history) > 0:
        story.append(Spacer(1, 20))
        story.append(Paragraph("Recent Hands", heading_style))

        # Build hand history table (limit to 10 hands)
        history_data = [["#", "Cards", "Board", "Pot", "Action"]]
        for i, hand in enumerate(hand_history[:10], 1):
            history_data.append([
                str(i),
                hand.get("hole_cards", "??"),
                hand.get("board", "-"),
                f"${hand.get('pot', 0)}",
                hand.get("action", "-"),
            ])

        history_table = Table(history_data, colWidths=[30, 80, 120, 60, 80])
        history_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor('#1976D2')),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor('#E0E0E0')),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor('#FAFAFA')),
        ]))
        story.append(history_table)

    # Footer
    story.append(Spacer(1, 40))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#999999'),
        alignment=1,  # Center
    )
    story.append(Paragraph(
        "Generated by WHAM Vision Poker HUD",
        footer_style
    ))

    # Build PDF
    doc.build(story)

    # Cleanup temp chart file
    chart_path = os.path.join(tempfile.gettempdir(), "vpip_chart_temp.png")
    if os.path.exists(chart_path):
        try:
            os.remove(chart_path)
        except OSError:
            pass

    return output_path


# Test
if __name__ == "__main__":
    # Test data
    test_vpip_data = {
        "vpip_percentage": 28.5,
        "hands_played": 47,
        "vpip_hands": 13,
        "leak_analysis": "Your VPIP of 28.5% is slightly above optimal for 6-max cash games (22-27%). Consider tightening up from early positions.",
        "recommendation": "Focus on position awareness. Play tighter from UTG and UTG+1, maintain aggression from late position."
    }

    test_hand_history = [
        {"hole_cards": "A♠K♠", "board": "K♥7♣2♦", "pot": 45, "action": "Raise"},
        {"hole_cards": "Q♣Q♦", "board": "A♠8♥3♣", "pot": 120, "action": "Call"},
        {"hole_cards": "9♥8♥", "board": "J♦T♠2♣7♥", "pot": 85, "action": "Fold"},
    ]

    pdf_path = generate_session_pdf(
        test_vpip_data,
        hand_history=test_hand_history,
        session_name="Test Session Report"
    )
    print(f"PDF generated: {pdf_path}")
