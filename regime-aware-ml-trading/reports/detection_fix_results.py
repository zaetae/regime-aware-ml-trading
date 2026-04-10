"""Generate PDF: Detection Fix Results — Before vs After comparison."""

import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

WIDTH, HEIGHT = A4
MARGIN = 25 * mm


def build_pdf(output_path="reports/detection_fix_results.pdf"):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "Title2", parent=styles["Title"], fontSize=18, spaceAfter=6,
        textColor=HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        "Sub", parent=styles["Normal"], fontSize=11,
        textColor=HexColor("#555555"), spaceAfter=14, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "H1", parent=styles["Heading1"], fontSize=14, spaceBefore=16,
        spaceAfter=6, textColor=HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=12, spaceBefore=12,
        spaceAfter=4, textColor=HexColor("#2d3436"),
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10, leading=14,
        spaceAfter=6, alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        "Bul", parent=styles["Normal"], fontSize=10, leading=14,
        leftIndent=20, bulletIndent=10, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        "Cap", parent=styles["Normal"], fontSize=9,
        textColor=HexColor("#666666"), alignment=TA_CENTER, spaceAfter=10,
    ))

    S = styles
    story = []

    def p(text, style="Body"):
        story.append(Paragraph(text, S[style]))

    def sp(h=6):
        story.append(Spacer(1, h))

    def bullet(text):
        story.append(Paragraph(f"\u2022  {text}", S["Bul"]))

    def hr():
        story.append(HRFlowable(
            width="100%", thickness=0.5, color=HexColor("#cccccc"),
            spaceBefore=6, spaceAfter=6,
        ))

    def tbl(data, col_widths=None):
        t = Table(data, colWidths=col_widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2d3436")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [HexColor("#ffffff"), HexColor("#f7f7f7")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        sp(8)

    # ================================================================
    # TITLE
    # ================================================================
    sp(30)
    p("Triangle &amp; Channel Detection Fix \u2014 Results", "Title2")
    p("Pivot-Point Fitting + Theil-Sen Regression + Containment Validation", "Sub")
    hr()
    p("Author: Zeineb Turki  |  Date: April 2026", "Sub")

    sp(10)
    p("<b>Summary.</b> The triangle and channel detectors were rebuilt using "
      "pivot-point pre-selection (swing highs/lows), Theil-Sen robust regression, "
      "and a hard containment validation gate. This document presents the "
      "before-vs-after comparison.")

    story.append(PageBreak())

    # ================================================================
    # 1. BEFORE vs AFTER
    # ================================================================
    p("1. Before vs After \u2014 Detection Counts", "H1")

    tbl([
        ["Metric", "Before (OLS)", "After (Pivot + Theil-Sen)"],
        ["Triangle detections", "38", "6"],
        ["  ascending_triangle", "3", "3"],
        ["  descending_triangle", "29", "0"],
        ["  desc_triangle_upper_test", "6", "3"],
        ["  symmetric_triangle", "0", "0"],
        ["Channel detections", "43", "1"],
        ["  channel_up", "43", "0"],
        ["  channel_down", "0", "1"],
        ["Total pattern detections", "81", "7"],
    ], col_widths=[160, 110, 160])

    p("The large reduction (81 \u2192 7) is intentional. The old detector fired on "
      "formations that did not visually match their claimed pattern type. "
      "The surviving 7 detections pass strict quality checks and all have "
      "trendlines that actually contain the price action.")

    # ================================================================
    # 2. CONTAINMENT
    # ================================================================
    p("2. Containment Ratio Comparison", "H1")

    p("Containment ratio = fraction of bars where High \u2264 upper trendline + tol "
      "AND Low \u2265 lower trendline \u2212 tol (tolerance = 0.1 \u00d7 ATR).")
    sp(4)

    tbl([
        ["", "Before (OLS)", "After (Pivot + Theil-Sen)"],
        ["Triangle mean containment", "3.8%", "87.7%"],
        ["Triangle min containment", "0.0%", "82.0%"],
        ["Channel mean containment", "8.5%", "74.0%"],
        ["Channel min containment", "0.0%", "74.0%"],
        ["Threshold enforced", "None", "80% (triangles), 70% (channels)"],
    ], col_widths=[160, 110, 160])

    p("<b>Key result:</b> containment improved from ~4\u20139% (OLS) to 74\u201396% "
      "(pivot + Theil-Sen). Every detection now has trendlines that visibly "
      "bound the price action \u2014 directly addressing the supervisor\u2019s feedback.")

    # ================================================================
    # 3. WHAT CHANGED
    # ================================================================
    p("3. What Changed in the Code", "H1")

    p("3.1 New shared module: src/patterns/pivots.py", "H2")
    bullet("<b>find_swing_highs(highs, order=5)</b> \u2014 identifies bars where High "
           "is the maximum in a \u00b15-bar window (confirmed local peaks).")
    bullet("<b>find_swing_lows(lows, order=5)</b> \u2014 same for local troughs.")
    bullet("<b>containment_ratio(highs, lows, upper, lower, tol)</b> \u2014 "
           "fraction of bars fully inside the pattern boundaries.")
    sp(4)

    p("3.2 Triangle detector (src/patterns/triangles.py)", "H2")
    tbl([
        ["Aspect", "Before", "After"],
        ["Data points", "All 50 bars' highs/lows", "Swing highs + swing lows\n(\u22653 per side)"],
        ["Regression", "np.polyfit (OLS)", "scipy.stats.theilslopes\n(robust median slopes)"],
        ["Slope thresholds", "Absolute (0.01)", "ATR-normalised\n(0.1 \u00d7 ATR / window)"],
        ["Containment check", "None", "\u226580% required"],
        ["Quality metadata", "Slope + intercept only", "+ containment_ratio,\npivot counts"],
    ], col_widths=[95, 130, 205])

    p("3.3 Channel detector (src/patterns/channels.py)", "H2")
    tbl([
        ["Aspect", "Before", "After"],
        ["Data points", "All 50 bars' highs/lows", "Swing highs + swing lows\n(\u22653 per side)"],
        ["Regression", "np.polyfit (OLS)", "scipy.stats.theilslopes\n(robust median slopes)"],
        ["R\u00b2 check", "On all 50 bars", "On pivot points only\n(more meaningful)"],
        ["Touch count", "Band = 0.3\u00d7ATR,\n\u22654 distinct", "Replaced by pivot count\n(\u22653 per side)"],
        ["Containment check", "None", "\u226570% required"],
    ], col_widths=[95, 130, 205])

    story.append(PageBreak())

    # ================================================================
    # 4. DETECTION CHARTS
    # ================================================================
    p("4. Detection Charts (After Fix)", "H1")

    p("Each chart shows the 50-bar formation window plus 10 forward context bars. "
      "Upper trendline (red) fitted to swing highs, lower trendline (blue) fitted "
      "to swing lows. Orange marker = detection event.")
    sp(6)

    chart_files = sorted(glob.glob("reports/charts/fixed_*.png"))
    for cf in chart_files:
        basename = os.path.basename(cf)
        # Extract info from filename
        parts = basename.replace("fixed_", "").replace(".png", "")

        try:
            img = Image(cf, width=440, height=160)
            story.append(img)
            p(f"<i>{parts}</i>", "Cap")
        except Exception as e:
            p(f"[Could not embed {basename}: {e}]")

        if chart_files.index(cf) == 3:
            story.append(PageBreak())
            p("4. Detection Charts (continued)", "H1")
            sp(4)

    story.append(PageBreak())

    # ================================================================
    # 5. IMPACT ON TRAINING PIPELINE
    # ================================================================
    p("5. Impact on Training Pipeline", "H1")

    tbl([
        ["Metric", "Before fix\n(tri/ch excluded)", "After fix\n(all patterns included)"],
        ["Total labeled events", "104", "111"],
        ["  S/R events", "41", "41"],
        ["  Multi top/bottom", "63", "63"],
        ["  Triangles", "0 (excluded)", "6"],
        ["  Channels", "0 (excluded)", "1"],
        ["Long labels", "46", "47"],
        ["Short labels", "32", "36"],
        ["No-trade labels", "26", "28"],
    ], col_widths=[130, 130, 170])

    p("With the fixed detectors, triangle and channel events can be safely "
      "re-included in the RF training set. The 7 additional events have "
      "verified geometric quality (containment \u226570%) and add pattern "
      "diversity to the model.")

    sp(8)
    p("<b>Recommendation:</b> re-include fixed triangle/channel events in "
      "training. Use <font face='Courier' size='9'>label_events(df)</font> "
      "without <font face='Courier' size='9'>exclude_patterns</font> to get "
      "all 111 events.")

    # ================================================================
    # 6. CONCLUSION
    # ================================================================
    sp(10)
    hr()
    p("6. Conclusion", "H1")

    p("The pivot-point approach fundamentally solves both issues identified "
      "by the supervisor:")
    sp(4)
    bullet("<b>Triangles:</b> candles are now contained within the trendlines "
           "(82\u201396% containment vs. ~4% before). Lines are fitted to "
           "confirmed swing points, not noise.")
    bullet("<b>Channels:</b> trendlines follow the price envelope without "
           "breakpoints (74% containment vs. ~9% before). The strict quality "
           "gate means only geometrically valid channels pass.")
    bullet("<b>Trade-off:</b> fewer detections (81 \u2192 7), but every "
           "surviving detection is visually correct and structurally sound.")
    sp(6)
    p("The detection quality is now suitable for ML training and thesis "
      "presentation. The containment_ratio metric is also available as "
      "a feature for downstream models.")

    doc.build(story)
    return output_path


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    path = build_pdf()
    print(f"PDF generated: {path}")
