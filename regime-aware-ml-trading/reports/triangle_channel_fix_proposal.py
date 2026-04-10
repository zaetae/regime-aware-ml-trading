"""Generate PDF: Proposed Fixes for Triangle & Channel Detectors."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

WIDTH, HEIGHT = A4
MARGIN = 25 * mm


def build_pdf(output_path="reports/triangle_channel_fix_proposal.pdf"):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(
        "Title2", parent=styles["Title"], fontSize=18, spaceAfter=6,
        textColor=HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=11,
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
        "H3", parent=styles["Heading3"], fontSize=11, spaceBefore=10,
        spaceAfter=3, textColor=HexColor("#2d3436"),
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10, leading=14,
        spaceAfter=6, alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        "CodeBlock", parent=styles["Code"], fontSize=8, leading=10,
        spaceAfter=6, backColor=HexColor("#f5f5f5"),
        leftIndent=12, rightIndent=12,
    ))
    styles.add(ParagraphStyle(
        "BulletItem", parent=styles["Normal"], fontSize=10, leading=14,
        leftIndent=20, bulletIndent=10, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        "Caption", parent=styles["Normal"], fontSize=9,
        textColor=HexColor("#666666"), alignment=TA_CENTER, spaceAfter=10,
    ))

    S = styles
    story = []

    def p(text, style="Body"):
        story.append(Paragraph(text, S[style]))

    def sp(h=6):
        story.append(Spacer(1, h))

    def bullet(text):
        story.append(Paragraph(f"\u2022  {text}", S["BulletItem"]))

    def hr():
        story.append(HRFlowable(
            width="100%", thickness=0.5, color=HexColor("#cccccc"),
            spaceBefore=6, spaceAfter=6,
        ))

    def table(data, col_widths=None):
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
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        sp(8)

    # ================================================================
    # TITLE PAGE
    # ================================================================
    sp(40)
    p("Proposed Fixes for Triangle &amp; Channel Detectors", "Title2")
    p("Regime-Aware ML Trading Project", "Subtitle")
    hr()
    p("Author: Zeineb Turki", "Subtitle")
    p("Date: April 2026", "Subtitle")
    sp(20)

    p("<b>Context.</b> The current triangle and channel pattern detectors use ordinary "
      "least-squares (OLS) regression on <i>all</i> highs and lows within a rolling window. "
      "The professor's review identified two structural problems: (1) triangle trendlines "
      "do not contain the price action \u2014 most candles fall outside the triangle area; "
      "(2) channel boundaries have breakpoints and many outlier candles escape the bands. "
      "This document diagnoses the root causes and proposes concrete fixes backed by "
      "quantitative finance literature.")
    sp(6)
    p("<b>Scope.</b> We cover the diagnosis of the current implementation, the proposed "
      "methodology (pivot-point fitting, robust regression, containment validation), "
      "implementation details, and expected impact on detection quality. "
      "The triangle and channel detectors will be discussed in the next meeting with "
      "the supervisor.")
    story.append(PageBreak())

    # ================================================================
    # 1. DIAGNOSIS
    # ================================================================
    p("1. Diagnosis of Current Detectors", "H1")

    p("1.1 Triangle Detector \u2014 Current Approach", "H2")
    p("The triangle detector (<font face='Courier' size='9'>src/patterns/triangles.py</font>) "
      "fits two linear regression lines via <font face='Courier' size='9'>np.polyfit</font> "
      "to <b>all</b> High values (upper line) and <b>all</b> Low values (lower line) "
      "within a 50-bar rolling window. It classifies the triangle type based on raw slope "
      "signs and checks a simple compression metric (range shrinkage between the first and "
      "last 5 bars).")

    p("1.1.1 Identified Weaknesses", "H3")

    table([
        ["Issue", "Severity", "Impact on Chart Quality"],
        ["OLS fits the mean of all\nhighs/lows, not the boundary",
         "HIGH",
         "Line cuts through the middle of the highs\nrather than capping them \u2192 candles above/below"],
        ["No containment validation",
         "HIGH",
         "Never checks what % of bars actually lie\nbetween the two fitted lines"],
        ["No R\u00b2 or goodness-of-fit\ncheck on trendlines",
         "HIGH",
         "Poorly-fitting lines accepted as valid\npatterns \u2192 visual garbage"],
        ["Slope thresholds are absolute\n(0.01), not ATR-normalized",
         "MEDIUM",
         "Same threshold for $5 and $500 stocks;\nmisclassifies triangle types at different\nprice levels"],
        ["Outlier sensitivity",
         "MEDIUM",
         "A single spike pulls the entire OLS line\naway from the true boundary"],
        ["No pivot-point pre-selection",
         "MEDIUM",
         "Every bar gets equal weight; noise\ndominates structure"],
    ], col_widths=[120, 55, 260])

    p("1.2 Channel Detector \u2014 Current Approach", "H2")
    p("The channel detector (<font face='Courier' size='9'>src/patterns/channels.py</font>) "
      "also uses OLS on all highs/lows. It adds several validation filters: R\u00b2 \u2265 0.70, "
      "slope parallelism within 15%, channel width between 1\u20136\u00d7 ATR, and \u22654 distinct "
      "touches per line. These are better than the triangle detector, but the core fitting "
      "problem persists.")

    p("1.2.1 Identified Weaknesses", "H3")

    table([
        ["Issue", "Severity", "Impact on Chart Quality"],
        ["OLS on all bars fits the\naverage, not the envelope",
         "HIGH",
         "~50% of points above the upper line by\nconstruction \u2192 many 'outlier' candles"],
        ["Touch band (0.3\u00d7ATR)\nis too wide",
         "MEDIUM",
         "Bars 0.3\u00d7ATR outside the channel still\ncount as 'touches' \u2192 loose validation"],
        ["Width uses mean(), not\nmin/max across window",
         "MEDIUM",
         "Channel can be broken in parts while\nmean check still passes"],
        ["No explicit containment\npercentage check",
         "MEDIUM",
         "R\u00b2 \u2265 0.70 allows 30% unexplained variance;\nmany bars can still escape the channel"],
        ["No outlier-robust\nregression",
         "MEDIUM",
         "Earnings gaps or flash moves distort\nthe entire 50-bar regression"],
    ], col_widths=[120, 55, 260])

    p("1.3 The Core Problem: OLS Is Not a Boundary Estimator", "H2")
    p("Ordinary least squares minimizes the sum of squared residuals, producing a line "
      "that passes through the <i>center</i> of the data cloud. By construction, roughly "
      "half the points lie above and half below the fitted line. For trendline detection, "
      "we need <b>boundary estimators</b> \u2014 lines that cap or floor the price action, "
      "with the vast majority of points on one side. This is the fundamental mismatch "
      "causing both the triangle and channel issues.")

    story.append(PageBreak())

    # ================================================================
    # 2. PROPOSED METHODOLOGY
    # ================================================================
    p("2. Proposed Fix Methodology", "H1")

    p("The fix involves three complementary changes: (A) fit trendlines to "
      "<b>pivot points</b> (swing highs/lows) instead of all bars, (B) use "
      "<b>robust or quantile regression</b> instead of OLS, and (C) add "
      "<b>containment validation</b> as a hard quality gate.")

    # --- 2.1 Pivot points ---
    p("2.1 Step A \u2014 Pivot-Point Pre-selection", "H2")
    p("Instead of fitting lines to every bar's High/Low, we first identify "
      "<b>swing highs</b> and <b>swing lows</b> (also called pivot points). "
      "A swing high at bar <i>i</i> requires that High[i] is the maximum High "
      "within a neighborhood of \u00b1<i>k</i> bars (typically <i>k</i> = 5). "
      "Symmetrically for swing lows.")

    p("<b>Why this helps:</b>")
    bullet("Reduces the input from ~50 noisy bars to ~5\u201310 structurally meaningful "
           "extrema.")
    bullet("Each pivot point is a confirmed local extreme, not intra-trend noise.")
    bullet("The resulting trendline is anchored to actual peaks/troughs that a human "
           "chartist would draw through.")
    bullet("This is the approach used in Lo, Mamaysky &amp; Wang (2000), the foundational "
           "academic paper on computational technical analysis.")

    sp(4)
    p("Pseudocode:", "H3")
    p("<font face='Courier' size='8'>"
      "def find_swing_highs(df, left=5, right=5):<br/>"
      "&nbsp;&nbsp;&nbsp;&nbsp;pivots = []<br/>"
      "&nbsp;&nbsp;&nbsp;&nbsp;for i in range(left, len(df) - right):<br/>"
      "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if df['High'].iloc[i] == "
      "df['High'].iloc[i-left:i+right+1].max():<br/>"
      "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
      "pivots.append(i)<br/>"
      "&nbsp;&nbsp;&nbsp;&nbsp;return pivots<br/>"
      "<br/>"
      "# Same logic with 'Low' and .min() for swing lows"
      "</font>", "CodeBlock")
    p("Minimum requirement: \u22653 swing highs for the upper line, \u22653 swing lows "
      "for the lower line. Bulkowski (2021) recommends \u22655 total touches.", "Caption")

    # --- 2.2 Robust regression ---
    p("2.2 Step B \u2014 Robust / Quantile Regression", "H2")
    p("After selecting pivot points, we replace OLS with a regression method "
      "suited to boundary fitting:")
    sp(4)

    p("2.2.1 For Channels: Quantile Regression (recommended)", "H3")
    p("Quantile regression at \u03c4 = 0.95 fits a line such that 95% of data points "
      "lie below it \u2014 a natural upper boundary. At \u03c4 = 0.05 it produces a lower "
      "boundary with 95% of points above. This directly solves the 'half above, half below' "
      "problem of OLS.")
    bullet("Implementation: <font face='Courier' size='9'>statsmodels.QuantReg</font> "
           "(already installable on Colab).")
    bullet("Apply on swing highs (\u03c4 = 0.90\u20130.95) for the upper line, "
           "swing lows (\u03c4 = 0.05\u20130.10) for the lower line.")
    bullet("The resulting channel naturally contains ~85\u201390% of price action.")

    sp(4)
    p("2.2.2 For Triangles: Theil-Sen Estimator (recommended)", "H3")
    p("Theil-Sen computes the median of all pairwise slopes. It breaks down only when "
      ">29% of input points are outliers (vs. a single outlier breaking OLS). Since we "
      "already pre-selected pivot points, the input is clean and Theil-Sen provides "
      "the most robust slope estimate.")
    bullet("Implementation: <font face='Courier' size='9'>scipy.stats.theilslopes</font> "
           "(already available).")
    bullet("Fit to swing highs for upper line, swing lows for lower line.")
    bullet("After fitting: verify convergence (slopes have opposite signs or one near-zero).")

    sp(4)
    p("2.2.3 Alternative: RANSAC", "H3")
    p("RANSAC (Random Sample Consensus) iteratively fits lines to random subsets and "
      "picks the fit with the most inliers. It naturally ignores outlier spikes. "
      "Available via <font face='Courier' size='9'>sklearn.linear_model.RANSACRegressor"
      "</font>. This is a viable alternative to Theil-Sen, especially if outlier spikes "
      "are a persistent issue after pivot pre-selection.")

    sp(4)
    table([
        ["Method", "Best For", "Outlier Tolerance", "Boundary Fit?"],
        ["OLS (current)", "Neither", "Breaks with 1 outlier", "No \u2014 fits the mean"],
        ["Theil-Sen", "Triangles", "Tolerates up to 29%\noutliers", "Partial \u2014 fits the\nmedian slope"],
        ["Quantile Reg.", "Channels", "Robust by design", "Yes \u2014 fits the\n\u03c4-th percentile"],
        ["RANSAC", "Both", "Ignores outlier\nsubset entirely", "Partial \u2014 fits the\ninlier majority"],
    ], col_widths=[80, 80, 110, 110])

    # --- 2.3 Containment ---
    p("2.3 Step C \u2014 Containment Validation", "H2")
    p("After fitting the trendlines, we add a <b>hard quality gate</b>: what fraction "
      "of bars actually lie within the pattern boundaries?")

    p("<b>Containment ratio</b> = (number of bars where High \u2264 upper_line + tol "
      "AND Low \u2265 lower_line \u2212 tol) / total bars in window.")
    sp(4)
    bullet("Tolerance: 0.1 \u00d7 ATR (allow tiny wick violations, not full-bar breaches).")
    bullet("<b>Triangle threshold</b>: containment \u2265 0.80 (80% of bars inside).")
    bullet("<b>Channel threshold</b>: containment \u2265 0.85 (channels should be tighter).")
    bullet("Any detection below the threshold is rejected \u2014 no signal fires.")

    sp(4)
    p("This single check would have caught most of the professor's objections: "
      "if most candles fall outside the triangle, containment is far below 80% and "
      "the detection is suppressed.")

    story.append(PageBreak())

    # ================================================================
    # 3. DETAILED IMPLEMENTATION PLAN
    # ================================================================
    p("3. Implementation Plan", "H1")

    p("3.1 Triangle Detector Fixes", "H2")
    p("File: <font face='Courier' size='9'>src/patterns/triangles.py</font>")
    sp(4)

    table([
        ["#", "Change", "Details"],
        ["1", "Add pivot detection\nhelper",
         "find_swing_highs(df, left=5, right=5) and\n"
         "find_swing_lows(). Place in src/patterns/pivots.py\n"
         "as shared utility."],
        ["2", "Replace np.polyfit\nwith Theil-Sen",
         "scipy.stats.theilslopes on pivot High values\n"
         "(upper line) and pivot Low values (lower line).\n"
         "Require \u22653 pivots per line."],
        ["3", "Add R\u00b2 check on\npivot fits",
         "Compute R\u00b2 of Theil-Sen fit against the pivot\n"
         "points. Reject if R\u00b2 < 0.70."],
        ["4", "Add containment\nratio check",
         "Iterate all bars in window; check High \u2264 upper\n"
         "and Low \u2265 lower (with 0.1\u00d7ATR tolerance).\n"
         "Reject if containment < 0.80."],
        ["5", "Normalize slope\nthresholds by ATR",
         "Replace absolute thresholds (0.01) with\n"
         "ATR-relative: |slope| < 0.1 \u00d7 ATR / window\n"
         "means 'flat'."],
        ["6", "Add convergence\ncheck",
         "Project lines forward; verify the gap narrows.\n"
         "Reject if projected gap \u2265 95% of current gap."],
    ], col_widths=[20, 100, 310])

    p("3.2 Channel Detector Fixes", "H2")
    p("File: <font face='Courier' size='9'>src/patterns/channels.py</font>")
    sp(4)

    table([
        ["#", "Change", "Details"],
        ["1", "Add pivot detection",
         "Same shared utility as triangles."],
        ["2", "Replace np.polyfit\nwith quantile regression",
         "statsmodels.QuantReg on swing highs (\u03c4=0.95)\n"
         "for upper line, swing lows (\u03c4=0.05) for lower\n"
         "line. Alternatively, Theil-Sen + percentile\n"
         "offset."],
        ["3", "Add containment\nratio check",
         "Require \u226585% of bars inside the channel.\n"
         "Tolerance: 0.1\u00d7ATR."],
        ["4", "Tighten touch band\nfrom 0.3 to 0.15\u00d7ATR",
         "Current 0.3\u00d7ATR is too permissive.\n"
         "0.15\u00d7ATR ensures touches are genuine\n"
         "boundary interactions."],
        ["5", "Use min(width) not\nmean(width)",
         "Validate that the channel doesn't collapse\n"
         "or blow out at any point in the window.\n"
         "Require: min(width) > 0.5\u00d7ATR."],
        ["6", "Keep existing R\u00b2\n\u2265 0.70 filter",
         "The current R\u00b2 check is good. The pivot +\n"
         "quantile regression will make it more\n"
         "meaningful (fewer points, better fits)."],
    ], col_widths=[20, 100, 310])

    p("3.3 Shared Pivot Utility", "H2")
    p("Create <font face='Courier' size='9'>src/patterns/pivots.py</font> with:")
    sp(2)
    bullet("<font face='Courier' size='9'>find_swing_highs(series, left=5, right=5)"
           "</font> \u2192 list of bar indices")
    bullet("<font face='Courier' size='9'>find_swing_lows(series, left=5, right=5)"
           "</font> \u2192 list of bar indices")
    bullet("<font face='Courier' size='9'>containment_ratio(df, upper_line, lower_line, "
           "tolerance)</font> \u2192 float")
    sp(4)
    p("These functions are reusable across both detectors and any future "
      "pattern recognizer (e.g., head-and-shoulders, wedges).")

    story.append(PageBreak())

    # ================================================================
    # 4. EXPECTED IMPACT
    # ================================================================
    p("4. Expected Impact", "H1")

    p("4.1 Detection Count", "H2")
    p("Adding pivot-point fitting and containment validation will <b>reduce</b> the "
      "number of detected patterns. This is intentional: the current detector fires "
      "on many low-quality formations that visually fail. After the fix, fewer but "
      "higher-quality detections should remain.")
    sp(4)

    table([
        ["Metric", "Current", "Expected After Fix"],
        ["Triangle detections", "~38", "~10\u201320 (quality-filtered)"],
        ["Channel detections", "~43", "~15\u201325 (quality-filtered)"],
        ["Containment ratio (avg)", "Unknown (not checked)", "\u226580% (enforced)"],
        ["Visual quality", "Many candles outside\npattern area", "Trendlines visibly bound\nprice action"],
    ], col_widths=[130, 130, 170])

    p("4.2 Impact on Training Pipeline", "H2")
    p("Currently, triangle and channel events are excluded from the RF training set "
      "(per supervisor direction). After fixing the detectors:")
    bullet("High-quality triangle/channel events can be <b>re-included</b> in training, "
           "increasing the event count from 104 to an estimated 130\u2013150.")
    bullet("More diverse event types improve the model's ability to generalize "
           "across different market structures.")
    bullet("The containment ratio itself becomes a useful <b>feature</b> for the ML "
           "model \u2014 higher-quality patterns may have better predictive power.")

    p("4.3 Backward Compatibility", "H2")
    bullet("The <font face='Courier' size='9'>scan_all_patterns()</font> API remains "
           "unchanged \u2014 returns the same columns.")
    bullet("<font face='Courier' size='9'>label_events(exclude_patterns=...)</font> "
           "still works for filtering if needed.")
    bullet("<font face='Courier' size='9'>return_details=True</font> will include "
           "additional quality metrics (containment_ratio, pivot_count).")
    bullet("Notebook 04 (structure validation) can visualize the improvement "
           "with a before/after comparison.")

    story.append(PageBreak())

    # ================================================================
    # 5. REFERENCES
    # ================================================================
    p("5. References", "H1")
    sp(4)

    refs = [
        ("Lo, A. W., Mamaysky, H., &amp; Wang, J. (2000).",
         "Foundations of Technical Analysis: Computational Algorithms, Statistical "
         "Inference, and Empirical Implementation. <i>Journal of Finance</i>, 55(4), "
         "1705\u20131765.",
         "Foundational paper. Uses kernel smoothing + local extrema for pattern detection. "
         "Establishes the pivot-point approach used in our proposed fix."),

        ("Bulkowski, T. N. (2021).",
         "<i>Encyclopedia of Chart Patterns</i> (3rd ed.). Wiley.",
         "Empirical reference for pattern quality criteria: minimum touches, containment, "
         "breakout position, and failure rates."),

        ("Koenker, R. (2005).",
         "<i>Quantile Regression</i>. Cambridge University Press.",
         "Mathematical foundation for quantile regression. Our proposed channel fix uses "
         "\u03c4 = 0.95 / 0.05 for boundary fitting."),

        ("Lopez de Prado, M. (2018).",
         "<i>Advances in Financial Machine Learning</i>. Wiley.",
         "Triple-barrier labeling method (already implemented in Phase 5). "
         "Also discusses the importance of quality event detection for ML inputs."),

        ("Rousseeuw, P. J., &amp; Leroy, A. M. (2003).",
         "<i>Robust Regression and Outlier Detection</i>. Wiley.",
         "Covers RANSAC, LMS, Theil-Sen, and other robust methods applicable to "
         "trendline fitting. Theil-Sen breakdown point = 29%."),

        ("Leigh, W., Modani, N., Purvis, R., &amp; Roberts, T. (2002).",
         "Stock market trading rule discovery using technical charting heuristics. "
         "<i>Expert Systems with Applications</i>, 23(2), 155\u2013159.",
         "Template-matching approach to chart pattern detection."),
    ]

    for i, (author, title, note) in enumerate(refs, 1):
        p(f"<b>[{i}]</b> {author} {title}")
        p(f"<i>Relevance:</i> {note}", "Caption")
        sp(2)

    # Build
    doc.build(story)
    return output_path


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    path = build_pdf()
    print(f"PDF generated: {path}")
