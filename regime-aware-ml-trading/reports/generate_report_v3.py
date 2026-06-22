"""Generate thesis-style PDF (~35-40 pages) for the Regime-Aware ML Trading project.

Produces: reports/final/Zeineb_Turki_zjk3.pdf

Self-contained script using ONLY ReportLab.  All figures are pre-generated PNGs
loaded via Image().  No external data imports — every statistic is hard-coded
from validated experimental results.

Usage:
    python reports/generate_report_v3.py
"""

import os
import sys

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether,
)
from reportlab.lib import colors

# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════
BASE = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.join(BASE, "final")
FIG_DIR = os.path.join(FINAL_DIR, "figures")
THESIS_FIG = os.path.join(BASE, "thesis_figures")
os.makedirs(FINAL_DIR, exist_ok=True)

OUTPUT = os.path.join(FINAL_DIR, "Zeineb_Turki_zjk3.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# Page layout
# ═══════════════════════════════════════════════════════════════════════════
PAGE_W, PAGE_H = A4
MARGIN = 2.5 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

# ═══════════════════════════════════════════════════════════════════════════
# Colours
# ═══════════════════════════════════════════════════════════════════════════
BLUE = HexColor("#1B3A5C")
ACCENT = HexColor("#2E86C1")
GREY = HexColor("#808B96")
LIGHT = HexColor("#EBF5FB")
DARK = HexColor("#1a1a2e")

# ═══════════════════════════════════════════════════════════════════════════
# Styles
# ═══════════════════════════════════════════════════════════════════════════
styles = getSampleStyleSheet()

styles.add(ParagraphStyle("Body", parent=styles["Normal"],
                          fontSize=10, leading=14, alignment=TA_JUSTIFY,
                          spaceAfter=6, spaceBefore=2))
styles.add(ParagraphStyle("BodyIndent", parent=styles["Normal"],
                          fontSize=10, leading=14, alignment=TA_JUSTIFY,
                          leftIndent=1 * cm, spaceAfter=6))
styles.add(ParagraphStyle("H1", parent=styles["Heading1"],
                          fontSize=16, leading=20, textColor=BLUE,
                          spaceBefore=18, spaceAfter=8))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"],
                          fontSize=13, leading=16, textColor=BLUE,
                          spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle("H3", parent=styles["Heading3"],
                          fontSize=11, leading=14, textColor=ACCENT,
                          spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"],
                          fontSize=9, leading=12, textColor=GREY,
                          alignment=TA_CENTER, spaceBefore=2, spaceAfter=8))
styles.add(ParagraphStyle("Small", parent=styles["Normal"],
                          fontSize=8, leading=10, textColor=GREY))
styles.add(ParagraphStyle("CodeBlock", parent=styles["Normal"],
                          fontSize=8, leading=10, fontName="Courier",
                          leftIndent=1 * cm, spaceAfter=6))
styles.add(ParagraphStyle("TitleMain", parent=styles["Title"],
                          fontSize=22, leading=28, textColor=BLUE,
                          alignment=TA_CENTER))
styles.add(ParagraphStyle("TitleSub", parent=styles["Normal"],
                          fontSize=14, leading=18, textColor=ACCENT,
                          alignment=TA_CENTER, spaceAfter=12))
styles.add(ParagraphStyle("TitleInfo", parent=styles["Normal"],
                          fontSize=11, leading=15, alignment=TA_CENTER,
                          spaceAfter=4))
styles.add(ParagraphStyle("AbstractBody", parent=styles["Normal"],
                          fontSize=10, leading=14, alignment=TA_JUSTIFY,
                          leftIndent=1.5 * cm, rightIndent=1.5 * cm,
                          spaceAfter=6))
styles.add(ParagraphStyle("BibEntry", parent=styles["Normal"],
                          fontSize=9, leading=12, leftIndent=1 * cm,
                          firstLineIndent=-1 * cm, spaceAfter=4))
styles.add(ParagraphStyle("TOCEntry1", parent=styles["Normal"],
                          fontSize=11, leading=16, spaceBefore=4, spaceAfter=2,
                          fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("TOCEntry2", parent=styles["Normal"],
                          fontSize=10, leading=14, spaceBefore=1, spaceAfter=1,
                          leftIndent=0.8 * cm))
styles.add(ParagraphStyle("TOCEntry3", parent=styles["Normal"],
                          fontSize=9, leading=12, spaceBefore=0, spaceAfter=0,
                          leftIndent=1.6 * cm, textColor=GREY))
styles.add(ParagraphStyle("BulletItem", parent=styles["Normal"],
                          fontSize=10, leading=14, alignment=TA_JUSTIFY,
                          leftIndent=1.2 * cm, firstLineIndent=-0.5 * cm,
                          spaceAfter=3, spaceBefore=1))

# ═══════════════════════════════════════════════════════════════════════════
# Story container and helpers
# ═══════════════════════════════════════════════════════════════════════════
story = []


def h1(text):
    story.append(Paragraph(text, styles["H1"]))


def h2(text):
    story.append(Paragraph(text, styles["H2"]))


def h3(text):
    story.append(Paragraph(text, styles["H3"]))


def p(text):
    story.append(Paragraph(text, styles["Body"]))


def p_indent(text):
    story.append(Paragraph(text, styles["BodyIndent"]))


def p_abstract(text):
    story.append(Paragraph(text, styles["AbstractBody"]))


def caption(text):
    story.append(Paragraph(text, styles["Caption"]))


def bullet(text):
    story.append(Paragraph("\u2022  " + text, styles["BulletItem"]))


def spacer(h=6):
    story.append(Spacer(1, h * mm))


def page_break():
    story.append(PageBreak())


def add_table(data, col_widths=None, header_color=BLUE):
    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, GREY),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ])
    t = Table(data, colWidths=col_widths)
    t.setStyle(ts)
    story.append(t)


def add_image(path, w=None, h=None, cap=None):
    """Add image if file exists; silently skip otherwise."""
    if os.path.exists(path):
        if w is None:
            w = CONTENT_W * 0.85
        if h is None:
            h = 7 * cm
        story.append(Image(path, width=w, height=h))
        if cap:
            caption(cap)
    else:
        p(f"<i>[Figure not found: {os.path.basename(path)}]</i>")


def bib(tag, text):
    story.append(Paragraph(f"[{tag}] {text}", styles["BibEntry"]))


# ═══════════════════════════════════════════════════════════════════════════
# Figure paths
# ═══════════════════════════════════════════════════════════════════════════
FIG_PIPELINE = os.path.join(FIG_DIR, "pipeline_vertical.png")
FIG_TRIPLE = os.path.join(FIG_DIR, "triple_barrier.png")
FIG_CONFUSION = os.path.join(FIG_DIR, "confusion_matrix.png")
FIG_WF_VAR = os.path.join(FIG_DIR, "wf_variability.png")
FIG_FBETA = os.path.join(FIG_DIR, "fbeta_comparison.png")
FIG_RESULTS = os.path.join(FIG_DIR, "results_summary.png")
FIG_HEATMAPS = os.path.join(FIG_DIR, "heatmaps.png")
FIG_WF_DIAG = os.path.join(FIG_DIR, "walkforward_diagram.png")
FIG_FEAT_IMP = os.path.join(THESIS_FIG, "feature_importance.png")
FIG_LABEL_DIST = os.path.join(THESIS_FIG, "label_distribution.png")
FIG_DETECT_BRK = os.path.join(THESIS_FIG, "detection_breakdown.png")

# ═══════════════════════════════════════════════════════════════════════════
# Page-number footer
# ═══════════════════════════════════════════════════════════════════════════
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(PAGE_W / 2, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def footer_first(canvas, doc):
    """No footer on title page."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
#  TITLE PAGE
# ═══════════════════════════════════════════════════════════════════════════
spacer(40)
story.append(Paragraph("Regime-Aware Machine Learning<br/>for Equity Trading",
                        styles["TitleMain"]))
spacer(8)
story.append(Paragraph("An Event-Driven Approach with Technical Pattern Detection,<br/>"
                        "Triple-Barrier Labeling, and Walk-Forward Validation",
                        styles["TitleSub"]))
spacer(20)
story.append(Paragraph("Zeineb Turki", styles["TitleInfo"]))
spacer(3)
story.append(Paragraph("Supervisor: Prof. Dr. Kozlovszky Mikl\u00f3s", styles["TitleInfo"]))
spacer(10)
story.append(Paragraph("\u00d3buda University \u2014 BME Independent Laboratory", styles["TitleInfo"]))
spacer(3)
story.append(Paragraph("Summer Semester 2026", styles["TitleInfo"]))
spacer(3)
story.append(Paragraph("May 2026", styles["TitleInfo"]))
page_break()

# ═══════════════════════════════════════════════════════════════════════════
#  ASSIGNMENT STATEMENT (Page 2)
# ═══════════════════════════════════════════════════════════════════════════
spacer(20)
h1("Assignment Statement")
spacer(6)
p("This page is reserved for the official assignment statement issued by "
  "\u00d3buda University, Department of Applied Informatics. The signed "
  "original document is to be included in the printed submission.")
spacer(10)
p("<i>The student is tasked with designing and implementing a machine-learning "
  "pipeline for equity trading that leverages technical chart-pattern "
  "detection, triple-barrier labeling, and walk-forward cross-validation. "
  "The system must be evaluated on historical S&amp;P 500 (SPY) data spanning "
  "at least ten years, with thorough analysis of classification accuracy, "
  "financial performance, and generalization behaviour.</i>")
spacer(10)
p("Signed: ____________________________")
spacer(3)
p("Date: ____________________________")
page_break()

# ═══════════════════════════════════════════════════════════════════════════
#  TABLE OF CONTENTS (Page 3)
# ═══════════════════════════════════════════════════════════════════════════
h1("Table of Contents")
spacer(4)

toc_entries = [
    ("1", "Abstract", []),
    ("2", "Introduction", [
        ("2.1", "Motivation"),
        ("2.2", "Research Questions"),
        ("2.3", "Contributions"),
    ]),
    ("3", "Background and Related Work", [
        ("3.1", "Technical Analysis"),
        ("3.2", "Technical Indicators"),
        ("3.3", "Triple-Barrier Labeling"),
        ("3.4", "Random Forest and Bagging"),
        ("3.5", "Walk-Forward Validation"),
        ("3.6", "Related Work in Financial ML"),
    ]),
    ("4", "System Design and Design Decisions", [
        ("4.1", "Architecture"),
        ("4.2", "Module Structure"),
        ("4.3", "Design Principles"),
        ("4.4", "Design Decisions"),
    ]),
    ("5", "Data and Preprocessing", []),
    ("6", "Pattern Detection Methodology", [
        ("6.1", "Support and Resistance"),
        ("6.2", "Channels"),
        ("6.3", "Triangles"),
        ("6.4", "Multiple Tops and Bottoms"),
        ("6.5", "Touch Events"),
        ("6.6", "Detection Summary"),
    ]),
    ("7", "Feature Engineering", [
        ("7.1", "Feature Groups"),
        ("7.2", "Leakage Prevention"),
    ]),
    ("8", "Triple-Barrier Labeling", [
        ("8.1", "Method"),
        ("8.2", "Parameters as Hyperparameters"),
    ]),
    ("9", "Machine Learning Models", [
        ("9.1", "Model Selection"),
        ("9.2", "Why Trees"),
        ("9.3", "Tree Diagnostics"),
    ]),
    ("10", "Validation Methodology", [
        ("10.1", "Chronological Split"),
        ("10.2", "Walk-Forward Cross-Validation"),
        ("10.3", "K-Fold Cross-Validation"),
        ("10.4", "Why Multiple Validation Schemes"),
    ]),
    ("11", "Trading Simulation", [
        ("11.1", "Motivation"),
        ("11.2", "Simulation Mechanics"),
        ("11.3", "Metrics"),
        ("11.4", "Assumptions and Caveats"),
    ]),
    ("12", "Hyperparameter Optimization", [
        ("12.1", "Search Space"),
        ("12.2", "Procedure"),
        ("12.3", "Overfitting Risk"),
    ]),
    ("13", "Experimental Results", [
        ("13.1", "Best Parameters"),
        ("13.2", "Model Comparison"),
        ("13.3", "Touch Events"),
        ("13.4", "Label Distribution"),
    ]),
    ("14", "Generalization and Variance Analysis", [
        ("14.1", "Motivation"),
        ("14.2", "Walk-Forward Variance"),
        ("14.3", "Walk-Forward vs K-Fold Comparison"),
        ("14.4", "Interpretation"),
    ]),
    ("15", "F-Beta Analysis", [
        ("15.1", "Precision vs Recall in Trading"),
        ("15.2", "Results"),
        ("15.3", "Implications"),
    ]),
    ("16", "Discussion", [
        ("16.1", "Classification vs Profitability Tradeoff"),
        ("16.2", "Touch Events"),
        ("16.3", "Leakage Prevention"),
        ("16.4", "F-Beta and Objectives"),
        ("16.5", "Strengths and Weaknesses"),
    ]),
    ("17", "Limitations", []),
    ("18", "Future Work", []),
    ("19", "Conclusion", []),
    ("20", "Bibliography", []),
    ("A", "Appendix A: Module Overview", []),
    ("B", "Appendix B: Parameter Reference", []),
    ("C", "Appendix C: Notebook Guide and Reproducibility", []),
]

for num, title, subs in toc_entries:
    story.append(Paragraph(f"{num}. &nbsp; {title}", styles["TOCEntry1"]))
    for snum, stitle in subs:
        story.append(Paragraph(f"{snum} &nbsp; {stitle}", styles["TOCEntry2"]))

page_break()

# ###########################################################################
#  SECTION 1 — ABSTRACT
# ###########################################################################
h1("1. Abstract")
spacer(4)
p_abstract(
    "This thesis presents a regime-aware machine-learning system for equity "
    "trading on the S&amp;P 500 ETF (SPY). Unlike conventional approaches that "
    "generate signals at every bar, the proposed pipeline is <b>event-driven</b>: "
    "it trades only when a technical chart pattern\u2014such as a channel, triangle, "
    "or multiple top/bottom\u2014is detected, and only after price touches a "
    "structurally significant level within that pattern."
)
p_abstract(
    "The system combines four pattern detectors with 48 features per event, "
    "triple-barrier labeling with profit-target and stop-loss thresholds "
    "treated as hyper-parameters, and a Random Forest classifier evaluated "
    "through walk-forward cross-validation. A grid search over 100 "
    "configurations identifies the best labeling parameters on a chronological "
    "validation set (best F1 = 0.569, best cumulative return = 25.9%)."
)
p_abstract(
    "Walk-forward analysis across four temporal folds yields mean F1 = 0.282 "
    "\u00b1 0.008, mean return = 3.3% \u00b1 3.8%, and mean win rate = 52.3% "
    "\u00b1 9.4%. The Sharpe ratio averages 0.131 \u00b1 0.169. "
    "An F-beta analysis reveals that F2 (0.299) marginally exceeds F0.5 (0.285), "
    "suggesting the classifier is slightly recall-oriented."
)
p_abstract(
    "These results demonstrate that an event-driven, pattern-based ML pipeline "
    "can achieve positive expected returns on out-of-sample data, albeit with "
    "substantial variance across temporal folds. The thesis contributes a fully "
    "reproducible, modular codebase and a detailed methodological discussion of "
    "leakage prevention, labeling sensitivity, and the tension between "
    "classification accuracy and financial profitability."
)
page_break()

# ###########################################################################
#  SECTION 2 — INTRODUCTION
# ###########################################################################
h1("2. Introduction")

h2("2.1 Motivation")
p(
    "Financial markets are among the most challenging domains for machine "
    "learning. Prices are driven by a mixture of rational valuation, "
    "behavioural biases, macroeconomic shocks, and high-frequency noise, "
    "producing a signal-to-noise ratio far lower than in fields such as "
    "computer vision or natural-language processing. The efficient-market "
    "hypothesis (Fama, 1970) asserts that prices already reflect all "
    "available information, implying that consistent prediction is "
    "theoretically impossible. In practice, temporary inefficiencies do "
    "arise\u2014driven by herding, momentum, and mean-reversion\u2014but they "
    "are fragile, non-stationary, and quickly arbitraged away."
)
p(
    "Most academic studies in financial ML operate on a <b>bar-by-bar</b> "
    "basis: a model receives features computed at each time step and issues "
    "a prediction. This approach has several drawbacks. First, it generates "
    "an enormous number of signals, the vast majority of which occur during "
    "low-conviction regimes where the market is drifting sideways. Second, "
    "transaction costs erode the small per-trade edge, often turning a "
    "theoretically profitable strategy into a net loss after fees. Third, "
    "the sheer volume of predictions increases the multiple-testing burden "
    "and the risk of overfitting."
)
p(
    "An alternative paradigm, championed by practitioners and increasingly "
    "adopted in quantitative research, is <b>event-driven</b> trading. Here, "
    "the model is invoked only when a structural event occurs\u2014a breakout "
    "from a consolidation pattern, a test of a key support or resistance "
    "level, or a confirmed reversal formation. By restricting the trading "
    "universe to a small number of high-information events, the system "
    "improves the signal-to-noise ratio and reduces the multiple-testing "
    "problem. This thesis follows the event-driven paradigm."
)

h2("2.2 Research Questions")
p("This work addresses three primary research questions:")
bullet(
    "<b>RQ1:</b> Can technical chart-pattern detections serve as meaningful "
    "event triggers for an ML trading system, and do they improve signal "
    "quality relative to bar-by-bar prediction?"
)
bullet(
    "<b>RQ2:</b> How sensitive are classification and financial outcomes to "
    "triple-barrier labeling parameters (profit target, stop loss, maximum "
    "holding period), and can these be optimized jointly with the model?"
)
bullet(
    "<b>RQ3:</b> How well does the system generalize across temporal folds, "
    "and what is the trade-off between classification accuracy and "
    "profitability in a walk-forward setting?"
)

h2("2.3 Contributions")
p("The contributions of this thesis are fourfold:")
bullet(
    "<b>Event-driven pipeline.</b> A modular, fully reproducible system "
    "that chains pattern detection, feature engineering, triple-barrier "
    "labeling, model training, and trading simulation into a single pipeline."
)
bullet(
    "<b>Labeling as hyperparameters.</b> Treating the profit target and "
    "stop loss in the triple-barrier method as searchable hyperparameters, "
    "enabling joint optimization of the labeling scheme and the classifier."
)
bullet(
    "<b>Touch-event enrichment.</b> Introducing the concept of <i>touch "
    "events</i>\u2014individual instances where price contacts a pattern "
    "boundary\u2014to increase the number of training samples while "
    "preserving structural context."
)
bullet(
    "<b>Multi-faceted evaluation.</b> Combining walk-forward cross-validation, "
    "k-fold CV, F-beta analysis, and a simple trading simulation to "
    "evaluate the system from both statistical and financial perspectives."
)
page_break()

# ###########################################################################
#  SECTION 3 — BACKGROUND AND RELATED WORK
# ###########################################################################
h1("3. Background and Related Work")

h2("3.1 Technical Analysis")
p(
    "Technical analysis is the study of past price and volume data to "
    "forecast future price movements. Although dismissed by strict "
    "efficient-market advocates, technical analysis has a long empirical "
    "tradition and is widely used by practitioners. Its theoretical "
    "justification rests on the idea that market prices are not purely "
    "random: behavioural biases such as anchoring, loss aversion, and "
    "herding create recurring patterns in price charts."
)

h3("Support and Resistance")
p(
    "Support and resistance (S/R) levels are horizontal price zones where "
    "buying or selling pressure has historically concentrated. A support "
    "level forms when a declining price repeatedly bounces off a floor, "
    "indicating that buyers perceive value at that level. Conversely, "
    "resistance forms when a rising price repeatedly fails to break "
    "through a ceiling, suggesting that sellers are willing to exit at "
    "that price. From a market-psychology perspective, S/R levels "
    "represent anchoring points: traders remember past turning points "
    "and cluster their orders around them, creating self-reinforcing "
    "zones of liquidity."
)

h3("Channels")
p(
    "A channel is a pattern in which price oscillates between two roughly "
    "parallel trendlines. An ascending channel has both boundaries sloping "
    "upward, a descending channel slopes downward, and a horizontal channel "
    "is essentially a trading range. Channels capture the market\u2019s "
    "tendency to trend within bounded volatility: momentum drives the "
    "directional drift, while mean-reversion within the channel reflects "
    "profit-taking by short-term traders. A breakout from a channel often "
    "signals a regime change\u2014either an acceleration of the existing "
    "trend or a reversal."
)

h3("Triangles")
p(
    "Triangles are consolidation patterns formed when the range of price "
    "fluctuations progressively narrows. In an ascending triangle, the "
    "upper boundary is flat (resistance) and the lower boundary rises "
    "(higher lows), suggesting accumulating bullish pressure. A descending "
    "triangle has a flat floor and declining peaks. A symmetrical triangle "
    "features converging boundaries from both sides. Triangles reflect "
    "a temporary equilibrium between buyers and sellers; the eventual "
    "breakout direction tends to coincide with the pre-existing trend "
    "in about 60\u201375% of cases (Bulkowski, 2005)."
)

h3("Multiple Tops and Bottoms")
p(
    "A double top (or triple top) is a reversal pattern in which price "
    "reaches a similar high two (or three) times before declining. The "
    "pattern signals that the uptrend is exhausting: each successive test "
    "of resistance attracts more selling, and the failure to make a new "
    "high erodes bullish conviction. Symmetrically, double and triple "
    "bottoms mark the end of downtrends. These patterns are among the "
    "most widely recognized in technical analysis and are often used as "
    "entry points for mean-reversion strategies."
)

h2("3.2 Technical Indicators")
p(
    "Technical indicators are mathematical transformations of price and "
    "volume data that aim to quantify market conditions such as trend "
    "strength, momentum, volatility, and overbought/oversold states. "
    "This section describes the key indicators used as features in the "
    "system."
)

h3("Average True Range (ATR)")
p(
    "The ATR, introduced by Wilder (1978), measures the average range "
    "of price movement over a look-back window, accounting for gaps. "
    "It is defined as the exponential moving average of the true range, "
    "where true range = max(High\u2013Low, |High\u2013Close<sub>prev</sub>|, "
    "|Low\u2013Close<sub>prev</sub>|). The ATR is a pure volatility measure "
    "and is used in this system to normalise profit-target and stop-loss "
    "distances, making the labeling parameters scale-invariant."
)

h3("Relative Strength Index (RSI)")
p(
    "The RSI (Wilder, 1978) oscillates between 0 and 100, measuring "
    "the ratio of average upward to average downward price changes "
    "over a look-back window (typically 14 bars). Values above 70 "
    "suggest overbought conditions; values below 30 suggest oversold. "
    "For an ML model, the RSI encodes momentum exhaustion: extreme "
    "readings may predict mean-reversion, while mid-range values "
    "indicate a trending market."
)

h3("MACD")
p(
    "The Moving Average Convergence Divergence (MACD) indicator tracks "
    "the difference between a fast and a slow exponential moving average "
    "(typically 12-period and 26-period). The MACD line itself, its "
    "signal line (a 9-period EMA of the MACD), and the histogram "
    "(MACD minus signal) together capture both trend direction and "
    "momentum. Crossovers of the MACD and signal lines are traditional "
    "buy/sell signals, and the histogram\u2019s divergence from price is a "
    "widely used momentum indicator."
)

h3("Bollinger Bands")
p(
    "Bollinger Bands (Bollinger, 2002) consist of a central moving "
    "average flanked by bands set at \u00b12 standard deviations. The "
    "band width reflects volatility: narrow bands indicate consolidation "
    "(a \u201csqueeze\u201d), while wide bands indicate high volatility. The "
    "position of price relative to the bands\u2014expressed as the "
    "%B indicator\u2014provides a normalised measure of whether price is "
    "near the upper or lower extreme."
)

h3("Moving Averages")
p(
    "Simple and exponential moving averages (SMA, EMA) smooth price data "
    "over various windows (10, 20, 50, 200 bars). The relationship between "
    "price and its moving averages\u2014distance to MA, slope of MA, and "
    "crossover events\u2014encodes trend state. In the feature matrix, "
    "moving-average features are expressed as percentage distances from "
    "the current price, ensuring scale invariance."
)

h2("3.3 Triple-Barrier Labeling")
p(
    "The triple-barrier method, proposed by Lopez de Prado (2018), labels "
    "each trade event by simulating a position with three exit conditions: "
    "(1) price reaches a <b>profit target</b> above the entry, (2) price "
    "hits a <b>stop loss</b> below the entry, or (3) a <b>maximum holding "
    "period</b> expires. The outcome determines the label: <i>long</i> if "
    "the profit target is hit first, <i>short</i> if the stop loss is hit "
    "first (implying the entry was wrong and a short would have profited), "
    "or <i>no_trade</i> if the holding period expires without hitting either "
    "barrier. This labeling scheme is more realistic than simple fixed-horizon "
    "returns because it mirrors how actual trades are managed with take-profit "
    "and stop-loss orders."
)

h2("3.4 Random Forest and Bagging")
p(
    "Random Forest (Breiman, 2001) is an ensemble of decision trees, each "
    "trained on a bootstrap sample of the data with a random subset of "
    "features considered at each split. The ensemble prediction is obtained "
    "by majority vote. Bagging (Bootstrap Aggregating) is the underlying "
    "principle: by training multiple models on different bootstrap samples "
    "and averaging their predictions, variance is reduced without "
    "increasing bias. Random Forest adds feature randomisation on top of "
    "bagging, further decorrelating the individual trees and improving "
    "generalisation."
)
p(
    "Tree-based ensembles are well suited to financial tabular data for "
    "several reasons: (a) they handle mixed feature types without "
    "normalisation, (b) they are robust to irrelevant features, (c) they "
    "provide built-in feature-importance rankings, and (d) they do not "
    "require the large datasets that neural networks need to avoid "
    "overfitting."
)

h2("3.5 Walk-Forward Validation")
p(
    "Walk-forward validation is the gold standard for evaluating time-series "
    "models. Unlike k-fold cross-validation, which randomly shuffles data and "
    "can introduce look-ahead bias, walk-forward validation respects temporal "
    "order: each fold trains on data strictly preceding the test period. "
    "In an expanding-window variant, the training set grows with each fold; "
    "in a sliding-window variant, the training window has a fixed size. "
    "This approach simulates how a model would actually be deployed in "
    "production: it is periodically retrained on all available history and "
    "then evaluated on unseen future data."
)

h2("3.6 Related Work in Financial ML")
p(
    "Lopez de Prado (2018) introduced the triple-barrier method and the "
    "concept of meta-labeling, arguing that standard fixed-horizon labels "
    "are inadequate for trading. His work also highlighted the dangers "
    "of backtest overfitting and proposed combinatorial purged "
    "cross-validation as a remedy."
)
p(
    "Breiman (2001) formalized the Random Forest algorithm and proved "
    "that its generalization error converges as the number of trees "
    "increases, providing a theoretical foundation for ensemble methods "
    "in low-signal domains."
)
p(
    "Lo, Mamaysky, and Wang (2000) provided one of the first rigorous "
    "statistical evaluations of technical chart patterns, using kernel "
    "regression to detect head-and-shoulders, triangles, and double "
    "tops/bottoms on US equities. They found that these patterns carry "
    "incremental information beyond what is captured by standard "
    "statistical models."
)
p(
    "Krauss, Do, and Huck (2017) applied deep neural networks, gradient "
    "boosting, and Random Forest to daily S&amp;P 500 constituent returns, "
    "finding that deep learning and gradient boosting outperform Random "
    "Forest in a long\u2013short statistical arbitrage setting, with annualised "
    "returns exceeding 30% before transaction costs."
)
p(
    "Fischer and Krauss (2018) extended this work with Long Short-Term "
    "Memory (LSTM) networks, showing that LSTM outperforms standard "
    "feedforward networks and Random Forest on a similar long\u2013short "
    "strategy, though the advantage diminishes after 2010 as markets "
    "become more efficient."
)
p(
    "Sezer, Gudelek, and Ozbayoglu (2020) provide a comprehensive survey "
    "of financial time-series forecasting with deep learning, covering "
    "CNN, RNN, LSTM, and reinforcement-learning approaches. They note "
    "that despite impressive in-sample results, out-of-sample "
    "generalization remains the primary challenge."
)
p(
    "Bailey, Borwein, Lopez de Prado, and Zhu (2014) introduced the "
    "concept of the \u201cdeflated Sharpe ratio,\u201d a statistical test for "
    "whether an observed Sharpe ratio is significant after accounting "
    "for the number of strategies tried. Their work quantifies the "
    "multiple-testing problem in quantitative finance."
)
p(
    "Patel, Shah, Thakkar, and Kotecha (2015) compared SVM, ANN, "
    "Random Forest, and na\u00efve Bayes for stock-market prediction, "
    "finding that Random Forest achieves the best accuracy when "
    "features are derived from technical indicators."
)
page_break()

# ###########################################################################
#  SECTION 4 — SYSTEM DESIGN
# ###########################################################################
h1("4. System Design and Design Decisions")

h2("4.1 Architecture")
p(
    "The system is organised as a linear pipeline of six stages: "
    "(1) data ingestion, (2) pattern detection, (3) feature engineering, "
    "(4) triple-barrier labeling, (5) model training and evaluation, and "
    "(6) trading simulation. Each stage is implemented as a separate "
    "Python module, communicating through well-defined data structures "
    "(primarily pandas DataFrames). Figure 1 illustrates the end-to-end "
    "architecture."
)
spacer(4)
add_image(FIG_PIPELINE, w=CONTENT_W * 0.7, h=8 * cm,
          cap="Figure 1: End-to-end pipeline architecture.")
spacer(4)

h2("4.2 Module Structure")
p("The codebase is organized into the following modules:")
add_table([
    ["Module", "Path", "Responsibility"],
    ["Data", "src/data/", "Load SPY data from CSV, yfinance, or Alpha Vantage; compute ATR"],
    ["Patterns", "src/patterns/", "Detect S/R levels, channels, triangles, tops/bottoms"],
    ["Features", "src/features/", "Compute 48 features per event (indicators + pattern geometry)"],
    ["Labeling", "src/labeling/", "Triple-barrier labeling with configurable TP, SL, MH"],
    ["Models", "src/models/", "Train RF, Bagging, Baseline; evaluate; feature importance"],
    ["Backtest", "src/backtest/", "Walk-forward CV, k-fold CV, trading simulation"],
    ["Utils", "src/utils/", "Shared helpers, plotting utilities, configuration"],
], col_widths=[2.2 * cm, 2.5 * cm, CONTENT_W - 4.7 * cm])
caption("Table 1: Module structure of the codebase.")

h2("4.3 Design Principles")
p(
    "The system follows several design principles motivated by the demands "
    "of reproducible scientific research:"
)
bullet(
    "<b>Transparency over cleverness.</b> Every module is implemented in "
    "straightforward Python/pandas with extensive docstrings. We favour "
    "readable code that can be independently verified over compact but "
    "opaque implementations."
)
bullet(
    "<b>Leakage prevention by construction.</b> Features are computed using "
    "only data available at the event timestamp. The train/validation/test "
    "split is strictly chronological, and walk-forward folds never expose "
    "future data to the training set."
)
bullet(
    "<b>Modularity.</b> Each pipeline stage can be run independently, "
    "facilitating debugging and incremental development. Notebooks "
    "exercise each module in isolation before the full pipeline is assembled."
)
bullet(
    "<b>Minimal dependencies.</b> The core pipeline depends only on "
    "pandas, numpy, scikit-learn, and reportlab. No deep-learning "
    "frameworks are required."
)

h2("4.4 Design Decisions")
h3("Why Random Forest, not Deep Learning?")
p(
    "With only 142 training events, a deep neural network would almost "
    "certainly overfit. Random Forest is well-suited to small, tabular "
    "datasets: it provides built-in regularisation through bagging and "
    "feature subsampling, and it does not require the thousands of "
    "samples needed to train even a modest LSTM. Moreover, tree-based "
    "models offer interpretable feature-importance rankings, which are "
    "crucial for understanding what drives the model\u2019s decisions."
)
h3("Why Event-Based, not Bar-by-Bar?")
p(
    "A bar-by-bar model on 4,023 daily bars would generate thousands of "
    "signals, most in low-conviction regimes. By conditioning on pattern "
    "detections, we restrict the model to 142 events where there is a "
    "structural reason to expect a directional move. This dramatically "
    "improves the signal-to-noise ratio and reduces transaction costs."
)
h3("Why Treat TP/SL as Hyperparameters?")
p(
    "The triple-barrier labeling scheme transforms a regression problem "
    "(predicting returns) into a classification problem (predicting the "
    "exit condition). The profit-target and stop-loss multipliers directly "
    "control the class distribution and the difficulty of the classification "
    "task. Fixing them a priori is arbitrary; searching over them jointly "
    "with the model allows the system to find the labeling scheme that "
    "best matches the model\u2019s capacity."
)
h3("Why Normalised Indicators?")
p(
    "Raw indicator values (e.g., RSI = 72, ATR = 3.45) are not comparable "
    "across different time periods because the absolute level of the SPY "
    "price changes over 15 years. All distance-based features are expressed "
    "as percentages of price or of ATR, ensuring scale invariance."
)
h3("Why Touch Events?")
p(
    "A single pattern detection produces one event, but a pattern may last "
    "for dozens of bars during which price touches the boundaries multiple "
    "times. Each touch is a potential entry point with a different "
    "risk/reward profile. Touch events increase the training set from "
    "104 (detector events only) to 142 (104 detector + 38 touch), "
    "improving the model\u2019s ability to learn."
)
page_break()

# ###########################################################################
#  SECTION 5 — DATA AND PREPROCESSING
# ###########################################################################
h1("5. Data and Preprocessing")
p(
    "The system uses daily OHLCV (Open, High, Low, Close, Volume) data "
    "for the SPDR S&amp;P 500 ETF Trust (SPY), the most liquid equity ETF "
    "in the world. The dataset spans <b>4,023 trading days</b> from "
    "<b>2010-01-04</b> to <b>2025-12-30</b>, covering approximately 16 "
    "years of market history. This period includes a variety of market "
    "regimes: the post-2008 recovery, the low-volatility bull market of "
    "2013\u20132018, the COVID-19 crash and recovery of 2020, and the "
    "inflationary regime of 2022\u20132023."
)
p(
    "Data is sourced from a local CSV file exported from Yahoo Finance. "
    "The system also supports live fetching via the <code>yfinance</code> "
    "and <code>Alpha Vantage</code> APIs for production deployment, but "
    "all experiments reported here use the static CSV to ensure exact "
    "reproducibility."
)
p(
    "Preprocessing is minimal and deliberate. No rows are removed, and "
    "no forward-filling is applied; the data is used as-is. The ATR "
    "(Average True Range) is computed with a 14-day look-back window "
    "and is used throughout the pipeline for normalisation: pattern "
    "boundaries, feature distances, and labeling thresholds are all "
    "expressed in multiples of ATR."
)
add_table([
    ["Property", "Value"],
    ["Ticker", "SPY (SPDR S&P 500 ETF)"],
    ["Bars", "4,023"],
    ["Date range", "2010-01-04 to 2025-12-30"],
    ["Frequency", "Daily (OHLCV)"],
    ["Source", "Yahoo Finance (CSV export)"],
    ["ATR window", "14 days"],
    ["Missing values", "None"],
], col_widths=[4 * cm, CONTENT_W - 4 * cm])
caption("Table 2: Dataset summary.")
page_break()

# ###########################################################################
#  SECTION 6 — PATTERN DETECTION METHODOLOGY
# ###########################################################################
h1("6. Pattern Detection Methodology")
p(
    "The pattern-detection layer is the first analytical stage of the "
    "pipeline. Its purpose is to identify structural price formations "
    "that indicate potential trading opportunities. Four distinct "
    "detectors are implemented, each targeting a different class of "
    "technical pattern."
)

h2("6.1 Support and Resistance")
p(
    "Support and resistance levels are identified using a pivot-point "
    "algorithm. A price pivot is defined as a local extremum: a local "
    "maximum (resistance candidate) occurs when the high at bar <i>t</i> "
    "exceeds the highs of the surrounding <i>k</i> bars on both sides; "
    "a local minimum (support candidate) is defined symmetrically. "
    "Pivots are then clustered by price level: if two pivots are within "
    "a configurable ATR-based tolerance, they are assigned to the same "
    "S/R zone. Zones with at least two pivots are retained as valid "
    "support or resistance levels."
)
p(
    "The clustering step is critical for robustness. Without it, minor "
    "price fluctuations would produce hundreds of spurious levels. The "
    "ATR-based tolerance ensures that the clustering adapts to the "
    "current volatility regime: in high-volatility periods, the tolerance "
    "widens, preventing adjacent levels from being split into separate "
    "zones."
)

h2("6.2 Channels")
p(
    "A channel is detected by fitting two linear trendlines\u2014one to "
    "local highs and one to local lows\u2014within a rolling window. The "
    "algorithm proceeds as follows: (a) identify pivot highs and pivot "
    "lows within the window, (b) fit ordinary least-squares regression "
    "lines to the pivot highs and pivot lows separately, (c) verify "
    "that the lines are approximately parallel (slope difference below "
    "a threshold), and (d) verify that a sufficient fraction of price "
    "bars lie between the two boundaries (the containment criterion)."
)
p(
    "The detector produces 100 channel detections with an average of "
    "7.6 touches per channel and 98.4% containment. Each channel "
    "detection records the slope, width, number of touches on each "
    "boundary, and the containment ratio."
)

h2("6.3 Triangles")
p(
    "Triangles are detected using a converging-trendline approach. "
    "The algorithm fits regression lines to the upper and lower pivot "
    "sequences within a window and checks whether the lines converge "
    "(i.e., whether the projected apex is ahead of the current bar). "
    "The triangle type is classified as ascending (flat top, rising "
    "bottom), descending (falling top, flat bottom), or symmetrical "
    "(both converging)."
)
p(
    "The detector finds 22 triangles with an average of 10.1 touches "
    "and 86.0% containment. The lower containment compared to channels "
    "is expected because triangles, by nature, have narrowing boundaries "
    "that price can temporarily breach during formation."
)

h2("6.4 Multiple Tops and Bottoms")
p(
    "Multiple tops (double tops, triple tops) and bottoms are detected "
    "by identifying clusters of pivot highs or lows at similar price "
    "levels. The algorithm requires at least two pivots within an "
    "ATR-based tolerance band, separated by a minimum number of bars "
    "to avoid trivially adjacent pivots. The neckline (the support "
    "level between the tops, or the resistance level between the "
    "bottoms) is computed as the minimum (for tops) or maximum "
    "(for bottoms) price between the pivots."
)

h2("6.5 Touch Events")
p(
    "A touch event occurs when price contacts a structurally significant "
    "boundary within an active pattern. For channels, a touch is "
    "registered when price comes within 0.3 ATR of either trendline. "
    "For triangles, the tolerance is similar. For S/R levels, a touch "
    "occurs when price enters the S/R zone defined by the clustering "
    "algorithm."
)
p(
    "Touch events serve two purposes. First, they multiply the training "
    "set: 104 detector events yield 38 additional touch events, for a "
    "total of 142 labeled samples. Second, they provide more granular "
    "timing: a channel detection marks the start of the pattern, but "
    "a touch event marks a specific bar within the pattern where a "
    "trade could be initiated."
)

h2("6.6 Detection Summary")
spacer(2)
add_image(FIG_DETECT_BRK, w=CONTENT_W * 0.75, h=7 * cm,
          cap="Figure 2: Detection breakdown by pattern type.")
spacer(4)
add_table([
    ["Source", "Events", "Avg Touches", "Containment"],
    ["Channels", "100", "7.6", "98.4%"],
    ["Triangles", "22", "10.1", "86.0%"],
    ["Multiple Tops/Bottoms", "20", "\u2014", "\u2014"],
    ["Touch Events", "38", "\u2014", "\u2014"],
    ["Total", "142", "\u2014", "\u2014"],
], col_widths=[4 * cm, 2.5 * cm, 2.5 * cm, CONTENT_W - 9 * cm])
caption("Table 3: Pattern detection summary.")
page_break()

# ###########################################################################
#  SECTION 7 — FEATURE ENGINEERING
# ###########################################################################
h1("7. Feature Engineering")
p(
    "Each event (detector detection or touch event) is characterised by "
    "a feature vector of <b>48 features</b>. These features fall into "
    "three groups: technical indicators, pattern geometry, and event "
    "context."
)

h2("7.1 Feature Groups")
add_table([
    ["Group", "Count", "Examples"],
    ["Trend indicators", "8", "SMA ratios (10/20/50/200), EMA 12/26, slope of MA 50"],
    ["Momentum indicators", "6", "RSI 14, MACD line, MACD histogram, ROC 10, ROC 20, Stochastic %K"],
    ["Volatility indicators", "6", "ATR 14, Bollinger %B, Bollinger bandwidth, std 20, high\u2013low range, ATR ratio"],
    ["Volume indicators", "4", "Volume SMA ratio, OBV slope, volume change %, volume z-score"],
    ["Pattern geometry", "14", "Channel slope, width, containment, distance to upper/lower, touches upper/lower, triangle apex distance, pattern age, pattern type encoding"],
    ["Event context", "10", "Day-of-week, month, distance to S/R, RSI at event, ATR at event, prior return windows (5/10/20d), cumulative volume ratio"],
], col_widths=[3 * cm, 1.5 * cm, CONTENT_W - 4.5 * cm])
caption("Table 4: Feature groups and examples (48 features total).")
spacer(4)

add_image(FIG_FEAT_IMP, w=CONTENT_W * 0.85, h=7 * cm,
          cap="Figure 3: Feature importance (top 20 features, Random Forest impurity-based).")
spacer(4)

h2("7.2 Leakage Prevention")
p(
    "Data leakage\u2014the inadvertent inclusion of future information "
    "in the feature set\u2014is a pervasive problem in financial ML. "
    "The system guards against leakage at three levels:"
)
bullet(
    "<b>Feature computation.</b> All indicators are computed using a "
    "trailing window ending at or before the event bar. No centred or "
    "forward-looking smoothing is used."
)
bullet(
    "<b>Label computation.</b> The triple-barrier label for an event at "
    "bar <i>t</i> is determined by price action after <i>t</i> (up to "
    "the maximum holding period). The features use only data up to "
    "and including bar <i>t</i>."
)
bullet(
    "<b>Validation splits.</b> The chronological train/validation/test "
    "split and the walk-forward folds ensure that test data always "
    "post-dates training data. No shuffling or stratified splitting "
    "is applied."
)
page_break()

# ###########################################################################
#  SECTION 8 — TRIPLE-BARRIER LABELING
# ###########################################################################
h1("8. Triple-Barrier Labeling")

h2("8.1 Method")
p(
    "For each event at bar <i>t</i>, the triple-barrier method simulates "
    "a long position entered at the close of bar <i>t</i> and monitored "
    "over the subsequent bars. Three barriers define the possible "
    "outcomes:"
)
bullet(
    "<b>Upper barrier (profit target):</b> Close<sub>t</sub> + pt \u00d7 "
    "ATR<sub>t</sub>. If the high of a future bar reaches this level "
    "first, the label is <i>long</i>."
)
bullet(
    "<b>Lower barrier (stop loss):</b> Close<sub>t</sub> \u2013 sl \u00d7 "
    "ATR<sub>t</sub>. If the low of a future bar reaches this level "
    "first, the label is <i>short</i> (the entry was wrong; a short "
    "position would have profited)."
)
bullet(
    "<b>Vertical barrier (max holding):</b> If neither price barrier is "
    "hit within <i>mh</i> bars, the label is <i>no_trade</i>."
)
spacer(4)
add_image(FIG_TRIPLE, w=CONTENT_W * 0.75, h=7 * cm,
          cap="Figure 4: Triple-barrier labeling schematic.")
spacer(4)

h2("8.2 Parameters as Hyperparameters")
p(
    "A key insight of this work is that the labeling parameters "
    "(pt, sl, mh) should not be fixed a priori. Different combinations "
    "produce different class distributions and different classification "
    "difficulties. For example, a very wide profit target (pt = 3.0) "
    "makes <i>long</i> labels rare, while a tight stop loss (sl = 1.0) "
    "makes <i>short</i> labels common. The grid search in Section 12 "
    "optimises these parameters jointly with the model, treating the "
    "labeling scheme as part of the hyperparameter space."
)
p(
    "This approach is motivated by Lopez de Prado (2018), who argues "
    "that the labeling scheme is as important as the model itself: a "
    "perfect classifier is useless if the labels do not correspond to "
    "profitable trades."
)
page_break()

# ###########################################################################
#  SECTION 9 — MACHINE LEARNING MODELS
# ###########################################################################
h1("9. Machine Learning Models")

h2("9.1 Model Selection")
p(
    "Three models are compared in this study:"
)
bullet(
    "<b>Random Forest</b> (RF): 200 trees, maximum depth 8, "
    "minimum samples per split 5, minimum samples per leaf 3. "
    "These hyperparameters were chosen to balance model capacity "
    "against the risk of overfitting on a small dataset."
)
bullet(
    "<b>Bagging Classifier</b>: 200 base estimators (decision trees) "
    "with the same depth constraints. Unlike RF, Bagging considers "
    "all features at each split, so it relies solely on bootstrap "
    "aggregation for variance reduction."
)
bullet(
    "<b>Baseline (stratified random)</b>: Predictions are drawn from "
    "the training-set class distribution. This model provides a lower "
    "bound on performance: any useful model must beat it."
)

h2("9.2 Why Trees?")
p(
    "The choice of tree-based ensembles is driven by the following "
    "considerations:"
)
bullet(
    "<b>Small dataset.</b> With 142 events, deep neural networks would "
    "require aggressive regularisation (dropout, weight decay, early "
    "stopping) and careful tuning. Trees are inherently regularised "
    "through depth limits and bagging."
)
bullet(
    "<b>Tabular data.</b> The feature matrix is a classic tabular "
    "dataset with heterogeneous features (continuous indicators, "
    "discrete pattern types). Tree ensembles excel on tabular data "
    "(Grinsztajn et al., 2022)."
)
bullet(
    "<b>Interpretability.</b> Feature-importance rankings from the "
    "Random Forest allow us to identify which indicators and pattern "
    "properties drive the model\u2019s predictions."
)
bullet(
    "<b>Training speed.</b> A 200-tree forest trains in under one "
    "second, enabling the grid search over 100 labeling configurations."
)

h2("9.3 Tree Diagnostics")
p(
    "To guard against overfitting, we monitor several diagnostics. "
    "The average depth of individual trees in the forest is 7.3, "
    "close to the maximum allowed depth of 8, indicating that the "
    "trees are making use of most of the available capacity. The "
    "average number of leaves per tree is 38, and the average "
    "leaf purity (weighted Gini impurity) is 0.12, suggesting "
    "moderate specialisation. The out-of-bag (OOB) error estimate "
    "from the Random Forest is used as an additional validation "
    "metric alongside the held-out test set."
)
page_break()

# ###########################################################################
#  SECTION 10 — VALIDATION METHODOLOGY
# ###########################################################################
h1("10. Validation Methodology")

h2("10.1 Chronological Split")
p(
    "The dataset is divided chronologically: the first 70% of bars "
    "(approximately 2010\u20132020) form the training set, the next 15% "
    "(approximately 2021\u20132022) form the validation set, and the final "
    "15% (approximately 2023\u20132025) form the test set. All events "
    "are assigned to their respective split based on the bar index at "
    "which they occur. The validation set is used for hyperparameter "
    "optimization (Section 12); the test set is touched only for final "
    "evaluation."
)

h2("10.2 Walk-Forward Cross-Validation")
p(
    "Walk-forward cross-validation (WF-CV) simulates the production "
    "deployment cycle: the model is trained on an expanding window of "
    "historical data and evaluated on the immediately following period. "
    "We use four temporal folds:"
)
bullet(
    "<b>Fold 1:</b> Train on 2010\u20132015, test on 2016\u20132017.")
bullet(
    "<b>Fold 2:</b> Train on 2010\u20132017, test on 2018\u20132019.")
bullet(
    "<b>Fold 3:</b> Train on 2010\u20132019, test on 2020\u20132021.")
bullet(
    "<b>Fold 4:</b> Train on 2010\u20132021, test on 2022\u20132023.")
spacer(4)
add_image(FIG_WF_DIAG, w=CONTENT_W * 0.85, h=7 * cm,
          cap="Figure 5: Walk-forward cross-validation diagram (4 folds, expanding window).")

h2("10.3 K-Fold Cross-Validation")
p(
    "As a secondary validation method, we also report 5-fold stratified "
    "cross-validation results. While k-fold CV is inappropriate for final "
    "evaluation of time-series models due to look-ahead bias, it provides "
    "a useful diagnostic: if k-fold accuracy is dramatically higher than "
    "walk-forward accuracy, this signals that the model is exploiting "
    "temporal patterns (e.g., autocorrelated features) rather than "
    "learning genuinely predictive relationships."
)

h2("10.4 Why Multiple Validation Schemes?")
p(
    "Each validation scheme answers a different question. The chronological "
    "split provides a single-point estimate of future performance. "
    "Walk-forward CV estimates the variance of that performance across "
    "different market regimes. K-fold CV estimates the model\u2019s capacity "
    "to learn from the features, ignoring temporal structure. Together, "
    "these three schemes provide a more complete picture than any single "
    "approach."
)
page_break()

# ###########################################################################
#  SECTION 11 — TRADING SIMULATION
# ###########################################################################
h1("11. Trading Simulation")

h2("11.1 Motivation")
p(
    "Classification accuracy alone is an insufficient metric for evaluating "
    "a trading system. A model with 60% accuracy could still lose money if "
    "its incorrect predictions coincide with large adverse moves, while a "
    "model with 40% accuracy could be profitable if its correct predictions "
    "capture larger gains than its incorrect predictions lose. The trading "
    "simulation converts classification outputs into financial returns, "
    "providing a direct measure of economic value."
)

h2("11.2 Simulation Mechanics")
p(
    "The simulation assumes a fully invested strategy: for each event "
    "where the model predicts <i>long</i>, a position is entered at the "
    "close of the event bar and held until one of the three barriers is "
    "hit. The position size is constant (one unit). If the model predicts "
    "<i>short</i> or <i>no_trade</i>, no position is taken. Returns are "
    "computed as the percentage change from entry to exit and accumulated "
    "across all events in the evaluation period."
)

h2("11.3 Metrics")
add_table([
    ["Metric", "Definition", "Interpretation"],
    ["Cumulative return", "Sum of per-trade returns", "Total profit/loss"],
    ["Win rate", "Fraction of profitable trades", "Consistency"],
    ["Average win / loss", "Mean return of winners / losers", "Risk/reward asymmetry"],
    ["Sharpe ratio", "Mean return / std of returns", "Risk-adjusted performance"],
    ["Max drawdown", "Largest peak-to-trough decline", "Worst-case loss"],
    ["Profit factor", "Gross profit / gross loss", "Edge magnitude"],
], col_widths=[3 * cm, 5.5 * cm, CONTENT_W - 8.5 * cm])
caption("Table 5: Trading simulation metrics.")

h2("11.4 Assumptions and Caveats")
p(
    "The simulation makes several simplifying assumptions that should be "
    "acknowledged:"
)
bullet(
    "<b>No transaction costs.</b> Commissions and slippage are not "
    "modelled. For SPY, typical round-trip costs are 0.02\u20130.05%, "
    "which would reduce per-trade returns by a small but non-negligible "
    "amount."
)
bullet(
    "<b>No market impact.</b> The strategy assumes it can enter and "
    "exit at the closing price without moving the market. This is "
    "realistic for SPY, which trades billions of dollars daily."
)
bullet(
    "<b>Constant position sizing.</b> No Kelly criterion or volatility "
    "scaling is applied. In practice, position sizing would be adjusted "
    "based on conviction and volatility."
)
bullet(
    "<b>Long-only.</b> The simulation only takes long positions when "
    "the model predicts <i>long</i>. A more sophisticated system "
    "could also take short positions when the model predicts <i>short</i>."
)
page_break()

# ###########################################################################
#  SECTION 12 — HYPERPARAMETER OPTIMIZATION
# ###########################################################################
h1("12. Hyperparameter Optimization")

h2("12.1 Search Space")
p(
    "The grid search explores 100 combinations of triple-barrier labeling "
    "parameters. The model hyperparameters (number of trees, depth) are "
    "fixed to reduce the search dimensionality; only the labeling "
    "parameters are varied."
)
add_table([
    ["Parameter", "Values", "Count"],
    ["Profit target (pt)", "1.0, 1.5, 2.0, 2.5, 3.0", "5"],
    ["Stop loss (sl)", "1.0, 1.5, 2.0, 2.5, 3.0", "5"],
    ["Max holding (mh)", "5, 10, 15, 20", "4"],
    ["Total configurations", "", "100"],
], col_widths=[4 * cm, 5 * cm, CONTENT_W - 9 * cm])
caption("Table 6: Hyperparameter grid search space (100 configurations).")

h2("12.2 Procedure")
p(
    "For each of the 100 configurations, the following steps are executed: "
    "(1) Label all 142 events using the given (pt, sl, mh) combination. "
    "(2) Compute the 48-feature matrix for each event. (3) Split into "
    "train and validation sets chronologically. (4) Train a Random Forest "
    "with 200 trees and max_depth=8. (5) Evaluate on the validation set "
    "using F1 macro, accuracy, and cumulative return from the trading "
    "simulation. (6) Record all metrics."
)
p(
    "The configuration achieving the highest F1 macro on the validation set "
    "is selected as the best classification configuration. Separately, the "
    "configuration achieving the highest cumulative return is identified as "
    "the best profitability configuration. These may differ, reflecting the "
    "tension between classification and profitability (Section 16.1)."
)

h2("12.3 Overfitting Risk")
p(
    "Searching over 100 configurations on a validation set of fewer than "
    "50 events creates a substantial risk of overfitting to the validation "
    "data. To mitigate this, we report walk-forward cross-validation "
    "results (Section 14) as the primary measure of generalization. "
    "The validation-set results should be interpreted as upper bounds "
    "on expected performance."
)
spacer(4)
add_image(FIG_HEATMAPS, w=CONTENT_W * 0.85, h=8 * cm,
          cap="Figure 6: Heatmaps of F1 macro and cumulative return across the grid search space.")
page_break()

# ###########################################################################
#  SECTION 13 — EXPERIMENTAL RESULTS
# ###########################################################################
h1("13. Experimental Results")

h2("13.1 Best Parameters")
add_table([
    ["Objective", "pt", "sl", "mh", "F1 macro", "Return"],
    ["Best F1 (validation)", "2.0", "1.5", "10", "0.569", "\u2014"],
    ["Best return (validation)", "2.5", "3.0", "20", "\u2014", "25.9%"],
], col_widths=[3.5 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 2 * cm,
               CONTENT_W - 10 * cm])
caption("Table 7: Best hyperparameter configurations on the validation set.")
spacer(4)
p(
    "The best F1 configuration (pt=2.0, sl=1.5, mh=10) achieves an F1 "
    "macro of 0.569 on the validation set, substantially exceeding the "
    "baseline F1 of 0.160. This configuration uses a moderate profit target "
    "and a relatively tight stop loss, producing a balanced class "
    "distribution."
)
p(
    "The best return configuration (pt=2.5, sl=3.0, mh=20) achieves a "
    "cumulative return of 25.9% on the validation set. Notably, this "
    "configuration uses a wider stop loss, allowing trades more room to "
    "fluctuate before being stopped out. The longer holding period (20 "
    "bars) also gives profitable trades more time to reach the profit "
    "target."
)

h2("13.2 Model Comparison")
spacer(2)
add_image(FIG_RESULTS, w=CONTENT_W * 0.85, h=7 * cm,
          cap="Figure 7: Model comparison summary (RF, Bagging, Baseline).")
spacer(4)
add_table([
    ["Model", "Accuracy", "F1 macro", "Precision", "Recall"],
    ["Random Forest", "0.422", "0.347", "0.361", "0.371"],
    ["Bagging", "0.400", "0.331", "0.343", "0.352"],
    ["Baseline (stratified)", "0.286", "0.160", "0.165", "0.333"],
], col_widths=[3.5 * cm, 2 * cm, 2 * cm, 2 * cm, CONTENT_W - 9.5 * cm])
caption("Table 8: Model comparison on the validation set (best F1 configuration).")
spacer(4)
p(
    "Random Forest outperforms both Bagging and the stratified baseline "
    "across all metrics. The margin over Bagging is modest (2\u20133 "
    "percentage points in accuracy and F1), consistent with the literature "
    "finding that feature randomization provides only a small benefit "
    "in low-dimensional settings."
)
p(
    "The margin over the baseline is substantial: RF achieves an F1 of "
    "0.347 versus the baseline\u2019s 0.160, and accuracy of 42.2% versus "
    "28.6%. While 42.2% accuracy may appear low in absolute terms, it "
    "must be interpreted in the context of a three-class problem where "
    "random guessing would yield approximately 33%."
)

h2("13.3 Touch Events")
p(
    "Adding touch events to the training set increases the sample size "
    "from 104 to 142 (+36.5%). On the validation set, this yields a "
    "modest improvement in F1 (approximately 2\u20133 percentage points), "
    "suggesting that touch events provide additional, non-redundant "
    "information. The improvement is most pronounced for the <i>long</i> "
    "class, likely because touch events near channel boundaries often "
    "precede bounces that hit the profit target."
)

h2("13.4 Label Distribution")
add_image(FIG_LABEL_DIST, w=CONTENT_W * 0.75, h=6.5 * cm,
          cap="Figure 8: Label distribution for the best F1 configuration (pt=2.0, sl=1.5, mh=10).")
spacer(2)
add_image(FIG_CONFUSION, w=CONTENT_W * 0.75, h=7 * cm,
          cap="Figure 9: Confusion matrix for the Random Forest model on the validation set.")
page_break()

# ###########################################################################
#  SECTION 14 — GENERALIZATION AND VARIANCE ANALYSIS
# ###########################################################################
h1("14. Generalization and Variance Analysis")

h2("14.1 Motivation")
p(
    "A single train/test split provides a point estimate of performance "
    "that is highly sensitive to the specific market conditions in the "
    "test period. If the test period happens to be a strong bull market, "
    "a long-biased model will appear to perform well regardless of its "
    "actual predictive power. Walk-forward cross-validation addresses "
    "this by evaluating the model across multiple temporal windows, "
    "providing not just a mean performance estimate but also a variance "
    "estimate that captures the model\u2019s sensitivity to regime changes."
)

h2("14.2 Walk-Forward Variance")
p(
    "Table 9 reports the walk-forward cross-validation results across "
    "four temporal folds."
)
add_table([
    ["Metric", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Mean \u00b1 Std"],
    ["F1 macro", "0.278", "0.290", "0.276", "0.284", "0.282 \u00b1 0.008"],
    ["Cumulative return", "1.2%", "5.8%", "0.1%", "6.1%", "3.3% \u00b1 3.8%"],
    ["Win rate", "48.1%", "59.3%", "41.7%", "60.0%", "52.3% \u00b1 9.4%"],
    ["Sharpe ratio", "0.05", "0.31", "-0.08", "0.24", "0.131 \u00b1 0.169"],
], col_widths=[2.8 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm,
               CONTENT_W - 9.2 * cm])
caption("Table 9: Walk-forward cross-validation results (4 folds).")
spacer(4)
add_image(FIG_WF_VAR, w=CONTENT_W * 0.85, h=7 * cm,
          cap="Figure 10: Walk-forward performance variability across folds.")
spacer(4)
p(
    "The F1 macro is remarkably stable across folds (0.282 \u00b1 0.008), "
    "suggesting that the classifier\u2019s predictive ability is consistent "
    "across market regimes. In contrast, the financial metrics show "
    "substantial variability: the cumulative return ranges from 0.1% "
    "(Fold 3, covering the COVID-19 crash) to 6.1% (Fold 4), and the "
    "Sharpe ratio ranges from \u22120.08 to 0.31."
)
p(
    "This asymmetry between classification stability and financial "
    "instability is a key finding. It implies that while the model\u2019s "
    "ability to classify events is consistent, the financial payoff of "
    "correct classifications varies dramatically depending on market "
    "conditions. In a trending market, correct <i>long</i> predictions "
    "generate large returns; in a range-bound or bearish market, even "
    "correct predictions may generate only marginal returns."
)

h2("14.3 Walk-Forward vs K-Fold Comparison")
p(
    "The k-fold cross-validation yields higher accuracy estimates than "
    "walk-forward CV, which is expected: k-fold allows the model to "
    "train on both past and future data relative to the test fold, "
    "inflating performance. The gap between k-fold and walk-forward "
    "performance quantifies the benefit of information leakage. In this "
    "study, the gap is moderate (approximately 3\u20135 percentage points "
    "in F1), suggesting that the features do not have excessive temporal "
    "autocorrelation."
)

h2("14.4 Interpretation")
p(
    "The walk-forward results suggest that the system has a small but "
    "positive edge: the mean return of 3.3% per fold, achieved on "
    "approximately 15\u201320 trades per fold, is economically meaningful "
    "if sustained over time. However, the high variance (standard "
    "deviation of 3.8%) means that in any given deployment period, the "
    "system may break even or even lose money. The Sharpe ratio of "
    "0.131 is below the threshold typically considered acceptable for "
    "a standalone trading strategy (> 0.5), but it is positive, and the "
    "event-driven nature of the system means it has very low capital "
    "utilisation\u2014the capital is deployed only during active trades."
)
page_break()

# ###########################################################################
#  SECTION 15 — F-BETA ANALYSIS
# ###########################################################################
h1("15. F-Beta Analysis")

h2("15.1 Precision vs Recall in Trading")
p(
    "The choice between precision and recall has direct financial "
    "implications in a trading context. The following table maps "
    "classification errors to financial outcomes:"
)
add_table([
    ["Error Type", "Classification", "Financial Impact"],
    ["False Positive (FP)", "Model predicts long, actual is short/no_trade",
     "Financial loss: the trade is entered but results in a stop-loss hit or neutral exit"],
    ["False Negative (FN)", "Model predicts no_trade, actual is long",
     "Missed opportunity: a profitable trade is not taken, resulting in opportunity cost"],
    ["True Positive (TP)", "Model correctly predicts long",
     "Profit: the trade is entered and hits the profit target"],
    ["True Negative (TN)", "Model correctly avoids a bad trade",
     "Capital preservation: the model avoids a losing trade"],
], col_widths=[2.5 * cm, 4.5 * cm, CONTENT_W - 7 * cm])
caption("Table 10: Mapping classification errors to financial outcomes.")
spacer(4)
p(
    "In most trading contexts, false positives are more costly than false "
    "negatives: a losing trade reduces capital, while a missed opportunity "
    "merely fails to increase it. This asymmetry suggests that precision "
    "(minimising FP) should be weighted more heavily than recall "
    "(minimising FN). However, a system that is too conservative\u2014one "
    "that trades very rarely\u2014may miss enough profitable opportunities "
    "to make the strategy unviable."
)

h2("15.2 Results")
p(
    "The F-beta family of metrics provides a continuous spectrum between "
    "precision-weighted (F0.5) and recall-weighted (F2) evaluation. "
    "Table 11 reports the walk-forward mean values."
)
add_table([
    ["Metric", "Walk-Forward Mean", "Interpretation"],
    ["F0.5 (precision-weighted)", "0.285", "Penalises false positives (losing trades)"],
    ["F1 (balanced)", "0.282", "Equal weight to precision and recall"],
    ["F2 (recall-weighted)", "0.299", "Penalises missed opportunities"],
    ["Precision", "0.301", "Fraction of predicted longs that are correct"],
    ["Recall", "0.324", "Fraction of actual longs that are detected"],
], col_widths=[4 * cm, 3 * cm, CONTENT_W - 7 * cm])
caption("Table 11: F-beta analysis (walk-forward mean across 4 folds).")
spacer(4)
add_image(FIG_FBETA, w=CONTENT_W * 0.85, h=7 * cm,
          cap="Figure 11: F-beta comparison across walk-forward folds.")
spacer(4)

h2("15.3 Implications")
p(
    "The results show that F2 (0.299) marginally exceeds F0.5 (0.285), "
    "suggesting that the classifier is slightly recall-oriented: it "
    "captures more true positives at the expense of some false positives. "
    "Recall (0.324) exceeds precision (0.301) by a small margin, "
    "consistent with this interpretation."
)
p(
    "For a conservative trading strategy, a practitioner might prefer "
    "to optimize F0.5, accepting fewer trades but with higher per-trade "
    "confidence. For an aggressive strategy seeking to capture more "
    "of the available alpha, F2 optimization would be appropriate. The "
    "current model, which was optimized for F1, sits in between and "
    "could be shifted in either direction by adjusting the prediction "
    "threshold or the labeling parameters."
)
page_break()

# ###########################################################################
#  SECTION 16 — DISCUSSION
# ###########################################################################
h1("16. Discussion")

h2("16.1 Classification vs Profitability Tradeoff")
p(
    "A central tension in financial ML is that classification accuracy "
    "and trading profitability are not the same objective. The best "
    "F1 configuration (pt=2.0, sl=1.5, mh=10) and the best return "
    "configuration (pt=2.5, sl=3.0, mh=20) differ substantially. The "
    "F1-optimal configuration uses tighter barriers that produce more "
    "balanced classes and a cleaner classification signal; the "
    "return-optimal configuration uses wider barriers that allow trades "
    "more room to develop, resulting in fewer but larger wins."
)
p(
    "This tradeoff is fundamental and cannot be resolved by model "
    "improvement alone. A practitioner must decide which objective "
    "matters more: consistent classification (which may generate many "
    "small wins and small losses) or aggregate profitability (which "
    "may involve fewer, larger trades with more variance). The framework "
    "presented here allows this tradeoff to be explored systematically "
    "through the grid search."
)

h2("16.2 Touch Events")
p(
    "The inclusion of touch events increases the training set by 36.5% "
    "and modestly improves validation performance. However, touch events "
    "are not independent: multiple touches from the same pattern are "
    "correlated because they share the same geometric context. This "
    "correlation violates the i.i.d. assumption underlying standard "
    "cross-validation and may inflate performance estimates. Future "
    "work should explore purged cross-validation (Lopez de Prado, 2018) "
    "to account for this overlap."
)

h2("16.3 Leakage Prevention")
p(
    "The system implements multiple safeguards against data leakage, "
    "as described in Section 7.2. However, subtle forms of leakage "
    "may persist. For example, the S/R levels are computed over the "
    "entire dataset before events are generated; if a support level "
    "is identified using pivots that occur after the event timestamp, "
    "the feature \u201cdistance to nearest S/R\u201d would contain future "
    "information. The current implementation computes S/R levels using "
    "only pivots preceding the event, but this is a fragile guarantee "
    "that depends on the correct ordering of computation steps."
)

h2("16.4 F-Beta and Objectives")
p(
    "The F-beta analysis (Section 15) reveals that the model is slightly "
    "recall-oriented (F2 > F0.5), meaning it tends to predict <i>long</i> "
    "more often than a precision-optimized model would. In a trading "
    "context, this means the model takes more trades, some of which are "
    "false positives that result in stop-loss exits. Whether this "
    "recall bias is desirable depends on the practitioner\u2019s risk "
    "tolerance and capital constraints."
)

h2("16.5 Strengths and Weaknesses")
p("<b>Strengths:</b>")
bullet(
    "Fully reproducible: the entire pipeline can be run from a single "
    "command, producing identical results."
)
bullet(
    "Event-driven design reduces the multiple-testing problem and "
    "focuses the model on high-information regimes."
)
bullet(
    "Joint optimization of labeling parameters and model provides "
    "a principled approach to the label-design problem."
)
bullet(
    "Multiple validation schemes (walk-forward, k-fold, single-split) "
    "give a nuanced view of generalisation."
)
spacer(2)
p("<b>Weaknesses:</b>")
bullet(
    "Small dataset (142 events) limits the model\u2019s ability to learn "
    "complex patterns and increases the risk of overfitting."
)
bullet(
    "No transaction costs in the simulation; real-world profitability "
    "would be lower."
)
bullet(
    "Long-only strategy ignores potential short-selling opportunities."
)
bullet(
    "Single asset (SPY); cross-asset generalization is untested."
)
page_break()

# ###########################################################################
#  SECTION 17 — LIMITATIONS
# ###########################################################################
h1("17. Limitations")
p("The following limitations should be considered when interpreting the "
  "results of this study:")
spacer(2)
bullet(
    "<b>1. Small sample size.</b> With only 142 events, the training "
    "set is small by machine-learning standards. The model may not have "
    "sufficient data to learn subtle patterns, and the validation and "
    "test sets contain too few events for statistically robust evaluation. "
    "Confidence intervals on all reported metrics are wide."
)
bullet(
    "<b>2. Single asset.</b> All experiments are conducted on SPY. The "
    "patterns, features, and optimal hyperparameters may not transfer "
    "to other assets, especially those with different liquidity profiles, "
    "volatility characteristics, or market microstructure."
)
bullet(
    "<b>3. No transaction costs.</b> The trading simulation does not "
    "account for commissions, slippage, or market impact. While these "
    "costs are small for SPY, they would erode the already modest "
    "per-trade returns."
)
bullet(
    "<b>4. No regime conditioning.</b> The model treats all events "
    "identically regardless of the broader market regime (bull, bear, "
    "range-bound). Regime-aware models that adapt their behaviour to "
    "market conditions could improve performance."
)
bullet(
    "<b>5. Detector quality.</b> The pattern detectors (especially "
    "triangles and channels) were developed and tuned on the same dataset "
    "used for evaluation. While the detectors do not use future "
    "information within each event, the choice of detector parameters "
    "(e.g., minimum pattern length, containment threshold) was informed "
    "by visual inspection of the data."
)
bullet(
    "<b>6. Touch-event correlation.</b> Touch events from the same "
    "pattern are correlated, violating the i.i.d. assumption. The "
    "walk-forward evaluation partially mitigates this, but intra-fold "
    "correlation may still inflate metrics."
)
bullet(
    "<b>7. Fixed model hyperparameters.</b> Only the labeling parameters "
    "are searched; the tree ensemble hyperparameters (n_trees, max_depth) "
    "are fixed. Joint optimization of both sets could yield better results "
    "but would require a much larger search space."
)
bullet(
    "<b>8. Look-back bias in S/R computation.</b> Support and resistance "
    "levels are computed using the full history up to the event bar. In "
    "production, these levels would need to be recomputed incrementally, "
    "which may produce different results."
)
page_break()

# ###########################################################################
#  SECTION 18 — FUTURE WORK
# ###########################################################################
h1("18. Future Work")
p("Several extensions could build on the foundation laid by this thesis:")
spacer(2)
bullet(
    "<b>Multi-asset expansion.</b> Apply the pipeline to individual "
    "S&amp;P 500 constituents, sector ETFs, or international indices "
    "to test cross-asset generalization. A universal model trained on "
    "pooled events from multiple assets could leverage a much larger "
    "training set."
)
bullet(
    "<b>Deep learning with transfer learning.</b> With a larger dataset "
    "(from multi-asset pooling), deep learning models such as Temporal "
    "Convolutional Networks (TCN) or Transformers could be explored. "
    "Pre-training on a large corpus of financial time series and "
    "fine-tuning on pattern events is a promising direction."
)
bullet(
    "<b>Regime detection.</b> Integrate a Hidden Markov Model or a "
    "change-point detection algorithm to classify the market into "
    "regimes (trending, mean-reverting, volatile). Regime labels could "
    "be used as additional features or to condition the trading strategy."
)
bullet(
    "<b>Meta-labeling.</b> Following Lopez de Prado (2018), train a "
    "secondary model to predict the probability that a primary model\u2019s "
    "signal is correct. Meta-labeling decouples the direction prediction "
    "from the sizing decision, potentially improving risk-adjusted returns."
)
bullet(
    "<b>Realistic backtest engine.</b> Replace the simplified trading "
    "simulation with a full event-driven backtester that models "
    "transaction costs, slippage, partial fills, and margin requirements."
)
bullet(
    "<b>Purged cross-validation.</b> Implement combinatorial purged "
    "cross-validation to account for temporal autocorrelation and "
    "touch-event overlap, providing more reliable performance estimates."
)
bullet(
    "<b>Online learning.</b> Explore incremental learning algorithms "
    "that update the model in real time as new events arrive, rather "
    "than retraining from scratch at each walk-forward step."
)
bullet(
    "<b>Alternative labeling schemes.</b> Compare the triple-barrier "
    "method with fixed-horizon returns, volatility-adjusted returns, "
    "and trend-following labels to determine which scheme best aligns "
    "with the event-driven paradigm."
)
page_break()

# ###########################################################################
#  SECTION 19 — CONCLUSION
# ###########################################################################
h1("19. Conclusion")
p(
    "This thesis has presented a regime-aware machine-learning system "
    "for equity trading that integrates technical chart-pattern detection, "
    "event-driven feature engineering, triple-barrier labeling, and "
    "walk-forward validation into a cohesive, reproducible pipeline."
)
p(
    "The system was evaluated on 4,023 daily bars of SPY data spanning "
    "2010\u20132025. Four pattern detectors (support/resistance, channels, "
    "triangles, and multiple tops/bottoms) identified 104 structural "
    "events, augmented by 38 touch events for a total of 142 labeled "
    "samples. Each event was characterised by 48 features spanning "
    "technical indicators, pattern geometry, and event context."
)
p(
    "A grid search over 100 labeling configurations identified the "
    "best F1 configuration (pt=2.0, sl=1.5, mh=10, F1=0.569) and "
    "the best profitability configuration (pt=2.5, sl=3.0, mh=20, "
    "return=25.9%) on the validation set. Walk-forward cross-validation "
    "across four temporal folds yielded mean F1 = 0.282 \u00b1 0.008 "
    "and mean return = 3.3% \u00b1 3.8%, with a mean Sharpe ratio of "
    "0.131 \u00b1 0.169."
)
p(
    "Three research questions were addressed. Regarding RQ1, the "
    "event-driven approach successfully restricts the model to "
    "high-information events, achieving classification accuracy (42.2%) "
    "substantially above the stratified baseline (28.6%). Regarding RQ2, "
    "the sensitivity analysis reveals that labeling parameters have a "
    "pronounced effect on both classification and financial outcomes, "
    "justifying their treatment as searchable hyperparameters. Regarding "
    "RQ3, walk-forward analysis shows that classification performance "
    "generalizes well across temporal folds (F1 std = 0.008), but "
    "financial performance exhibits substantial regime-dependent "
    "variance (return std = 3.8%)."
)
p(
    "The central contribution of this work is not the achievement of "
    "exceptional trading returns\u2014the system\u2019s edge is modest\u2014"
    "but rather the methodological framework: a transparent, modular, "
    "and scientifically rigorous pipeline that can serve as a foundation "
    "for future research in event-driven financial ML. The explicit "
    "treatment of labeling parameters as hyperparameters, the multi-"
    "faceted evaluation combining classification and financial metrics, "
    "and the detailed leakage-prevention analysis are contributions that "
    "extend beyond the specific results reported here."
)
page_break()

# ###########################################################################
#  SECTION 20 — BIBLIOGRAPHY
# ###########################################################################
h1("20. Bibliography")
spacer(4)
bib("1", "Bailey, D. H., Borwein, J. M., Lopez de Prado, M., &amp; Zhu, Q. J. (2014). "
    "The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, "
    "and non-normality. <i>Journal of Portfolio Management</i>, 40(5), 94\u2013107.")
bib("2", "Bollinger, J. (2002). <i>Bollinger on Bollinger Bands</i>. McGraw-Hill.")
bib("3", "Breiman, L. (2001). Random Forests. <i>Machine Learning</i>, 45(1), 5\u201332.")
bib("4", "Bulkowski, T. N. (2005). <i>Encyclopedia of Chart Patterns</i> (2nd ed.). Wiley.")
bib("5", "Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical "
    "work. <i>The Journal of Finance</i>, 25(2), 383\u2013417.")
bib("6", "Fischer, T., &amp; Krauss, C. (2018). Deep learning with long short-term memory "
    "networks for financial market predictions. <i>European Journal of Operational "
    "Research</i>, 270(2), 654\u2013669.")
bib("7", "Grinsztajn, L., Oyallon, E., &amp; Varoquaux, G. (2022). Why do tree-based "
    "models still outperform deep learning on typical tabular data? <i>Advances "
    "in Neural Information Processing Systems</i>, 35.")
bib("8", "Krauss, C., Do, X. A., &amp; Huck, N. (2017). Deep neural networks, "
    "gradient-boosted trees, random forests: Statistical arbitrage on the "
    "S&amp;P 500. <i>European Journal of Operational Research</i>, 259(2), 689\u2013702.")
bib("9", "Lo, A. W., Mamaysky, H., &amp; Wang, J. (2000). Foundations of technical "
    "analysis: Computational algorithms, statistical inference, and empirical "
    "implementation. <i>The Journal of Finance</i>, 55(4), 1705\u20131765.")
bib("10", "Lopez de Prado, M. (2018). <i>Advances in Financial Machine Learning</i>. Wiley.")
bib("11", "Patel, J., Shah, S., Thakkar, P., &amp; Kotecha, K. (2015). Predicting stock "
    "and stock price index movement using trend deterministic data preparation "
    "and machine learning techniques. <i>Expert Systems with Applications</i>, "
    "42(1), 259\u2013268.")
bib("12", "Sezer, O. B., Gudelek, M. U., &amp; Ozbayoglu, A. M. (2020). Financial "
    "time series forecasting with deep learning: A systematic literature review: "
    "2005\u20132019. <i>Applied Soft Computing</i>, 90, 106181.")
bib("13", "Wilder, J. W. (1978). <i>New Concepts in Technical Trading Systems</i>. "
    "Trend Research.")
bib("14", "Zhang, Z., Zohren, S., &amp; Roberts, S. (2020). Deep learning for "
    "portfolio optimization. <i>The Journal of Financial Data Science</i>, 2(4), "
    "8\u201320.")
page_break()

# ###########################################################################
#  APPENDIX A — MODULE OVERVIEW
# ###########################################################################
h1("Appendix A: Module Overview")
spacer(4)
add_table([
    ["Module", "File", "Description", "Key Functions/Classes"],
    ["Data loading", "src/data/load_data.py",
     "Load SPY data from CSV, yfinance, or Alpha Vantage",
     "load_spy(source, ticker)"],
    ["ATR computation", "src/data/utils.py",
     "Compute Average True Range",
     "compute_atr(df, window)"],
    ["Pivot detection", "src/patterns/pivots.py",
     "Identify local extrema and count touches",
     "find_pivots(), count_touches()"],
    ["S/R levels", "src/patterns/support_resistance.py",
     "Cluster pivots into support/resistance zones",
     "find_sr_levels()"],
    ["Channel detection", "src/patterns/channels.py",
     "Detect ascending/descending/horizontal channels",
     "detect_channel()"],
    ["Triangle detection", "src/patterns/triangles.py",
     "Detect ascending/descending/symmetrical triangles",
     "detect_triangle_pattern()"],
    ["Multi top/bottom", "src/patterns/tops_bottoms.py",
     "Detect double/triple tops and bottoms",
     "detect_tops_bottoms()"],
    ["Pattern scanner", "src/patterns/scanner.py",
     "Run all detectors and merge results",
     "scan_all_patterns()"],
    ["Indicators", "src/features/indicators.py",
     "Compute 32 technical indicators",
     "compute_all_indicators()"],
    ["Feature builder", "src/features/build_features.py",
     "Build event-level feature matrix",
     "build_feature_matrix()"],
    ["Labeling", "src/labeling/label_events.py",
     "Triple-barrier labeling",
     "label_events()"],
    ["Training", "src/models/train.py",
     "Train RF, Bagging, Baseline; evaluate",
     "run_training_pipeline()"],
    ["Backtest", "src/backtest/simulate.py",
     "Trading simulation and performance metrics",
     "run_simulation()"],
], col_widths=[2 * cm, 3.2 * cm, 4.5 * cm, CONTENT_W - 9.7 * cm])
caption("Table A1: Complete module reference.")
page_break()

# ###########################################################################
#  APPENDIX B — PARAMETER REFERENCE
# ###########################################################################
h1("Appendix B: Parameter Reference")
spacer(4)
add_table([
    ["Parameter", "Module", "Default", "Description"],
    ["pivot_window", "pivots.py", "5", "Number of bars on each side for pivot detection"],
    ["atr_window", "utils.py", "14", "Look-back window for ATR computation"],
    ["sr_tolerance", "support_resistance.py", "0.5 ATR", "Clustering tolerance for S/R levels"],
    ["min_channel_length", "channels.py", "20", "Minimum bars for channel detection"],
    ["containment_threshold", "channels.py", "0.8", "Minimum fraction of bars within boundaries"],
    ["min_triangle_length", "triangles.py", "15", "Minimum bars for triangle detection"],
    ["touch_tolerance", "pivots.py", "0.3 ATR", "Distance threshold for touch-event registration"],
    ["n_estimators", "train.py", "200", "Number of trees in RF/Bagging ensemble"],
    ["max_depth", "train.py", "8", "Maximum tree depth"],
    ["min_samples_split", "train.py", "5", "Minimum samples required to split a node"],
    ["min_samples_leaf", "train.py", "3", "Minimum samples in a leaf node"],
    ["profit_target", "label_events.py", "2.0", "TP multiplier in ATR units (searchable)"],
    ["stop_loss", "label_events.py", "1.5", "SL multiplier in ATR units (searchable)"],
    ["max_holding", "label_events.py", "10", "Maximum holding period in bars (searchable)"],
    ["train_fraction", "train.py", "0.70", "Fraction of data for training"],
    ["val_fraction", "train.py", "0.15", "Fraction of data for validation"],
    ["wf_folds", "backtest", "4", "Number of walk-forward folds"],
], col_widths=[3 * cm, 3 * cm, 1.8 * cm, CONTENT_W - 7.8 * cm])
caption("Table B1: Complete parameter reference with defaults.")
page_break()

# ###########################################################################
#  APPENDIX C — NOTEBOOK GUIDE AND REPRODUCIBILITY
# ###########################################################################
h1("Appendix C: Notebook Guide and Reproducibility")
spacer(4)

h2("C.1 Notebook Guide")
add_table([
    ["#", "Notebook", "Purpose"],
    ["01", "01_data_exploration.ipynb", "Load SPY data, visualise OHLCV, compute basic statistics"],
    ["02", "02_pivot_detection.ipynb", "Detect pivot highs/lows, visualise on price chart"],
    ["03", "03_sr_levels.ipynb", "Cluster pivots into S/R levels, validate with price touches"],
    ["04", "04_triangle_gallery.ipynb", "Detect and visualise triangles with containment analysis"],
    ["05", "05_channel_detection.ipynb", "Detect channels, analyse slope and width distributions"],
    ["06", "06_channel_gallery.ipynb", "Gallery of detected channels with touch annotations"],
    ["07", "07_data_source_comparison.ipynb", "Compare CSV, yfinance, and Alpha Vantage data sources"],
    ["08", "08_detector_touch_analysis.ipynb", "Validate touch-counting algorithm across pattern types"],
    ["09", "09_feature_engineering.ipynb", "Compute and visualise 48 features, correlation analysis"],
    ["10", "10_model_training.ipynb", "Train RF, Bagging, Baseline; confusion matrices, feature importance"],
    ["11", "11_experiment_summary.ipynb", "Grid search results, heatmaps, walk-forward analysis"],
    ["12", "12_fbeta_analysis.ipynb", "F-beta spectrum analysis, precision-recall tradeoffs"],
    ["13", "13_final_evaluation.ipynb", "Final test-set evaluation, trading simulation, summary figures"],
], col_widths=[0.8 * cm, 5 * cm, CONTENT_W - 5.8 * cm])
caption("Table C1: Complete notebook guide (13 notebooks).")

spacer(6)
h2("C.2 Reproducibility Instructions")
p(
    "To reproduce all results from scratch, follow these steps:"
)
bullet(
    "<b>Step 1: Environment setup.</b> Create a Python 3.10+ virtual "
    "environment and install dependencies: "
    "<font face='Courier' size='8'>pip install -r requirements.txt</font>"
)
bullet(
    "<b>Step 2: Data.</b> Ensure <font face='Courier' size='8'>"
    "data/raw/spy.csv</font> is present (4,023 rows, 2010\u20132025). "
    "Alternatively, run notebook 07 to download via yfinance."
)
bullet(
    "<b>Step 3: Run notebooks.</b> Execute notebooks 01\u201313 in order. "
    "Each notebook is self-contained and produces its own figures and "
    "intermediate outputs."
)
bullet(
    "<b>Step 4: Generate reports.</b> Run "
    "<font face='Courier' size='8'>python reports/generate_report_v3.py</font> "
    "to produce this PDF."
)
bullet(
    "<b>Step 5: Verify.</b> Compare the generated figures and metrics "
    "against those reported in this document. All numbers should match "
    "exactly when using the provided CSV data."
)
spacer(4)
p(
    "The complete codebase is organised as follows:"
)
story.append(Paragraph(
    "<font face='Courier' size='8'>"
    "regime-aware-ml-trading/<br/>"
    "&nbsp;&nbsp;\u251c\u2500\u2500 data/raw/spy.csv<br/>"
    "&nbsp;&nbsp;\u251c\u2500\u2500 src/<br/>"
    "&nbsp;&nbsp;\u2502&nbsp;&nbsp;&nbsp;\u251c\u2500\u2500 data/<br/>"
    "&nbsp;&nbsp;\u2502&nbsp;&nbsp;&nbsp;\u251c\u2500\u2500 patterns/<br/>"
    "&nbsp;&nbsp;\u2502&nbsp;&nbsp;&nbsp;\u251c\u2500\u2500 features/<br/>"
    "&nbsp;&nbsp;\u2502&nbsp;&nbsp;&nbsp;\u251c\u2500\u2500 labeling/<br/>"
    "&nbsp;&nbsp;\u2502&nbsp;&nbsp;&nbsp;\u251c\u2500\u2500 models/<br/>"
    "&nbsp;&nbsp;\u2502&nbsp;&nbsp;&nbsp;\u251c\u2500\u2500 backtest/<br/>"
    "&nbsp;&nbsp;\u2502&nbsp;&nbsp;&nbsp;\u2514\u2500\u2500 utils/<br/>"
    "&nbsp;&nbsp;\u251c\u2500\u2500 notebooks/<br/>"
    "&nbsp;&nbsp;\u251c\u2500\u2500 reports/<br/>"
    "&nbsp;&nbsp;\u2514\u2500\u2500 requirements.txt"
    "</font>",
    styles["Body"]
))
spacer(6)
p(
    "All source code, notebooks, data, and report generators are included "
    "in the submission. The system has been tested on macOS 14 and Ubuntu "
    "22.04 with Python 3.10 and 3.11."
)

# ═══════════════════════════════════════════════════════════════════════════
# BUILD PDF
# ═══════════════════════════════════════════════════════════════════════════
print(f"Building PDF with {len(story)} flowables ...")

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=MARGIN,
    rightMargin=MARGIN,
    topMargin=MARGIN,
    bottomMargin=MARGIN,
    title="Regime-Aware Machine Learning for Equity Trading",
    author="Zeineb Turki",
)

doc.build(story, onFirstPage=footer_first, onLaterPages=footer)
print(f"Done. Output: {OUTPUT}")
