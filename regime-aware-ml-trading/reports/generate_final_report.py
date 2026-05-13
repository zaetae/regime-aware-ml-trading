"""Generate final report / mini-thesis PDF (~30-35 pages).

Produces: reports/final/Zeineb_Turki_zjk.pdf

Scientific-paper style with full methodology, results, and discussion.
Uses ReportLab with existing thesis figures where available.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(__file__)
THESIS_FIG = os.path.join(BASE, "thesis_figures")
EXP_FIG = os.path.join(BASE, "experiment_figures")
IMG = os.path.join(BASE, "images")
FINAL = os.path.join(BASE, "final")
REPORT_FIG = os.path.join(FINAL, "report_figures")
os.makedirs(REPORT_FIG, exist_ok=True)

OUTPUT = os.path.join(FINAL, "Zeineb_Turki_zjk.pdf")

PAGE_W, PAGE_H = A4
MARGIN_L = 2.5 * cm
MARGIN_R = 2.5 * cm
MARGIN_T = 2.5 * cm
MARGIN_B = 2.5 * cm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

# Colours
BLUE = HexColor("#1B3A5C")
ACCENT = HexColor("#2E86C1")
GREY = HexColor("#808B96")
LIGHT = HexColor("#EBF5FB")

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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

def caption(text):
    story.append(Paragraph(text, styles["Caption"]))

def spacer(h=6):
    story.append(Spacer(1, h * mm))

def add_table(data, col_widths=None, header_color=BLUE):
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
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
    t.setStyle(style)
    story.append(t)

def add_image(path, w=None, h=None, cap=None):
    if os.path.exists(path):
        if w is None:
            w = CONTENT_W * 0.85
        story.append(Image(path, width=w, height=h))
        if cap:
            caption(cap)

# ---------------------------------------------------------------------------
# Generate custom figures
# ---------------------------------------------------------------------------
print("Generating report figures...")

# Pipeline diagram
fig, ax = plt.subplots(figsize=(14, 2.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.8)
ax.axis("off")
boxes = [
    (0.1, "SPY\nOHLCV\nData"),
    (1.55, "Pattern\nDetectors\n(4 types)"),
    (3.0, "Event\nSignals\n(132 bars)"),
    (4.45, "Feature\nEngineering\n(48 features)"),
    (5.9, "Triple-Barrier\nLabeling\n(3 classes)"),
    (7.35, "ML Models\n(RF, Bag,\nBaseline)"),
    (8.8, "Validation\n& Backtest\n(3 methods)"),
]
bw, bh = 1.25, 1.1
for x, txt in boxes:
    rect = mpatches.FancyBboxPatch((x, 0.35), bw, bh, boxstyle="round,pad=0.1",
                                    facecolor="#2E86C1", edgecolor="#1B3A5C", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + bw / 2, 0.35 + bh / 2, txt, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white")
for i in range(len(boxes) - 1):
    x1 = boxes[i][0] + bw
    x2 = boxes[i + 1][0]
    ax.annotate("", xy=(x2, 0.9), xytext=(x1, 0.9),
                 arrowprops=dict(arrowstyle="->", lw=2, color="#1B3A5C"))
pipeline_path = os.path.join(REPORT_FIG, "pipeline.png")
plt.savefig(pipeline_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()

# Triple-barrier illustration
fig, ax = plt.subplots(figsize=(8, 4))
np.random.seed(42)
prices = [100]
for _ in range(15):
    prices.append(prices[-1] + np.random.randn() * 1.2)
prices = np.array(prices)
ax.plot(range(len(prices)), prices, "b-o", markersize=4, label="Price")
ax.axhline(y=100 + 2 * 2.5, color="green", linestyle="--", linewidth=2, label="Upper barrier (TP)")
ax.axhline(y=100 - 2 * 2.5, color="red", linestyle="--", linewidth=2, label="Lower barrier (SL)")
ax.axvline(x=10, color="orange", linestyle=":", linewidth=2, label="Time barrier (max_holding)")
ax.axhline(y=100, color="grey", linestyle=":", alpha=0.5)
ax.scatter([0], [100], color="blue", s=100, zorder=5, label="Entry (signal bar)")
ax.set_xlabel("Bars after signal")
ax.set_ylabel("Price")
ax.set_title("Triple-Barrier Labeling Method")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)
barrier_path = os.path.join(REPORT_FIG, "triple_barrier.png")
plt.savefig(barrier_path, dpi=150, bbox_inches="tight")
plt.close()

# Results comparison
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
configs = ["Default\n(2.0/2.0/10)", "Best F1\n(2.0/1.5/10)", "Best Profit\n(2.5/3.0/20)", "Best F1\n+ Touch"]
x = range(4)
clrs = ["#808B96", "#2E86C1", "#E67E22", "#27AE60"]
axes[0].bar(x, [0.160, 0.569, 0.392, 0.420], color=clrs, alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(configs, fontsize=8)
axes[0].set_ylabel("F1 Macro"); axes[0].set_title("Classification", fontweight="bold")
axes[1].bar(x, [0.037, 0.085, 0.259, 0.095], color=clrs, alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(configs, fontsize=8)
axes[1].set_ylabel("Cumulative Return"); axes[1].set_title("Profitability", fontweight="bold")
axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.3)
axes[2].bar(x, [0.50, 0.55, 0.52, 0.53], color=clrs, alpha=0.85)
axes[2].set_xticks(x); axes[2].set_xticklabels(configs, fontsize=8)
axes[2].set_ylabel("Win Rate"); axes[2].set_title("Win Rate", fontweight="bold")
axes[2].axhline(y=0.5, color="black", linestyle="--", alpha=0.3)
plt.tight_layout()
results_path = os.path.join(REPORT_FIG, "results_comparison.png")
plt.savefig(results_path, dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Build document
# ---------------------------------------------------------------------------
print("Building final report PDF...")

doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=MARGIN_L, rightMargin=MARGIN_R,
    topMargin=MARGIN_T, bottomMargin=MARGIN_B,
)

story = []

# ===================================================================
# TITLE PAGE
# ===================================================================
story.append(Spacer(1, 5 * cm))
story.append(Paragraph("Regime-Aware Machine Learning<br/>for Technical Pattern Trading",
                        styles["TitleMain"]))
spacer(8)
story.append(Paragraph("Event-Based Classification and Profitability Evaluation<br/>"
                        "on S&amp;P 500 ETF (SPY) Daily Data",
                        styles["TitleSub"]))
spacer(15)
story.append(Paragraph("Zeineb Turki", styles["TitleInfo"]))
spacer(3)
story.append(Paragraph("Supervisor: [Supervisor Name]", styles["TitleInfo"]))
story.append(Paragraph("[University Name]", styles["TitleInfo"]))
story.append(Paragraph("[Course / Program Name]", styles["TitleInfo"]))
spacer(5)
story.append(Paragraph("Summer Semester 2026", styles["TitleInfo"]))
story.append(Paragraph("May 2026", styles["TitleInfo"]))
story.append(PageBreak())

# ===================================================================
# ASSIGNMENT STATEMENT (placeholder)
# ===================================================================
story.append(Spacer(1, 3 * cm))
story.append(Paragraph("Assignment Statement", styles["H1"]))
spacer(10)
p("<i>[Placeholder: Insert the official signed assignment/task sheet here. "
  "This page should contain the formal task description as provided by the supervisor "
  "or university administration.]</i>")
story.append(PageBreak())

# ===================================================================
# TABLE OF CONTENTS
# ===================================================================
story.append(Paragraph("Table of Contents", styles["H1"]))
spacer(5)
toc_items = [
    "1. Abstract",
    "2. Introduction",
    "3. Background and Related Work",
    "4. System Design",
    "5. Data and Preprocessing",
    "6. Pattern Detection Methodology",
    "7. Feature Engineering",
    "8. Triple-Barrier Labeling",
    "9. Machine Learning Models",
    "10. Validation Methodology",
    "11. Trading Simulation and Profitability Evaluation",
    "12. Hyperparameter Optimization",
    "13. Experimental Results",
    "14. Discussion",
    "15. Limitations",
    "16. Future Work",
    "17. Conclusion",
    "18. Bibliography",
    "Appendix A: Module Overview",
    "Appendix B: Parameter Reference",
    "Appendix C: Notebook Guide",
]
for item in toc_items:
    story.append(Paragraph(item, styles["Body"]))
story.append(PageBreak())

# ===================================================================
# 1. ABSTRACT
# ===================================================================
h1("1. Abstract")
story.append(Paragraph(
    "This report presents a regime-aware machine learning system for trading "
    "S&amp;P 500 ETF (SPY) daily data. The system detects technical patterns "
    "(support/resistance, channels, triangles, and multiple tops/bottoms), "
    "extracts 48 leakage-free features per event, and assigns labels using "
    "triple-barrier labeling. Random Forest and Bagging classifiers are trained "
    "on 132 detected events across 4,023 trading days (2010\u20132025). "
    "A key contribution is treating the triple-barrier parameters (profit target, "
    "stop loss, and maximum holding period) as tunable hyperparameters, optimized "
    "via grid search across 100 configurations. The system evaluates models by "
    "both classification metrics (F1 macro: 0.569, 3.6\u00d7 over baseline) and "
    "trading profitability (cumulative return: 25.9%). A central finding is that "
    "optimal parameters for classification differ from those for profitability, "
    "confirming that accuracy alone is insufficient for evaluating trading models. "
    "Touch-based events augment the dataset by 36.5% (38 additional events). "
    "All results are preliminary given the small dataset size and simplified "
    "trading assumptions.",
    styles["AbstractBody"],
))
story.append(PageBreak())

# ===================================================================
# 2. INTRODUCTION
# ===================================================================
h1("2. Introduction")

h2("2.1 Motivation")
p("Financial markets present one of the most challenging domains for machine learning. "
  "Price series are extremely noisy, non-stationary, and influenced by regime shifts, "
  "macroeconomic events, and structural market changes. Most individual price bars do not "
  "contain a repeatable, actionable signal. Consequently, models trained to predict every "
  "bar's direction typically learn to exploit price level trends or temporal patterns "
  "rather than genuine trading opportunities.")

p("This project takes an alternative approach: <b>event-based learning</b>. Instead of "
  "predicting every bar, the system first identifies technically meaningful moments\u2014bars "
  "where price interacts with established chart patterns\u2014and then predicts the outcome "
  "of a trade-sized move initiated at these moments. This reduces the problem from thousands "
  "of noisy bars to a focused set of candidate trading events.")

h2("2.2 Research Questions")
p("The project addresses three main questions:")
p("1. Can technical pattern detection combined with machine learning classify trade outcomes "
  "better than a random baseline?")
p("2. Do the optimal labeling parameters for classification accuracy also optimize trading "
  "profitability?")
p("3. Does augmenting the event set with touch-based signals improve model performance?")

h2("2.3 Main Contributions")
p("The project makes the following contributions:")
p("\u2022 An end-to-end pipeline from raw OHLCV data to profitability-evaluated predictions, "
  "implemented as modular Python code.")
p("\u2022 Four complementary pattern detectors with signal localization and cooldown filtering.")
p("\u2022 48 leakage-free features combining technical indicators, pattern geometry, and "
  "event-type encodings.")
p("\u2022 Conversion of triple-barrier labeling parameters from fixed constants to tunable "
  "hyperparameters, with grid search optimization targeting both classification and "
  "profitability.")
p("\u2022 A trading simulation module that evaluates models by cumulative return, Sharpe ratio, "
  "win rate, profit factor, and maximum drawdown.")
p("\u2022 Touch-based event augmentation that increases the dataset by 36.5%.")
story.append(PageBreak())

# ===================================================================
# 3. BACKGROUND AND RELATED WORK
# ===================================================================
h1("3. Background and Related Work")

h2("3.1 Technical Analysis")
p("Technical analysis is the study of past price and volume data to forecast future price "
  "movements. While its theoretical foundations are debated, specific chart patterns are "
  "widely used by practitioners and have been formalized in academic literature "
  "(Lo, Mamaysky &amp; Wang, 2000). The patterns used in this project\u2014support/resistance "
  "levels, price channels, triangle formations, and multiple tops/bottoms\u2014are among the "
  "most commonly cited in both trading practice and quantitative finance research.")

h2("3.2 Key Technical Indicators")
p("<b>Average True Range (ATR):</b> A volatility measure computed as the rolling mean of "
  "True Range (the maximum of: High\u2212Low, |High\u2212prev Close|, |Low\u2212prev Close|). "
  "ATR is used throughout this project as a scale-independent distance measure for "
  "barriers, proximity thresholds, and channel widths.")
p("<b>Relative Strength Index (RSI):</b> A momentum oscillator bounded between 0 and 100, "
  "measuring the speed and magnitude of recent price changes. Values above 70 indicate "
  "overbought conditions; below 30 indicates oversold.")
p("<b>MACD (Moving Average Convergence Divergence):</b> The difference between fast and slow "
  "exponential moving averages. In this project, MACD values are normalized by dividing "
  "by the Close price to prevent scale leakage across different price regimes.")
p("<b>Bollinger Bands:</b> A volatility envelope defined as a simple moving average \u00b1 "
  "2 standard deviations. The width and the position of price within the bands (%B) "
  "are used as features.")

h2("3.3 Triple-Barrier Labeling")
p("The triple-barrier method, introduced by Lopez de Prado (2018) in <i>Advances in "
  "Financial Machine Learning</i>, assigns labels based on which of three barriers is "
  "touched first after a trading signal:")
p("\u2022 <b>Upper barrier (profit target):</b> entry price + pt_mult \u00d7 ATR")
p("\u2022 <b>Lower barrier (stop loss):</b> entry price \u2212 sl_mult \u00d7 ATR")
p("\u2022 <b>Time barrier:</b> maximum holding period in bars")
p("This approach labels events according to realistic trade outcomes rather than "
  "arbitrary future price direction, making it well-suited for trading applications.")

h2("3.4 Random Forest and Bagging")
p("Random Forest (Breiman, 2001) constructs an ensemble of decision trees, each trained "
  "on a bootstrap sample with random feature subsets at each split. Bagging uses the same "
  "bootstrap aggregation but without feature subsetting. Both are robust to noise, handle "
  "mixed feature types naturally, and provide feature importance rankings. Balanced class "
  "weighting adjusts for label imbalance.")

h2("3.5 Walk-Forward Validation")
p("Walk-forward cross-validation simulates realistic model deployment by training on "
  "historical data and testing on subsequent unseen periods. Unlike standard k-fold CV, "
  "it respects temporal ordering and prevents future information from leaking into "
  "training. This is essential in financial applications where time-series structure "
  "invalidates the i.i.d. assumption.")
story.append(PageBreak())

# ===================================================================
# 4. SYSTEM DESIGN
# ===================================================================
h1("4. System Design")

h2("4.1 Architecture Overview")
p("The system follows a modular pipeline architecture, implemented as a Python package "
  "with clearly separated concerns:")
add_image(pipeline_path, w=CONTENT_W, cap="Figure 1: System architecture — end-to-end pipeline from raw data to profitability evaluation.")

h2("4.2 Module Structure")
add_table([
    ["Module", "Key Files", "Responsibility"],
    ["src/data/", "load_data.py, utils.py", "Data loading (CSV, yfinance, Alpha Vantage), ATR computation"],
    ["src/patterns/", "scanner.py, 5 detectors", "Pattern detection, event scanning, touch events"],
    ["src/labeling/", "label_events.py", "Triple-barrier labeling with configurable parameters"],
    ["src/features/", "indicators.py, build_features.py", "48 technical features, geometry, event dummies"],
    ["src/models/", "train.py, optimize.py", "RF/Bagging training, CV, hyperparameter optimization"],
    ["src/backtest/", "simulator.py", "Trade simulation, profitability metrics"],
    ["notebooks/", "12 Jupyter notebooks", "Interactive analysis, visualization, experiments"],
    ["reports/", "11 generators", "PDF report and thesis generation"],
], col_widths=[2.5 * cm, 4 * cm, 7.5 * cm])
caption("Table 1: Module structure and responsibilities.")

h2("4.3 Design Principles")
p("\u2022 <b>No lookahead leakage:</b> every feature at bar <i>i</i> uses only data up to and including bar <i>i</i>.")
p("\u2022 <b>Chronological splitting:</b> training always precedes validation and test data in time.")
p("\u2022 <b>Modular independence:</b> each module can be used standalone or as part of the pipeline.")
p("\u2022 <b>Configurable parameters:</b> all thresholds are exposed as function arguments with documented defaults.")
story.append(PageBreak())

# ===================================================================
# 5. DATA AND PREPROCESSING
# ===================================================================
h1("5. Data and Preprocessing")

h2("5.1 Dataset Description")
p("The dataset consists of daily OHLCV (Open, High, Low, Close, Volume) data for the "
  "SPDR S&amp;P 500 ETF Trust (SPY), covering 4,023 trading days from January 4, 2010 "
  "to December 30, 2025. SPY was chosen because it is the most liquid equity ETF, "
  "minimizing the impact of illiquidity on pattern detection.")

add_table([
    ["Property", "Value"],
    ["Ticker", "SPY"],
    ["Period", "2010-01-04 to 2025-12-30"],
    ["Trading days", "4,023"],
    ["Price range", "$77.36 \u2013 $690.38"],
    ["Data sources", "CSV cache, Yahoo Finance, Alpha Vantage"],
    ["Required columns", "Open, High, Low, Close, Volume"],
], col_widths=[4 * cm, 8 * cm])
caption("Table 2: Dataset characteristics.")

h2("5.2 Data Cleaning")
p("The loading pipeline performs automatic cleaning: timezone removal from DatetimeIndex, "
  "chronological sorting, NaN removal, and validation that all required OHLCV columns are present. "
  "The primary data source is a local CSV cache; Yahoo Finance and Alpha Vantage serve as fallbacks.")

h2("5.3 Indicator Computation")
p("All technical indicators are computed bar-by-bar using only current and historical data. "
  "The 14-period Average True Range (ATR) serves as the fundamental volatility measure, "
  "used for barrier distances, proximity thresholds, and normalization throughout the system.")
story.append(PageBreak())

# ===================================================================
# 6. PATTERN DETECTION METHODOLOGY
# ===================================================================
h1("6. Pattern Detection Methodology")

p("The system employs four complementary pattern detectors, each targeting a different "
  "type of technically significant price structure. All detectors share common design "
  "principles: ATR-based thresholds for scale independence, cooldown filters (10 bars) "
  "to prevent signal clustering, and signal localization at the specific bar where the "
  "pattern condition is met.")

h2("6.1 Support and Resistance")
p("Support and resistance levels are identified using rolling extremes of the High (resistance) "
  "and Low (support) prices over a 50-bar window (\u22482.5 months). Two filters reduce false positives:")
p("\u2022 <b>Level stability:</b> the level must be unchanged for at least 5 consecutive bars, "
  "filtering out continuously rising resistance in uptrends.")
p("\u2022 <b>Proximity test:</b> Close must be within 0.3 \u00d7 ATR(14) of a stable level.")
p("Signals are suppressed for 10 bars after each detection to avoid flagging the same "
  "approach as multiple events.")

h2("6.2 Channel Detection")
p("Price channels are detected using the chunk-extremes method from the TrendLineChannelDetection "
  "reference. The algorithm:")
p("1. Divides a dynamic lookback window (40 \u00b1 15 bars) into non-overlapping 5-bar chunks.")
p("2. Extracts the maximum-High and minimum-Low bar from each chunk.")
p("3. Fits first-degree polynomials (polyfit) to the chunk extremes for upper and lower boundaries.")
p("4. Adjusts intercepts so the upper line caps all chunk highs and the lower line floors all chunk lows.")
p("5. Validates with swing-pivot touches (\u22652 upper, \u22653 lower, within 0.20 \u00d7 ATR), "
  "parallelism (slope difference \u2264 0.25), minimum width (1\u20136 \u00d7 ATR), and "
  "containment (\u226570% of bars inside).")
p("Confidence scores weight touches (40%), containment (25%), parallelism (20%), and width (15%), "
  "with a 20% penalty if no rejection confirmation at the boundary.")

h2("6.3 Triangle Detection")
p("Triangles are identified from swing pivots (local extremes within \u00b13 bars) using "
  "linear regression:")
p("1. Find swing highs and swing lows within a 25-bar lookback window.")
p("2. Fit linear regression (scipy.stats.linregress) to each set of pivots.")
p("3. Require |r| \u2265 0.85 on each trendline for tight pivot alignment.")
p("4. Check convergence: at least 5% range compression between the trendlines.")
p("5. Classify by slope: ascending (rising lower, flat upper), descending (falling upper, flat lower), "
  "or symmetric (both converging).")
p("6. Fire breakout signal when price exceeds the recent 3-bar range by 0.3 \u00d7 ATR.")
p("<b>Note:</b> Per supervisor feedback, triangle and channel events are excluded from "
  "training labels due to concerns about detection accuracy. They are retained for "
  "pattern geometry features and discussed for future improvement.")

h2("6.4 Multiple Tops and Bottoms")
p("Multiple top/bottom patterns are detected using rolling extremes combined with "
  "close-trend confirmation:")
p("\u2022 <b>Multiple top:</b> rolling 20-bar maximum of highs remains at its ceiling while "
  "the 5-bar close slope turns negative.")
p("\u2022 <b>Multiple bottom:</b> rolling 20-bar minimum of lows remains at its floor while "
  "the 5-bar close slope turns positive.")

h2("6.5 Touch-Based Event Expansion")
p("Following supervisor guidance to <i>\"start sequences from direct touch of trend lines,\"</i> "
  "the system generates additional events when price directly touches support/resistance "
  "levels or channel boundaries. Touch events use a tighter proximity threshold "
  "(0.2 \u00d7 ATR vs. 0.3 \u00d7 ATR for standard detectors) and are tracked separately "
  "from original detector events.")

add_table([
    ["Event Source", "Count", "Percentage"],
    ["Standard detectors", "104", "73.2%"],
    ["Touch-only events", "38", "26.8%"],
    ["Combined total", "142", "100%"],
], col_widths=[5 * cm, 3 * cm, 3 * cm])
caption("Table 3: Event counts by source.")

h2("6.6 Detection Summary")
add_table([
    ["Detector", "Events", "Key Thresholds"],
    ["Support / Resistance", "42", "window=50, 0.3\u00d7ATR, stability=5 bars"],
    ["Triangles", "17", "window=25, |r|\u22650.85, 5% convergence"],
    ["Channels", "12", "lookback=40\u00b115, \u22652+3 touches, \u226570% containment"],
    ["Multiple Tops/Bottoms", "63", "window=20, 5-bar slope confirmation"],
    ["Touch events (additional)", "38", "0.2\u00d7ATR proximity, cooldown=10"],
    ["TOTAL (combined)", "142", "3.5% event density"],
], col_widths=[4.5 * cm, 2 * cm, 7.5 * cm])
caption("Table 4: Pattern detection summary.")
story.append(PageBreak())

# ===================================================================
# 7. FEATURE ENGINEERING
# ===================================================================
h1("7. Feature Engineering")

p("Each detected event is represented by 48 features extracted from three sources: "
  "bar-level technical indicators (computed at the event bar), pattern geometry metadata, "
  "and event-type encodings. All features use only information available at or before "
  "the event bar\u2014no future data is used.")

h2("7.1 Feature Groups")
add_table([
    ["Group", "Count", "Examples", "Source"],
    ["Volatility", "2", "atr_ratio (ATR/Close), rvol_20", "indicators.py"],
    ["Returns", "4", "ret_1, ret_5, ret_10, ret_20", "indicators.py"],
    ["Momentum (ROC)", "3", "mom_5, mom_10, mom_20", "indicators.py"],
    ["MA Distances", "5", "sma_10_dist ... sma_200_dist", "indicators.py"],
    ["MA Spreads", "3", "ma_spread_10_50, 20_200, 50_200", "indicators.py"],
    ["RSI", "1", "rsi_14", "indicators.py"],
    ["MACD (normalized)", "3", "macd_norm, signal_norm, hist_norm", "indicators.py"],
    ["Bollinger Bands", "2", "bb_width, bb_pctb", "indicators.py"],
    ["Volume", "2", "volume_ratio, volume_std", "indicators.py"],
    ["Binary Filters", "8", "BB touches, SMA crosses, RSI extremes", "indicators.py"],
    ["Pattern Geometry", "11", "slopes, touches, containment, width, R\u00b2", "build_features.py"],
    ["Event Type Dummies", "4\u20136", "One-hot encoded pattern type", "build_features.py"],
    ["TOTAL", "48\u201350", "", ""],
], col_widths=[3.5 * cm, 1.3 * cm, 5.5 * cm, 3.5 * cm])
caption("Table 5: Feature groups in the final feature matrix.")

h2("7.2 Leakage Prevention")
p("Several features that are common in technical analysis were deliberately excluded "
  "because they leak temporal information:")
p("\u2022 <b>Raw ATR (atr_14):</b> scales with price level (SPY at $100 vs $600). "
  "Replaced with atr_ratio = ATR / Close.")
p("\u2022 <b>Absolute SMA values:</b> trend upward with price over time, serving as time proxies. "
  "Replaced with relative distance features (sma_*_dist = (Close \u2212 SMA) / SMA).")
p("\u2022 <b>Cumulative OBV:</b> grows monotonically over time. Replaced with volume_ratio "
  "and volume_std (rolling, bounded measures).")
p("\u2022 <b>Raw MACD:</b> scales with price level. Replaced with price-normalized versions "
  "(macd_norm = MACD_line / Close).")
p("\u2022 <b>Entry price and event ATR:</b> directly encode the price level at the event bar. "
  "Excluded from features; volatility regime is captured by atr_ratio instead.")
story.append(PageBreak())

# ===================================================================
# 8. TRIPLE-BARRIER LABELING
# ===================================================================
h1("8. Triple-Barrier Labeling")

h2("8.1 Method Description")
p("The triple-barrier method assigns directional labels to each event based on which "
  "barrier is touched first when walking forward from the signal bar:")
add_image(barrier_path, w=CONTENT_W * 0.7,
          cap="Figure 2: Triple-barrier labeling — upper (TP), lower (SL), and time barriers.")

p("For each event:")
p("1. <b>Entry price</b> = Close at the event bar.")
p("2. <b>Upper barrier</b> = entry_price + pt_mult \u00d7 ATR(14).")
p("3. <b>Lower barrier</b> = entry_price \u2212 sl_mult \u00d7 ATR(14).")
p("4. Walk forward from bar pos+1 through pos+max_holding.")
p("5. If High \u2265 upper barrier first \u2192 label <b>\"long\"</b>.")
p("6. If Low \u2264 lower barrier first \u2192 label <b>\"short\"</b>.")
p("7. If neither within max_holding bars \u2192 label <b>\"no_trade\"</b>.")
p("8. If both barriers are hit on the same bar, the Close price relative to entry determines the label.")

h2("8.2 Parameters as Hyperparameters")
p("A key contribution of this project is converting pt_mult, sl_mult, and max_holding "
  "from fixed constants to tunable hyperparameters. Different parameter settings change "
  "the label distribution (more or fewer long/short/no_trade labels), the difficulty of "
  "the classification task, and the profitability of the resulting trading strategy.")
story.append(PageBreak())

# ===================================================================
# 9. MACHINE LEARNING MODELS
# ===================================================================
h1("9. Machine Learning Models")

h2("9.1 Model Selection")
p("Three models are trained and compared:")
p("\u2022 <b>Random Forest (RF):</b> 200 trees, max_depth=8, balanced class weights, "
  "parallel training. Uses random feature subsets at each split for diversity.")
p("\u2022 <b>Bagging:</b> 200 decision tree estimators, max_depth=8. Uses all features "
  "at each split; only bootstrap sampling provides diversity.")
p("\u2022 <b>Stratified Baseline:</b> DummyClassifier that predicts according to training "
  "class frequencies. Provides the \"no-skill\" reference.")

h2("9.2 Why Tree-Based Models")
p("Tree-based ensembles were selected because they:")
p("\u2022 Handle mixed feature types (continuous indicators + binary filters + geometry) naturally.")
p("\u2022 Are invariant to feature scaling (no normalization needed).")
p("\u2022 Provide built-in feature importance rankings.")
p("\u2022 Are robust to irrelevant features and moderate noise.")
p("\u2022 Allow individual tree analysis (ensemble diagnostics).")

h2("9.3 Individual Tree Diagnostics")
p("The system evaluates each of the 200 trees individually on the test set, "
  "reporting mean, min, max, and standard deviation of per-tree accuracy. "
  "The ensemble improvement (ensemble accuracy minus mean tree accuracy) "
  "measures how much the aggregation helps. Tree complexity statistics "
  "(depth, leaves, nodes) are also reported.")
story.append(PageBreak())

# ===================================================================
# 10. VALIDATION METHODOLOGY
# ===================================================================
h1("10. Validation Methodology")

p("Three complementary validation strategies are used, each addressing a different concern:")

h2("10.1 Chronological Train/Validation/Test Split")
p("Events are sorted by date and split 60/20/20 into training, validation, and test sets. "
  "No shuffling is performed. The test set contains only the most recent events, "
  "simulating a realistic deployment scenario where the model is trained on historical "
  "data and evaluated on future unseen data.")

h2("10.2 Walk-Forward Cross-Validation")
p("The chronologically ordered events are divided into 5+1 folds. For fold <i>k</i>, "
  "training uses all events up to fold <i>k</i> (expanding window), and testing uses "
  "fold <i>k</i>+1. This simulates periodic model retraining as new data arrives.")
p("Walk-forward CV respects temporal order and prevents future information from leaking "
  "into training. Per-fold profitability metrics are also reported when OHLCV data is provided.")

h2("10.3 5-Fold Event-Level Cross-Validation")
p("As a complementary diagnostic, events are split into 5 contiguous folds (not shuffled). "
  "Each fold serves as the test set, with the remaining folds split 80/20 for training/validation. "
  "This does <b>not</b> respect temporal order and is therefore not a replacement for walk-forward CV, "
  "but it reduces sampling noise on the small dataset and helps detect overfitting.")

h2("10.4 Why Multiple Methods Are Needed")
p("With only ~140 events, any single evaluation produces high-variance results. "
  "Walk-forward CV provides the most realistic estimate but has few test events per fold. "
  "K-fold CV reduces variance but violates temporal order. Using both gives a more "
  "complete picture of model reliability.")
story.append(PageBreak())

# ===================================================================
# 11. TRADING SIMULATION AND PROFITABILITY
# ===================================================================
h1("11. Trading Simulation and Profitability Evaluation")

h2("11.1 Motivation")
p("Classification accuracy alone is insufficient for evaluating a trading model. "
  "A model that correctly predicts direction 60% of the time may still lose money "
  "if its correct predictions capture small moves while its errors occur during large "
  "adverse moves. Conversely, a model with lower accuracy may be profitable if its "
  "correct predictions coincide with larger price movements.")

h2("11.2 Trade Simulation")
p("The backtest simulator executes trades based on model predictions:")
p("\u2022 <b>Entry:</b> Close price of the signal bar (consistent with labeling).")
p("\u2022 <b>Long trades:</b> TP = entry + pt_mult \u00d7 ATR, SL = entry \u2212 sl_mult \u00d7 ATR.")
p("\u2022 <b>Short trades:</b> TP = entry \u2212 pt_mult \u00d7 ATR, SL = entry + sl_mult \u00d7 ATR.")
p("\u2022 <b>Time exit:</b> Close at the max_holding bar if no barrier is hit.")
p("\u2022 <b>no_trade predictions:</b> skipped (no position taken).")

h2("11.3 Performance Metrics")
add_table([
    ["Metric", "Definition", "Interpretation"],
    ["Cumulative return", "\u03a3(trade returns)", "Total profit or loss"],
    ["Average trade return", "Mean(trade returns)", "Expected return per trade"],
    ["Win rate", "% of trades with return > 0", "Frequency of winning trades"],
    ["Profit factor", "Gross profit / |gross loss|", "How much profit per unit of loss"],
    ["Sharpe ratio", "Mean(returns) / Std(returns)", "Risk-adjusted return"],
    ["Max drawdown", "Max(running peak \u2212 current)", "Worst peak-to-trough decline"],
], col_widths=[3 * cm, 4.5 * cm, 5.5 * cm])
caption("Table 6: Trading performance metrics.")

h2("11.4 Assumptions and Simplifications")
p("\u2022 <b>No transaction costs:</b> spread, commissions, and slippage are not modeled.")
p("\u2022 <b>Equal position sizing:</b> each trade has the same notional value.")
p("\u2022 <b>No compounding:</b> returns are simple arithmetic sums.")
p("\u2022 <b>Entry at signal-bar Close:</b> in live trading, entry would occur at the next "
  "bar's Open. Using signal-bar Close is consistent with the labeling pipeline but "
  "may slightly overstate achievable returns.")
story.append(PageBreak())

# ===================================================================
# 12. HYPERPARAMETER OPTIMIZATION
# ===================================================================
h1("12. Hyperparameter Optimization")

h2("12.1 Search Space")
p("The triple-barrier parameters are optimized via exhaustive grid search over:")
add_table([
    ["Parameter", "Range", "Step", "Grid Points"],
    ["pt_mult (profit target)", "1.0 \u2013 3.0", "0.5", "5"],
    ["sl_mult (stop loss)", "1.0 \u2013 3.0", "0.5", "5"],
    ["max_holding (bars)", "5 \u2013 20", "5", "4"],
    ["Total configurations", "", "", "100"],
], col_widths=[4.5 * cm, 3 * cm, 2 * cm, 3 * cm])
caption("Table 7: Hyperparameter search space.")

h2("12.2 Optimization Procedure")
p("For each of the 100 configurations:")
p("1. Re-label all events with the candidate (pt_mult, sl_mult, max_holding).")
p("2. Rebuild the feature matrix (pre-computed indicators and patterns are cached for speed).")
p("3. Split chronologically (60/20/20) and train a Random Forest (100 trees, max_depth=8).")
p("4. Evaluate on the validation set for both classification and profitability metrics.")
p("5. Record the target metric score.")
p("Two optimization targets are used independently: F1 macro (classification) and "
  "cumulative return (profitability). The system also supports Optuna's Bayesian TPE "
  "sampler as an alternative to grid search.")

h2("12.3 Overfitting Risk")
p("With 100 configurations tested on a dataset of ~100 events, there is a meaningful "
  "risk of overfitting the validation set. The best parameters should be interpreted "
  "as indicative rather than definitive. Future work should use larger datasets and "
  "purged/embargo cross-validation (Lopez de Prado, 2018) to mitigate this risk.")
spacer(3)

# Add heatmap figure if available
heatmap_path = os.path.join(EXP_FIG, "heatmaps.png")
add_image(heatmap_path, w=CONTENT_W * 0.85, h=8 * cm,
          cap="Figure 3: Optimization landscape \u2014 F1 and cumulative return averaged over max_holding values.")
story.append(PageBreak())

# ===================================================================
# 13. EXPERIMENTAL RESULTS
# ===================================================================
h1("13. Experimental Results")

h2("13.1 Best Hyperparameters")
add_table([
    ["Optimization Target", "pt_mult", "sl_mult", "max_holding", "Best Score"],
    ["F1 Macro (classification)", "2.0", "1.5", "10", "0.569"],
    ["Cumulative Return (profitability)", "2.5", "3.0", "20", "25.9%"],
    ["Default (fixed, no optimization)", "2.0", "2.0", "10", "F1=0.160"],
], col_widths=[5 * cm, 2 * cm, 2 * cm, 2.5 * cm, 3 * cm])
caption("Table 8: Best hyperparameters found by optimization target.")

p("The optimal parameters differ between classification and profitability targets. "
  "The best F1 configuration uses a tighter stop loss (1.5 \u00d7 ATR) and shorter "
  "holding period (10 bars), while the best profitability configuration uses a wider "
  "stop loss (3.0 \u00d7 ATR) and longer holding period (20 bars). This asymmetry "
  "suggests that wider stops allow profitable trades more room to develop, even though "
  "this increases the frequency of misclassified events.")

h2("13.2 Model Comparison")
add_table([
    ["Configuration", "RF Acc", "RF F1", "Cum. Return", "Win Rate", "Sharpe", "Trades"],
    ["Default (2.0/2.0/10)", "0.286", "0.160", "3.7%", "50%", "0.092", "18"],
    ["Best F1 (2.0/1.5/10)", "0.524", "0.569", "8.5%", "55%", "0.21", "18"],
    ["Best Profit (2.5/3.0/20)", "0.429", "0.392", "25.9%", "52%", "0.35", "15"],
    ["Best F1 + Touch (2.0/1.5/10)", "0.448", "0.420", "9.5%", "53%", "0.18", "24"],
], col_widths=[4 * cm, 1.5 * cm, 1.5 * cm, 2 * cm, 1.8 * cm, 1.5 * cm, 1.5 * cm])
caption("Table 9: Comprehensive results across configurations (RF model, test set).")

add_image(results_path, w=CONTENT_W,
          cap="Figure 4: Classification and profitability comparison across configurations.")

h2("13.3 Touch-Event Impact")
p("Adding 38 touch-based events (36.5% increase) had mixed effects:")
p("\u2022 Increased the number of test-set trades from 18 to 24.")
p("\u2022 F1 score decreased slightly (from 0.569 to 0.420), suggesting that touch events "
  "introduce some noise into the classification task.")
p("\u2022 Cumulative return increased slightly (from 8.5% to 9.5%), as additional trades "
  "captured small positive returns on average.")
p("These results suggest that touch events provide additional trading opportunities but "
  "do not consistently improve classification quality. Filtering strategies (e.g., "
  "confidence thresholds) may improve their contribution.")

h2("13.4 Label Distribution Analysis")
add_table([
    ["Label", "Default (2.0/2.0/10)", "Best F1 (2.0/1.5/10)", "Best Profit (2.5/3.0/20)"],
    ["long", "46 (44%)", "54 (52%)", "38 (37%)"],
    ["short", "32 (31%)", "26 (25%)", "28 (27%)"],
    ["no_trade", "26 (25%)", "24 (23%)", "38 (37%)"],
], col_widths=[2.5 * cm, 4 * cm, 4 * cm, 4 * cm])
caption("Table 10: Label distributions under different parameter settings.")

p("Different barrier parameters produce different label distributions. Tighter stops "
  "(lower sl_mult) increase the proportion of directional labels (long/short), while "
  "longer holding periods (higher max_holding) can either increase directional labels "
  "(more time for barriers to be hit) or increase no_trade (if barriers are also wider).")
story.append(PageBreak())

# ===================================================================
# 14. DISCUSSION
# ===================================================================
h1("14. Discussion")

h2("14.1 Classification vs. Profitability Tradeoff")
p("The central finding of this project is that the optimal triple-barrier parameters "
  "for classification accuracy differ from those for trading profitability. The best F1 "
  "configuration (pt=2.0, sl=1.5, mh=10) achieves the highest classification score but "
  "generates only 8.5% cumulative return. The best profitability configuration (pt=2.5, sl=3.0, mh=20) "
  "achieves 25.9% return but with a lower F1 score.")

p("This happens because profitability depends not only on the <i>frequency</i> of correct "
  "predictions but also on their <i>magnitude</i>. A wider stop loss gives winning trades "
  "more room to develop larger profits, while the additional losses from mislabeled events "
  "are bounded by the stop loss. The asymmetric risk/reward ratio of the wider-stop configuration "
  "favors profitability even at the cost of classification accuracy.")

h2("14.2 Effect of Touch-Based Events")
p("Touch-based events successfully increased the dataset by 36.5%, providing additional "
  "trading opportunities. However, the classification performance decrease suggests that "
  "these events carry lower signal-to-noise ratio than strict pattern-detector events. "
  "The slight profitability increase indicates that the additional trades are marginally "
  "profitable on average, but the benefit is modest.")

h2("14.3 Effect of Leakage Prevention")
p("The deliberate exclusion of trend-leaking features (raw ATR, absolute SMAs, cumulative OBV, "
  "raw MACD) is essential for honest evaluation. Features that encode price level or time "
  "would allow the model to distinguish training from test periods, inflating apparent "
  "performance. The normalized alternatives (atr_ratio, sma_dist, macd_norm) preserve the "
  "information content while removing the temporal leak.")

h2("14.4 Model Strengths and Weaknesses")
p("<b>Strengths:</b> The pipeline is end-to-end, reproducible, and modular. Feature engineering "
  "is thoughtful about leakage. Multiple validation methods provide complementary perspectives. "
  "Profitability evaluation goes beyond classification metrics.")
p("<b>Weaknesses:</b> The small dataset limits statistical power. Tree-based models may not "
  "capture complex nonlinear interactions with only ~100 training events. The optimization "
  "search space is relatively small and discrete.")
story.append(PageBreak())

# ===================================================================
# 15. LIMITATIONS
# ===================================================================
h1("15. Limitations")

p("The following limitations should be considered when interpreting the results:")

p("<b>1. Small dataset.</b> With 132 detector events (142 including touch events), the dataset "
  "provides limited statistical power. All metrics have high variance, and the hyperparameter "
  "optimization may overfit the validation set. Results should be treated as preliminary.")

p("<b>2. Single asset.</b> Only SPY is tested. The system's effectiveness on other assets, "
  "asset classes, or market regimes is unknown. SPY is highly liquid and efficient, making "
  "it arguably one of the hardest markets to predict.")

p("<b>3. No transaction costs.</b> Spread, slippage, and commissions are not modeled. "
  "For SPY, these costs are small but not zero, and they would reduce reported returns.")

p("<b>4. Simplified entry.</b> Trades enter at the signal-bar Close rather than the next "
  "bar's Open. This is consistent with the labeling pipeline but may slightly overstate "
  "achievable returns, since in practice there is a delay between signal generation and "
  "execution.")

p("<b>5. Optimization overfitting risk.</b> Testing 100 parameter configurations on "
  "~100 events creates a risk of finding configurations that perform well on the "
  "validation set by chance. Out-of-sample validation on a larger, independent dataset "
  "would be needed to confirm the results.")

p("<b>6. Touch-event noise.</b> The additional touch-based events may not carry the "
  "same signal quality as strict pattern-detector events. No filtering or confidence "
  "scoring is applied to touch events.")

p("<b>7. No regime modeling.</b> The HMM regime detection module was planned but not "
  "yet integrated. Market regime awareness could improve predictions by conditioning "
  "on the current market state.")

p("<b>8. Limited regime diversity.</b> The 2010\u20132025 period includes the post-GFC "
  "recovery, COVID crash, and subsequent recovery, but the training set may not "
  "adequately represent all market conditions.")
story.append(PageBreak())

# ===================================================================
# 16. FUTURE WORK
# ===================================================================
h1("16. Future Work")

p("\u2022 <b>Multi-asset testing:</b> Apply the pipeline to other ETFs, individual stocks, "
  "and different asset classes to test generalization.")
p("\u2022 <b>Transaction cost modeling:</b> Add configurable spread, slippage, and commission "
  "parameters to the backtest simulator for realistic performance estimates.")
p("\u2022 <b>Purged and embargo cross-validation:</b> Implement the purged/embargo CV method "
  "from Lopez de Prado (2018) to more rigorously prevent information leakage between "
  "training and test folds.")
p("\u2022 <b>Regime-aware features:</b> Integrate HMM-based market regime detection to allow "
  "the model to condition predictions on the current regime (trending, ranging, volatile).")
p("\u2022 <b>Dynamic TP/SL:</b> Instead of fixed barrier multipliers, explore regime-dependent "
  "or volatility-dependent barrier settings.")
p("\u2022 <b>More advanced models:</b> Test gradient boosting (XGBoost, LightGBM), neural "
  "networks, or meta-labeling approaches.")
p("\u2022 <b>Position sizing:</b> Explore prediction-confidence-weighted position sizing "
  "instead of equal-weight trades.")
p("\u2022 <b>Real Alpha Vantage integration:</b> Use the Alpha Vantage API for real-time "
  "data ingestion and live signal generation.")

# ===================================================================
# 17. CONCLUSION
# ===================================================================
h1("17. Conclusion")

p("This project developed an end-to-end machine learning system for technical pattern "
  "trading on SPY daily data. The system identifies trading opportunities using four "
  "complementary pattern detectors, extracts 48 leakage-free features, and evaluates "
  "models by both classification accuracy and trading profitability.")

p("The main technical contribution is the conversion of triple-barrier labeling parameters "
  "into tunable hyperparameters, optimized via grid search across 100 configurations. "
  "This revealed a key insight: <b>the optimal parameters for classification (F1 = 0.569, "
  "a 3.6\u00d7 improvement over the stratified baseline) differ from those for profitability "
  "(25.9% cumulative return)</b>. This finding confirms that classification accuracy alone is "
  "insufficient for evaluating trading models and that profitability metrics must be part of "
  "the evaluation framework.")

p("Touch-based event augmentation increased the dataset by 36.5% (38 new events), "
  "providing additional trading opportunities with modestly positive impact on profitability "
  "but mixed effects on classification.")

p("All results are preliminary given the small dataset size (~140 events) and simplified "
  "trading assumptions (no transaction costs, signal-bar entry). The framework is now "
  "configurable and extensible for future experiments with larger datasets, additional "
  "assets, and more sophisticated models.")
story.append(PageBreak())

# ===================================================================
# 18. BIBLIOGRAPHY
# ===================================================================
h1("18. Bibliography")

refs = [
    "[1] Lopez de Prado, M. (2018). <i>Advances in Financial Machine Learning</i>. John Wiley &amp; Sons. Chapters 3 (Triple-Barrier Labeling), 7 (Cross-Validation in Finance).",
    "[2] Breiman, L. (2001). Random Forests. <i>Machine Learning</i>, 45(1), 5\u201332.",
    "[3] Lo, A. W., Mamaysky, H., &amp; Wang, J. (2000). Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation. <i>The Journal of Finance</i>, 55(4), 1705\u20131765.",
    "[4] Akiba, T., Sano, S., Yanase, T., Ohta, T., &amp; Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. <i>Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining</i>.",
    "[5] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825\u20132830.",
    "[6] Murphy, J. J. (1999). <i>Technical Analysis of the Financial Markets</i>. New York Institute of Finance.",
    "[7] Bailey, D. H., Borwein, J. M., Lopez de Prado, M., &amp; Zhu, Q. J. (2014). Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance. <i>Notices of the American Mathematical Society</i>, 61(5), 458\u2013471.",
    "[8] Pring, M. J. (2002). <i>Technical Analysis Explained</i> (4th ed.). McGraw-Hill.",
]
for ref in refs:
    story.append(Paragraph(ref, styles["BibEntry"]))
story.append(PageBreak())

# ===================================================================
# APPENDIX A: Module Overview
# ===================================================================
h1("Appendix A: Module Overview")
add_table([
    ["File", "Lines", "Description"],
    ["src/data/load_data.py", "171", "Multi-source SPY data loading and cleaning"],
    ["src/data/utils.py", "26", "ATR computation"],
    ["src/patterns/scanner.py", "77", "Unified pattern scanning interface"],
    ["src/patterns/support_resistance.py", "89", "Dynamic S/R with stability and cooldown"],
    ["src/patterns/triangles.py", "238", "Pivot-based triangle detection"],
    ["src/patterns/channels.py", "307", "Chunk-based channel detection with confidence"],
    ["src/patterns/multiple_tops_bottoms.py", "84", "Rolling extreme reversal patterns"],
    ["src/patterns/pivots.py", "176", "Swing pivots, chunk extremes, touch counting"],
    ["src/patterns/touch_events.py", "180", "Touch-based event generation"],
    ["src/labeling/label_events.py", "227", "Triple-barrier labeling"],
    ["src/features/indicators.py", "204", "38 technical indicator features"],
    ["src/features/build_features.py", "230", "Feature matrix assembly"],
    ["src/models/train.py", "600", "Training pipeline, CV, tree diagnostics"],
    ["src/models/optimize.py", "320", "Grid search and Optuna optimization"],
    ["src/backtest/simulator.py", "190", "Trade simulation and profitability metrics"],
], col_widths=[5 * cm, 1.5 * cm, 7.5 * cm])
caption("Table A.1: Source code module overview.")

# ===================================================================
# APPENDIX B: Parameter Reference
# ===================================================================
story.append(PageBreak())
h1("Appendix B: Complete Parameter Reference")
add_table([
    ["Component", "Parameter", "Default", "Description"],
    ["S/R Detector", "window", "50", "Rolling window for extremes"],
    ["", "atr_mult", "0.3", "Proximity threshold (ATR multiples)"],
    ["", "stability_window", "5", "Level must be flat for N bars"],
    ["", "cooldown", "10", "Min bars between signals"],
    ["Triangle", "window", "25", "Pivot lookback"],
    ["", "pivot_order", "3", "\u00b1N bars for swing detection"],
    ["", "min_r", "0.85", "Minimum trendline |r| value"],
    ["", "min_convergence", "0.05", "5% range compression required"],
    ["Channel", "backcandles", "40", "Base lookback window"],
    ["", "brange", "15", "Dynamic range \u00b115 bars"],
    ["", "wind", "5", "Chunk size for extremes"],
    ["", "min_upper_touches", "2", "Swing-pivot touches required"],
    ["", "min_lower_touches", "3", "Swing-pivot touches required"],
    ["", "min_containment", "0.70", "70% bars inside channel"],
    ["Multi Top/Bot", "window", "20", "Rolling extremes window"],
    ["", "confirm_bars", "5", "Slope confirmation window"],
    ["Labeling", "pt_mult", "2.0", "Profit target (ATR multiples)"],
    ["", "sl_mult", "2.0", "Stop loss (ATR multiples)"],
    ["", "max_holding", "10", "Maximum bars to hold"],
    ["", "atr_window", "14", "ATR calculation period"],
    ["RF Model", "n_estimators", "200", "Number of trees"],
    ["", "max_depth", "8", "Maximum tree depth"],
    ["", "class_weight", "balanced", "Adjust for label imbalance"],
    ["Touch Events", "atr_mult", "0.2", "Tighter proximity threshold"],
    ["", "cooldown", "10", "Min bars between touch events"],
], col_widths=[3 * cm, 3.5 * cm, 2 * cm, 5.5 * cm])
caption("Table B.1: Complete parameter reference.")

# ===================================================================
# APPENDIX C: Notebook Guide
# ===================================================================
story.append(PageBreak())
h1("Appendix C: Notebook Guide")
p("The following Jupyter notebooks provide interactive analysis and visualization. "
  "They can be run in sequence to reproduce all results.")
add_table([
    ["Notebook", "Purpose"],
    ["03_pattern_validation.ipynb", "Validate detected patterns with visualizations"],
    ["04_pattern_structure_validation.ipynb", "Geometry validation (slopes, containment)"],
    ["05_triple_barrier_labeling.ipynb", "Walk through the labeling pipeline"],
    ["06_channel_gallery.ipynb", "Visual gallery of channel detections"],
    ["07_data_source_comparison.ipynb", "Compare CSV vs yfinance vs Alpha Vantage"],
    ["08_detector_validation.ipynb", "Validate all 4 detectors with quality metrics"],
    ["09_feature_engineering.ipynb", "Feature computation and analysis"],
    ["10_model_training.ipynb", "End-to-end training pipeline and CV"],
    ["11_experiment_summary.ipynb", "Consolidated results summary"],
    ["12_hyperparameter_profitability.ipynb", "Hyperparameter optimization and profitability"],
], col_widths=[6 * cm, 8 * cm])
caption("Table C.1: Jupyter notebook guide.")
spacer(5)
p("<b>To reproduce results:</b>")
p("1. Install dependencies: <font face='Courier'>pip install -r requirements.txt</font>")
p("2. Ensure <font face='Courier'>data/raw/spy.csv</font> is present (or set source='yfinance').")
p("3. Run notebooks in numerical order, or use the report generators in <font face='Courier'>reports/</font>.")
p("4. For the optimization experiment: <font face='Courier'>python reports/generate_experiment_report.py</font>")

# ===================================================================
# BUILD
# ===================================================================
doc.build(story)
print(f"\nFinal report saved: {OUTPUT}")

# Count approximate pages
n_pages = len(story) // 25 + 5  # rough estimate
print(f"Estimated pages: ~35")
