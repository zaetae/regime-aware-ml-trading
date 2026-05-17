"""Generate a 10-minute presentation PDF (10–12 slides).

Produces: reports/final/Zeineb_Turki_bem.pdf

Uses ReportLab to build slide-style pages with diagrams, tables, and figures.
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

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, Frame, PageTemplate, BaseDocTemplate,
    KeepTogether,
)
from reportlab.lib import colors

# ---------------------------------------------------------------------------
# Page dimensions
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = landscape(A4)  # 29.7 x 21 cm
MARGIN = 1.5 * cm

# Colours
BLUE = HexColor("#1B3A5C")
ACCENT = HexColor("#2E86C1")
LIGHT_BG = HexColor("#EBF5FB")
DARK_TEXT = HexColor("#1C2833")
GREY = HexColor("#808B96")
WHITE = white

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()

styles.add(ParagraphStyle("SlideTitle", parent=styles["Heading1"],
                          fontSize=24, leading=30, textColor=BLUE,
                          alignment=TA_LEFT, spaceAfter=6))
styles.add(ParagraphStyle("SlideSubtitle", parent=styles["Normal"],
                          fontSize=14, leading=18, textColor=ACCENT,
                          spaceAfter=10))
styles.add(ParagraphStyle("SlideBullet", parent=styles["Normal"],
                          fontSize=12, leading=16, textColor=DARK_TEXT,
                          leftIndent=20, bulletIndent=10, spaceBefore=3,
                          spaceAfter=3))
styles.add(ParagraphStyle("SlideBody", parent=styles["Normal"],
                          fontSize=11, leading=15, textColor=DARK_TEXT,
                          spaceAfter=6))
styles.add(ParagraphStyle("SlideSmall", parent=styles["Normal"],
                          fontSize=9, leading=12, textColor=GREY))
styles.add(ParagraphStyle("TitleSlideTitle", parent=styles["Title"],
                          fontSize=30, leading=36, textColor=BLUE,
                          alignment=TA_CENTER))
styles.add(ParagraphStyle("TitleSlideSubtitle", parent=styles["Normal"],
                          fontSize=16, leading=20, textColor=ACCENT,
                          alignment=TA_CENTER))
styles.add(ParagraphStyle("TitleSlideInfo", parent=styles["Normal"],
                          fontSize=12, leading=16, textColor=DARK_TEXT,
                          alignment=TA_CENTER))
styles.add(ParagraphStyle("SlideNumber", parent=styles["Normal"],
                          fontSize=8, textColor=GREY, alignment=TA_RIGHT))

# ---------------------------------------------------------------------------
# Image paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(__file__)
IMG = os.path.join(BASE, "images")
THESIS_FIG = os.path.join(BASE, "thesis_figures")
EXP_FIG = os.path.join(BASE, "experiment_figures")
FINAL = os.path.join(BASE, "final")
PRES_FIG = os.path.join(FINAL, "pres_figures")
os.makedirs(PRES_FIG, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: generate pipeline diagram
# ---------------------------------------------------------------------------
def _make_pipeline_figure():
    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.6)
    ax.axis("off")

    boxes = [
        (0.2, "SPY\nOHLCV"),
        (1.55, "Pattern\nDetectors"),
        (2.9, "Event\nSignals"),
        (4.25, "Feature\nEngineering"),
        (5.6, "Triple-Barrier\nLabeling"),
        (6.95, "ML\nModels"),
        (8.3, "Validation\n& Backtest"),
    ]
    bw, bh = 1.15, 0.9
    for x, txt in boxes:
        rect = mpatches.FancyBboxPatch((x, 0.35), bw, bh, boxstyle="round,pad=0.1",
                                        facecolor="#2E86C1", edgecolor="#1B3A5C", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + bw / 2, 0.35 + bh / 2, txt, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + bw
        x2 = boxes[i + 1][0]
        ax.annotate("", xy=(x2, 0.8), xytext=(x1, 0.8),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#1B3A5C"))

    path = os.path.join(PRES_FIG, "pipeline.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def _make_results_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    configs = ["Default\n(2.0/2.0/10)", "Best F1\n(2.0/1.5/10)",
               "Best Profit\n(2.5/3.0/20)", "Best F1\n+ Touch"]
    x = range(4)

    f1_vals = [0.160, 0.569, 0.392, 0.420]
    cum_ret = [0.037, 0.085, 0.259, 0.095]
    win_rate = [0.50, 0.55, 0.52, 0.53]

    axes[0].bar(x, f1_vals, color=["#808B96", "#2E86C1", "#E67E22", "#27AE60"], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs, fontsize=8)
    axes[0].set_ylabel("F1 Macro", fontsize=10)
    axes[0].set_title("Classification", fontsize=11, fontweight="bold")

    bar_colors = ["green" if v > 0 else "red" for v in cum_ret]
    axes[1].bar(x, cum_ret, color=["#808B96", "#2E86C1", "#E67E22", "#27AE60"], alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs, fontsize=8)
    axes[1].set_ylabel("Cumulative Return", fontsize=10)
    axes[1].set_title("Profitability", fontsize=11, fontweight="bold")
    axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.3)

    axes[2].bar(x, win_rate, color=["#808B96", "#2E86C1", "#E67E22", "#27AE60"], alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(configs, fontsize=8)
    axes[2].set_ylabel("Win Rate", fontsize=10)
    axes[2].set_title("Win Rate", fontsize=11, fontweight="bold")
    axes[2].axhline(y=0.5, color="black", linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PRES_FIG, "results_comparison.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path

# ---------------------------------------------------------------------------
# Generate figures
# ---------------------------------------------------------------------------
print("Generating presentation figures...")
pipeline_img = _make_pipeline_figure()
results_img = _make_results_comparison()

# ---------------------------------------------------------------------------
# Build slides
# ---------------------------------------------------------------------------
print("Building presentation PDF...")

OUTPUT = os.path.join(FINAL, "Zeineb_Turki_bem2.pdf")

doc = SimpleDocTemplate(
    OUTPUT, pagesize=landscape(A4),
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN,
)

story = []

def slide_break():
    story.append(PageBreak())

def slide_title(title, subtitle=None):
    story.append(Paragraph(title, styles["SlideTitle"]))
    if subtitle:
        story.append(Paragraph(subtitle, styles["SlideSubtitle"]))
    story.append(Spacer(1, 4 * mm))

def bullet(text):
    story.append(Paragraph(f"\u2022  {text}", styles["SlideBullet"]))

def body(text):
    story.append(Paragraph(text, styles["SlideBody"]))

def small(text):
    story.append(Paragraph(text, styles["SlideSmall"]))

def add_table(data, col_widths=None, header_color=BLUE):
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, GREY),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])
    t = Table(data, colWidths=col_widths)
    t.setStyle(style)
    story.append(t)

def add_image(path, w=None, h=None):
    if os.path.exists(path):
        if w is None:
            w = PAGE_W - 2 * MARGIN - 1 * cm
        story.append(Image(path, width=w, height=h))


# ===== SLIDE 1: Title =====
story.append(Spacer(1, 4 * cm))
story.append(Paragraph("Regime-Aware Machine Learning", styles["TitleSlideTitle"]))
story.append(Paragraph("for Technical Pattern Trading", styles["TitleSlideTitle"]))
story.append(Spacer(1, 1 * cm))
story.append(Paragraph("Event-Based Classification and Profitability Evaluation on SPY", styles["TitleSlideSubtitle"]))
story.append(Spacer(1, 1.5 * cm))
story.append(Paragraph("Zeineb Turki", styles["TitleSlideInfo"]))
story.append(Spacer(1, 3 * mm))
story.append(Paragraph("Supervisor: [Supervisor Name]", styles["TitleSlideInfo"]))
story.append(Paragraph("[University / Course Name]", styles["TitleSlideInfo"]))
story.append(Paragraph("Summer Semester 2026", styles["TitleSlideInfo"]))

# ===== SLIDE 2: Problem Motivation =====
slide_break()
slide_title("Problem Motivation", "Why financial ML needs event-based approaches")
bullet("<b>Financial markets are extremely noisy</b> — most bars contain no actionable signal")
bullet("<b>Naive price prediction</b> models learn noise, price level, or time trends")
bullet("<b>Event-based learning</b> focuses on <i>technically meaningful moments</i>")
story.append(Spacer(1, 6 * mm))
add_table([
    ["Approach", "Prediction Target", "Key Problem"],
    ["Naive regression", "Next bar's close price", "Learns noise & trends"],
    ["Bar-level classification", "Up/down per bar", "No trading context"],
    ["Event-based (this work)", "Trade outcome at signal bars", "Focused, actionable"],
], col_widths=[5.5 * cm, 6 * cm, 6 * cm])
story.append(Spacer(1, 8 * mm))
body("Core question: <i>When price is at a technically meaningful structure, "
     "can we predict the outcome of a trade-sized move?</i>")

# ===== SLIDE 3: Background =====
slide_break()
slide_title("Background", "Foundations of the approach")
bullet("<b>Technical analysis patterns:</b> support/resistance, channels, triangles, multiple tops/bottoms")
bullet("<b>Triple-barrier labeling</b> (Lopez de Prado, 2018): labels based on trade outcomes, not arbitrary direction")
bullet("<b>Random Forest / Bagging:</b> ensemble tree models — robust, interpretable, handle mixed features")
bullet("<b>Walk-forward validation:</b> simulates realistic deployment with expanding training windows")
bullet("<b>Profitability evaluation:</b> backtesting with TP/SL/time exits — accuracy alone is insufficient")
story.append(Spacer(1, 6 * mm))
body("<b>Key insight from financial ML literature:</b> the best classifier is not necessarily the best trading model. "
     "Profitability depends on the <i>magnitude</i> of correct predictions, not just their frequency.")

# ===== SLIDE 4: System Architecture =====
slide_break()
slide_title("System Architecture", "End-to-end pipeline from raw data to profitability evaluation")
add_image(pipeline_img, w=25 * cm, h=4.5 * cm)
story.append(Spacer(1, 6 * mm))
add_table([
    ["Stage", "Module", "Output"],
    ["Data", "load_data.py", "4,023 SPY daily bars (2010-2025)"],
    ["Detection", "scanner.py + 4 detectors", "132 event signals"],
    ["Features", "indicators.py + build_features.py", "48 features per event"],
    ["Labeling", "label_events.py", "long / short / no_trade labels"],
    ["Training", "train.py + optimize.py", "RF, Bagging, Baseline models"],
    ["Evaluation", "simulator.py + train.py", "Classification + profitability metrics"],
], col_widths=[2.5 * cm, 6 * cm, 8.5 * cm])

# ===== SLIDE 5: Pattern Detection =====
slide_break()
slide_title("Pattern Detection", "Four complementary detectors identify trading opportunities")
add_table([
    ["Detector", "Method", "Signals", "Key Parameters"],
    ["Support / Resistance", "Rolling max/min + stability filter", "42 events", "window=50, 0.3\u00d7ATR, cooldown=10"],
    ["Triangles", "Swing pivots + linear regression", "17 events", "window=25, |r|\u22650.85, convergence\u22655%"],
    ["Channels", "Chunk extremes + polyfit", "12 events", "lookback=40\u00b115, \u22652+3 touches"],
    ["Multiple Tops/Bottoms", "Rolling extremes + trend confirm", "63 events", "window=20, 5-bar slope confirm"],
    ["TOTAL", "", "132 events", "3.3% event density"],
], col_widths=[4.5 * cm, 5.5 * cm, 2.5 * cm, 7 * cm])
story.append(Spacer(1, 5 * mm))
bullet("All detectors use <b>cooldown filters</b> (10 bars) to prevent signal clustering")
bullet("Signal localized at the <b>event bar</b> — the bar where the pattern condition is met")
bullet("Per supervisor feedback: triangles and channels excluded from training labels")

# ===== SLIDE 6: Features & Leakage Prevention =====
slide_break()
slide_title("Feature Engineering", "48 features with strict leakage prevention")
add_table([
    ["Feature Group", "Count", "Examples"],
    ["Volatility & Returns", "9", "atr_ratio, rvol_20, ret_1/5/10/20, mom_5/10/20"],
    ["Moving Average Distances", "8", "sma_10_dist ... sma_200_dist, MA spreads"],
    ["Momentum Indicators", "4", "rsi_14, macd_norm, macd_signal_norm, macd_hist_norm"],
    ["Bollinger Bands", "2", "bb_width, bb_pctb"],
    ["Binary Technical Filters", "8", "bb_touch_upper/lower, SMA cross, RSI extremes"],
    ["Volume", "2", "volume_ratio, volume_std"],
    ["Pattern Geometry", "11", "slopes, touches, containment, width, R\u00b2"],
    ["Event Type Dummies", "4\u20136", "One-hot encoded pattern type"],
    ["TOTAL", "48\u201350", ""],
], col_widths=[5 * cm, 1.5 * cm, 12.5 * cm])
story.append(Spacer(1, 5 * mm))
body("<b>Removed (trend-leaking):</b> raw ATR, absolute SMA values, cumulative OBV, raw MACD. "
     "All features use relative, normalized, or bounded values only.")

# ===== SLIDE 7: Labeling & Hyperparameter Optimization =====
slide_break()
slide_title("Triple-Barrier Labeling & Optimization",
            "TP/SL/max_holding treated as tunable hyperparameters")
bullet("<b>Triple-barrier method:</b> for each event, walk forward and check barriers")
story.append(Spacer(1, 3 * mm))
add_table([
    ["Barrier", "Definition", "Label if hit first"],
    ["Upper (TP)", "entry_price + pt_mult \u00d7 ATR", "long"],
    ["Lower (SL)", "entry_price \u2212 sl_mult \u00d7 ATR", "short"],
    ["Time", "max_holding bars elapsed", "no_trade"],
], col_widths=[3.5 * cm, 7 * cm, 5 * cm])
story.append(Spacer(1, 5 * mm))
body("<b>Innovation:</b> pt_mult, sl_mult, max_holding are <i>not fixed</i> — they are optimized via grid search (100 configurations).")
story.append(Spacer(1, 3 * mm))
add_table([
    ["Search Space", "Range", "Best for F1", "Best for Profit"],
    ["pt_mult", "1.0 \u2013 3.0", "2.0", "2.5"],
    ["sl_mult", "1.0 \u2013 3.0", "1.5", "3.0"],
    ["max_holding", "5 \u2013 20 bars", "10", "20"],
    ["Score", "", "F1 = 0.569", "Return = 25.9%"],
], col_widths=[3.5 * cm, 3 * cm, 4.5 * cm, 4.5 * cm])

# ===== SLIDE 8: ML & Validation =====
slide_break()
slide_title("Machine Learning & Validation", "Three models, three validation strategies")
bullet("<b>Models:</b> Random Forest (200 trees), Bagging (200 trees), Stratified Baseline")
bullet("<b>Class weighting:</b> balanced (adjusts for label imbalance)")
story.append(Spacer(1, 3 * mm))
add_table([
    ["Validation Method", "Description", "Purpose"],
    ["Chronological split", "60% train / 20% val / 20% test", "Primary evaluation"],
    ["Walk-forward CV", "5 expanding folds, temporal order", "Realistic deployment simulation"],
    ["5-fold event CV", "Rotate test fold, 80/20 train/val", "Reduce sampling noise (diagnostic)"],
    ["Tree diagnostics", "Per-tree accuracy, ensemble gain", "Detect ensemble dependency"],
], col_widths=[4 * cm, 7.5 * cm, 6 * cm])
story.append(Spacer(1, 5 * mm))
bullet("Walk-forward CV respects temporal order — no future data in training")
bullet("Individual tree analysis shows whether ensemble relies on few strong trees or distributes strength")

# ===== SLIDE 9: Profitability Evaluation =====
slide_break()
slide_title("Trading Simulation & Profitability", "Why accuracy alone is insufficient")
bullet("<b>Simulated trading:</b> enter at signal-bar close, exit at TP/SL/time barrier")
bullet("A model with <b>lower accuracy</b> can be <b>more profitable</b> if correct predictions capture larger moves")
story.append(Spacer(1, 3 * mm))
add_table([
    ["Metric", "Definition", "Interpretation"],
    ["Cumulative return", "\u03a3(net returns)", "Total P&L"],
    ["Win rate", "% trades with positive return", "Hit rate"],
    ["Profit factor", "Gross profit / gross loss", "Reward-to-risk ratio"],
    ["Sharpe ratio", "Mean return / std(returns)", "Risk-adjusted performance"],
    ["Max drawdown", "Largest peak-to-trough decline", "Worst-case scenario"],
], col_widths=[3.5 * cm, 6 * cm, 6 * cm])
story.append(Spacer(1, 5 * mm))
small("<b>Assumptions:</b> no transaction costs, no slippage, equal position sizing, no compounding. "
      "Entry at signal-bar Close (consistent with labeling pipeline).")

# ===== SLIDE 10: Results =====
slide_break()
slide_title("Experimental Results", "104 detector events + 38 touch events = 142 total")
add_image(results_img, w=24 * cm, h=6.5 * cm)
story.append(Spacer(1, 4 * mm))
add_table([
    ["Configuration", "pt / sl / mh", "F1 Macro", "Cum. Return", "Win Rate", "Trades"],
    ["Default (baseline)", "2.0 / 2.0 / 10", "0.160", "3.7%", "50%", "18"],
    ["Best for F1", "2.0 / 1.5 / 10", "0.569", "8.5%", "55%", "18"],
    ["Best for Profit", "2.5 / 3.0 / 20", "0.392", "25.9%", "52%", "15"],
    ["Best F1 + Touch", "2.0 / 1.5 / 10", "0.420", "9.5%", "53%", "24"],
], col_widths=[4 * cm, 3 * cm, 2.5 * cm, 3 * cm, 2.5 * cm, 2 * cm])
story.append(Spacer(1, 3 * mm))
body("<b>Key finding:</b> optimal parameters for classification (F1) differ from those for profitability (return). "
     "This confirms that trading evaluation requires profitability metrics beyond accuracy.")

# ===== SLIDE 11: Generalization & Variance =====
slide_break()
slide_title("Generalization & Variance Analysis", "How stable are results across time folds?")
bullet("Walk-forward CV (5 folds): <b>mean \u00b1 std</b> reported for all metrics")
bullet("High variance across folds = instability \u2014 results depend on which period is tested")
story.append(Spacer(1, 3 * mm))
add_table([
    ["Metric", "WF Mean", "WF Std", "Interpretation"],
    ["F1 Macro", "\u22480.27", "\u22480.03", "Moderate stability"],
    ["Cum. Return", "\u22480.03", "\u22480.03", "High relative variance"],
    ["Win Rate", "\u224850%", "\u22487%", "Thin edge, regime-dependent"],
    ["Sharpe", "\u22480.11", "\u22480.14", "Unstable risk-adjusted return"],
], col_widths=[3.5 * cm, 3 * cm, 3 * cm, 6 * cm])
story.append(Spacer(1, 4 * mm))
body("<b>Key insight:</b> point estimates like \u2018F1=0.569\u2019 represent one realisation from a "
     "high-variance distribution. The true out-of-sample performance could be substantially different.")

# ===== SLIDE 12: F-Beta Analysis =====
slide_break()
slide_title("F-Beta Analysis", "Precision vs recall for trading decisions")
bullet("<b>F0.5 (precision-heavy):</b> fewer but cleaner trades \u2014 avoid false positives")
bullet("<b>F1.0 (balanced):</b> equal weight to false positives and false negatives")
bullet("<b>F2.0 (recall-heavy):</b> capture more opportunities \u2014 tolerate false positives")
story.append(Spacer(1, 3 * mm))
add_table([
    ["Error Type", "Trading Meaning", "Cost"],
    ["False Positive", "Model says trade, but it loses", "Direct financial loss"],
    ["False Negative", "Model says skip, but it would have won", "Missed opportunity"],
], col_widths=[3.5 * cm, 7 * cm, 5 * cm])
story.append(Spacer(1, 4 * mm))
body("<b>Finding:</b> Precision remains low across all folds, meaning many signals are false alarms. "
     "The optimal beta depends on risk tolerance: capital-preservation strategies should favour F0.5, "
     "opportunity-seeking strategies should favour F2.0.")

# ===== SLIDE 13: Limitations =====
slide_break()
slide_title("Limitations", "Honest assessment of constraints")
bullet("<b>Small dataset:</b> ~140 events provides limited statistical power; high result variance")
bullet("<b>Single asset:</b> only SPY tested — generalization to other markets is unknown")
bullet("<b>No transaction costs:</b> spread, slippage, and commissions are not modeled")
bullet("<b>Simplified entry:</b> signal-bar Close, not next-bar Open (slight optimism)")
bullet("<b>Optimization overfitting risk:</b> 100 configurations on ~100 events")
bullet("<b>Touch-event noise:</b> additional events may not carry the same signal quality as strict detectors")
bullet("<b>No regime modeling:</b> HMM regime detection planned but not yet integrated")
story.append(Spacer(1, 8 * mm))
body("All results should be interpreted as <i>preliminary research findings</i>, "
     "not production-ready trading signals.")

# ===== SLIDE 14: Conclusion & Future Work =====
slide_break()
slide_title("Conclusion & Future Work")
body("<b>What was achieved:</b>")
bullet("End-to-end pipeline: data \u2192 detection \u2192 features \u2192 labeling \u2192 ML \u2192 validation \u2192 backtest")
bullet("48 leakage-free features from 4 pattern detectors on 4,023 SPY bars")
bullet("Triple-barrier parameters converted from fixed constants to tunable hyperparameters")
bullet("Profitability evaluation integrated into training and cross-validation")
bullet("3.6\u00d7 F1 improvement over baseline; 25.9% cumulative return (best profit config)")
story.append(Spacer(1, 5 * mm))
body("<b>Main scientific lesson:</b>")
bullet("Classification accuracy and trading profitability optimize at <i>different</i> parameter settings")
story.append(Spacer(1, 5 * mm))
body("<b>Future work:</b>")
bullet("Multi-asset testing, transaction cost modeling, purged/embargo CV")
bullet("Regime-aware dynamic TP/SL, HMM integration, more advanced models")

# ===== BUILD =====
doc.build(story)
print(f"\nPresentation saved: {OUTPUT}")
print(f"Slides: 14")
