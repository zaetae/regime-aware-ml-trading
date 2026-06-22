"""BEM research-defense presentation, visually dense, notebook-backed.

12 slides, vertical A4, 27+ embedded figures.
Narrative: problem → solution → experiment → finding → next problem.

Produces: reports/final/Zeineb_Turki_bem3.pdf (overwrites)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings; warnings.filterwarnings("ignore")

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)
from reportlab.lib import colors

BASE = os.path.dirname(__file__)
FIG  = os.path.join(BASE, "final", "figures")
OUT  = os.path.join(BASE, "final", "Zeineb_Turki_bem3.pdf")
PW, PH = A4; M = 1.4*cm; CW = PW - 2*M

BLUE=HexColor("#1B3A5C"); ACC=HexColor("#2E86C1"); GREY=HexColor("#808B96")
LIGHT=HexColor("#EBF5FB"); DARK=HexColor("#1C2833")
GREEN=HexColor("#27AE60"); RED=HexColor("#E74C3C"); ORG=HexColor("#E67E22")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("ST", fontSize=22, leading=28, textColor=BLUE,
                          alignment=TA_CENTER, spaceAfter=3, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("SS", fontSize=12, leading=16, textColor=ACC,
                          alignment=TA_CENTER, spaceAfter=6))
styles.add(ParagraphStyle("SI", fontSize=11, leading=14, alignment=TA_CENTER, spaceAfter=2))
styles.add(ParagraphStyle("SH", fontSize=16, leading=20, textColor=BLUE, spaceAfter=2,
                          fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("SH2", fontSize=10.5, leading=13, textColor=ACC, spaceAfter=3,
                          fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("SB", fontSize=10, leading=13.5, textColor=DARK, spaceAfter=2))
styles.add(ParagraphStyle("SBul", fontSize=10, leading=13, textColor=DARK,
                          leftIndent=12, bulletIndent=4, spaceAfter=1.5))
styles.add(ParagraphStyle("STk", fontSize=10, leading=13, textColor=GREEN,
                          fontName="Helvetica-BoldOblique", spaceAfter=3,
                          borderColor=GREEN, borderWidth=0.5, borderPadding=4,
                          backColor=HexColor("#E8F8F5")))
styles.add(ParagraphStyle("SPr", fontSize=10, leading=13, textColor=RED,
                          fontName="Helvetica-BoldOblique", spaceAfter=3,
                          borderColor=RED, borderWidth=0.5, borderPadding=4,
                          backColor=HexColor("#FDEDEC")))
styles.add(ParagraphStyle("SC", fontSize=7.5, leading=9, textColor=GREY,
                          alignment=TA_CENTER, spaceAfter=2))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=M, rightMargin=M,
                        topMargin=M, bottomMargin=M)
story = []

def sl(t, s=None):
    story.append(Paragraph(t, styles["SH"]))
    if s: story.append(Paragraph(s, styles["SH2"]))
    story.append(Spacer(1, 0.5*mm))
def bul(t): story.append(Paragraph(f"\u2022 {t}", styles["SBul"]))
def body(t): story.append(Paragraph(t, styles["SB"]))
def take(t): story.append(Paragraph(f"\u2794 {t}", styles["STk"]))
def prob(t): story.append(Paragraph(f"\u26a0 {t}", styles["SPr"]))
def cap(t): story.append(Paragraph(t, styles["SC"]))
def sp(h=2): story.append(Spacer(1, h*mm))
def img(name, w=None, h=None, c=None):
    p = os.path.join(FIG, name)
    if os.path.exists(p):
        story.append(Image(p, width=w or CW, height=h))
        if c: cap(c)
def tbl(data, widths=None):
    s = TableStyle([
        ("BACKGROUND",(0,0),(-1,0),BLUE),("TEXTCOLOR",(0,0),(-1,0),white),
        ("FONTSIZE",(0,0),(-1,0),8),("FONTSIZE",(0,1),(-1,-1),8),
        ("GRID",(0,0),(-1,-1),0.3,GREY),("ALIGN",(1,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[white,LIGHT]),
        ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),
    ])
    t = Table(data, colWidths=widths); t.setStyle(s); story.append(t)

# ═══════════════════════════════════════════════════════
# SLIDE 1, Title
# ═══════════════════════════════════════════════════════
story.append(Spacer(1, 2.5*cm))
story.append(Paragraph("Regime-Aware Machine Learning<br/>for Technical Pattern Trading", styles["ST"]))
sp(5)
story.append(Paragraph("An Investigation into Event-Based Classification,<br/>"
                        "Profitability Evaluation, and Generalization on SPY Daily Data", styles["SS"]))
story.append(Spacer(1, 1*cm))
story.append(Paragraph("Zeineb Turki", styles["SI"]))
sp(1)
story.append(Paragraph("Supervisor: Hadh\u00e1zi D\u00e1niel", styles["SI"]))
story.append(Paragraph("Budapest University of Technology and Economics, 2026", styles["SI"]))
story.append(Spacer(1, 1*cm))
body("<b>Context:</b> Financial markets are among the hardest prediction domains in ML. "
     "Most daily price bars are noise. This project investigates whether focusing on "
     "technically meaningful events, rather than predicting every bar, can produce "
     "measurable classification and trading performance.")
sp(3)
body("<b>Research questions:</b>")
bul("Can pattern-filtered ML beat a random baseline on trade-outcome prediction?")
bul("Do classification-optimal parameters also maximise trading profit?")
bul("How stable are results across different market regimes?")
sp(3)
body("<b>Methodology:</b> 4 pattern detectors \u2192 48 features \u2192 triple-barrier labels "
     "\u2192 Random Forest \u2192 walk-forward CV \u2192 profitability simulation. "
     "Evaluated on 4,023 SPY daily bars (2010-2025), producing 132 events.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════
# SLIDE 2, The Problem
# ═══════════════════════════════════════════════════════
sl("The Problem: Markets Are Hostile to ML")
img("noise_regime.png", w=CW, h=8*cm,
    c="SPY COVID crash: \u221234% in 23 days. Daily returns swing \u221212% to +9%.")
body("The top panel shows SPY's price collapsing from $337 to $222 in just 23 trading days. "
     "The bottom panel reveals the daily return distribution during this period, a pattern "
     "no stationary model can anticipate. Most individual bars are indistinguishable from random noise.")
sp(1)
img("volatility_regimes.png", w=CW, h=4.5*cm,
    c="Rolling 20-day annualised volatility across the full 2010\u20132025 dataset.")
body("Volatility is not constant. The 2017 market had annualised vol below 5%; during COVID it "
     "spiked above 90%. A model trained during one regime faces entirely different statistical "
     "properties in the next. This non-stationarity is the fundamental enemy of financial ML.")
sp(1)
prob("Naive bar-by-bar models learn price trends and time artefacts rather than genuine trading edges. "
     "When the regime shifts, they fail catastrophically.")
take("Our approach: detect technically meaningful events and predict only at those moments. "
     "This reduces 4,023 noisy bars to ~100 focused trading candidates.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════
# SLIDE 3, The Solution Chain
# ═══════════════════════════════════════════════════════
sl("Research Design", "Each solution creates the next problem")
img("problem_solution_chain.png", w=CW*0.8, h=12*cm,
    c="The causal chain that shaped the system design.")
body("Read top-to-bottom: raw bars are too noisy, so we introduce event detection. "
     "Subjective patterns need algorithmic formalisation. Naive up/down labels ignore "
     "trade outcomes, so we adopt triple-barrier labeling. Fixed barrier parameters embed "
     "untested assumptions, so we optimise them. Accuracy can mislead, so we simulate trading. "
     "Standard CV leaks future data, so we use walk-forward validation. Each problem "
     "motivated the next design decision.")
sp(1)
take("The pipeline is not arbitrary, every component exists because the previous step was insufficient.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════
# SLIDE 4, Event Detection + Case Studies
# ═══════════════════════════════════════════════════════
sl("Step 1: Event Detection", "What does the system actually see?")
img("spy_events.png", w=CW, h=4.5*cm,
    c="SPY close price (2010\u20132025) with 104 detected events overlaid in red.")
body("The red dots are sparse and intentional. Four independent detectors (support/resistance, "
     "channels, triangles, multiple tops/bottoms) collectively flag only 3.3% of all bars. "
     "The remaining 97% is discarded as noise. This aggressive filtering is the foundation "
     "of the event-based approach.")
sp(1)
img("case_sr.png", w=CW, h=5.5*cm,
    c="Case study: a real support/resistance event detected on SPY.")
body("The red triangle marks where the S/R detector fired. The green dashed line is the "
     "take-profit barrier (entry + 2.0\u00d7ATR) and the red dashed line is the stop-loss "
     "(entry \u2212 1.5\u00d7ATR). The horizontal green line shows the support level the "
     "detector identified. This event illustrates how the system translates a technical "
     "pattern into a concrete trade setup with defined risk.")
sp(1)
img("case_mt.png", w=CW*0.95, h=5*cm,
    c="Case study: a multiple-top reversal pattern with barrier overlay.")
body("Here the rolling 20-bar high formed a ceiling while the 5-bar close slope turned "
     "negative, a classic exhaustion reversal. The detector captures the moment where "
     "buyers can no longer push price higher, suggesting a directional opportunity.")
sp(1)
take("132 events from 4 detectors. Each uses ATR-based thresholds for scale independence "
     "and 10-bar cooldown to prevent duplicate signals. Triangles/channels excluded per supervisor feedback.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════
# SLIDE 5, What The Model Sees (Features + Indicators)
# ═══════════════════════════════════════════════════════
sl("Step 2: What The Model Sees", "48 features: indicators, geometry, event type")
img("indicator_examples.png", w=CW, h=8.5*cm,
    c="Four-panel indicator view of SPY during the 2022 bear market onset.")
body("The top panel shows price declining from January through June 2022. The RSI panel "
     "highlights oversold zones (below 30, shaded green) where mean-reversion signals fire. "
     "The normalised MACD captures the momentum shift from positive to deeply negative. "
     "Bollinger Band width widens as volatility spikes. These are the signals the model "
     "receives at each event bar, not raw prices, but derived, bounded indicators.")
sp(1)
img("feature_grouped.png", w=CW, h=5.5*cm,
    c="Feature importance: grouped (left) and individual top 12 (right).")
body("Trend features (MA distances) and momentum features (returns, rate-of-change) together "
     "account for over 60% of total importance. Pattern geometry features, slopes, touches, "
     "containment, contribute less than expected. The model relies more on the <i>market context</i> "
     "around a pattern than on the pattern's geometric shape. This suggests that technical "
     "patterns may act as useful event filters, while the predictive signal comes from "
     "the surrounding market dynamics.")
sp(1)
prob("Removed features: raw ATR (scales with price level), absolute SMAs (act as time proxies), "
     "cumulative OBV (trends monotonically), raw MACD (price-dependent). Including any of these "
     "would allow the model to identify which time period a sample comes from, inflating test metrics.")
take("Every feature at bar i uses only data up to bar i. Normalised values prevent temporal leakage.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════
# SLIDE 6, Labeling + Failed Example
# ═══════════════════════════════════════════════════════
sl("Step 3: Triple-Barrier Labels + Failed Predictions")
img("triple_barrier.png", w=CW*0.5, h=5.5*cm,
    c="Triple-barrier labeling method (Lopez de Prado, 2018).")
body("After each event signal, three barriers compete: the take-profit (entry + pt_mult\u00d7ATR), "
     "the stop-loss (entry \u2212 sl_mult\u00d7ATR), and the time limit (max_holding bars). "
     "Whichever is hit first determines the label. This is fundamentally different from naive "
     "up/down labeling, it encodes realistic trade outcomes with built-in risk management.")
sp(1)
img("label_dist.png", w=CW*0.5, h=4*cm,
    c="Label distribution with the best-F1 configuration (pt=2.0, sl=1.5, mh=10).")
body("The classes are roughly balanced, though \"long\" slightly dominates, reflecting "
     "SPY's historical upward bias. Changing the barrier parameters shifts this distribution "
     "substantially: wider stops produce more directional labels, tighter stops produce more no_trade.")
sp(1)
img("failed_prediction.png", w=CW, h=5*cm,
    c="A real failed prediction from the test set.")
body("Not every signal works. Here the model predicted incorrectly and the trade hit the "
     "stop-loss. Showing failures is essential for scientific honesty. These misclassifications "
     "are precisely why we need profitability analysis, classification metrics alone "
     "cannot tell us whether the winning trades compensate for losses like this one.")
sp(1)
take("pt_mult, sl_mult, max_holding are hyperparameters, not constants. Changing them redefines the task.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════
# SLIDE 7, Validation
# ═══════════════════════════════════════════════════════
sl("Step 4: Honest Validation", "Walk-forward CV reveals what k-fold hides")
img("walkforward_diagram.png", w=CW*0.85, h=4.5*cm,
    c="Walk-forward CV: training always precedes testing in time.")
body("In standard k-fold CV, data from 2023 can appear in the training set while data from 2018 "
     "is in the test set, the model literally sees the future. Walk-forward CV prevents this "
     "by using an expanding window: fold k trains on everything before fold k and tests on the next "
     "unseen period. This simulates how the model would perform if deployed and periodically retrained.")
sp(1)
img("wf_timeline.png", w=CW, h=4.5*cm,
    c="Test folds overlaid on SPY price, showing which market regime each fold covers.")
body("The coloured bands show which price history each test fold spans. Each fold covers a "
     "different market environment: a correction, a recovery, a bear market. Performance naturally "
     "varies across these regimes, this variation is real, not an artefact. K-fold would hide "
     "it by blending events from all periods together.")
sp(1)
tbl([
    ["Method", "Temporal?", "F1 variability", "Purpose"],
    ["Walk-forward (4 folds)", "Yes", "0.282 \u00b1 0.008", "Honest deployment simulation"],
    ["K-fold (5 folds)", "No", "Lower variance", "Sampling noise reduction (diagnostic only)"],
], widths=[3.5*cm, 1.5*cm, 3*cm, 6*cm])
sp(1)
take("Walk-forward is harder to pass but honest. It is the gold standard for financial ML evaluation.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════
# SLIDE 8, Central Finding
# ═══════════════════════════════════════════════════════
sl("Central Finding: F1 \u2260 Profitability")
img("f1_vs_return_scatter.png", w=CW*0.85, h=6.5*cm,
    c="Each dot = one (pt, sl) configuration. Colour = stop-loss width (green = wide, red = tight).")
body("This scatter plot is the most direct visualisation of the central finding. "
     "Points on the right have high F1 but are not necessarily at the top (high return). "
     "Green dots (wide stops, sl=2.5\u20133.0) cluster at higher returns despite mid-range F1. "
     "Red dots (tight stops) achieve the highest F1 scores but produce lower cumulative returns. "
     "The two objectives pull in different directions.")
sp(1)
img("tight_vs_wide.png", w=CW, h=5*cm,
    c="Side-by-side equity curves: tight stops (left) vs wide stops (right).")
body("With tight stops, the model makes more trades and classifies labels more accurately, "
     "but individual gains are small because normal volatility frequently triggers the stop-loss "
     "even when the directional prediction is correct. With wide stops, the model makes fewer "
     "trades, misclassifies more events, but winning trades have room to develop into larger "
     "gains. The asymmetric reward structure favours wide stops for profitability.")
sp(1)
take("Best F1=0.569 at pt=2.0/sl=1.5. Best return=25.9% at pt=2.5/sl=3.0. "
     "Classification and profitability optimise at different parameter settings.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════
# SLIDE 9, Optimization Landscape
# ═══════════════════════════════════════════════════════
sl("Hyperparameter Landscape", "100 configurations, divergent optima")
img("heatmap_annotated.png", w=CW, h=6.5*cm,
    c="Optimisation landscape: F1 (left) and cumulative return (right) across 25 (pt, sl) combinations.")
body("The left heatmap shows F1 performance; the right shows cumulative return. The brightest "
     "cells in each map occupy different positions. High F1 concentrates at moderate pt with "
     "tight sl (lower rows). High return requires wider sl (upper rows), where the model misclassifies "
     "more events but the profitable trades generate larger gains per trade.")
sp(1)
tbl([
    ["Config", "pt / sl / mh", "F1", "Return", "Win Rate", "Trades"],
    ["Default", "2.0 / 2.0 / 10", "0.160", "3.7%", "50%", "18"],
    ["Best F1", "2.0 / 1.5 / 10", "0.569", "8.5%", "55%", "18"],
    ["Best Profit", "2.5 / 3.0 / 20", "0.392", "25.9%", "52%", "15"],
], widths=[2.5*cm, 3*cm, 1.5*cm, 2*cm, 2*cm, 1.5*cm])
sp(1)
body("The default configuration achieves F1 of just 0.16, barely above random. Optimisation "
     "lifts this to 0.569, a 3.6\u00d7 improvement. But the most profitable configuration (F1=0.392) "
     "uses entirely different barrier parameters. These parameters do not merely tune the model, "
     "they <i>redefine the classification task itself</i>. A \"long\" label with tight stops is "
     "a fundamentally different outcome than a \"long\" with wide stops.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════
# SLIDE 10, Profitability Deep Dive
# ═══════════════════════════════════════════════════════
sl("Profitability Analysis", "Equity curves, drawdowns, trade fragility")
img("equity_drawdown.png", w=CW, h=6*cm,
    c="Equity curve (top) and drawdown (bottom) on the held-out test set.")
body("Green dots mark winning trades; red dots mark losers. The strategy accumulates gains "
     "through a handful of larger winners, but the drawdown panel shows periods where those "
     "gains are partially given back. A few dominant trades drive the overall return, "
     "removing the top 2\u20133 winners would dramatically change the picture. This concentration "
     "risk is typical of small-sample trading strategies.")
sp(1)
img("confusion_matrix_large.png", w=CW*0.48, h=5*cm,
    c="RF confusion matrix on the test set (best-F1 parameters).")
body("The model over-predicts \"long\", consistent with SPY's historical upward bias over "
     "2010\u20132025. Short predictions are both rare and frequently incorrect. Learning to predict "
     "shorts is harder because bearish patterns on a long-term bullish asset are intrinsically "
     "less common and more noisy. The no_trade class is also challenging, as the model tends "
     "to take a directional position rather than abstain.")
sp(1)
img("wf_fold_equities.png", w=CW, h=6*cm,
    c="Per-fold equity curves from walk-forward CV.")
body("Each subplot shows the equity curve for one walk-forward fold. The variation is substantial: "
     "some folds produce positive returns, others barely break even. With only ~17 test events "
     "per fold, a single large trade can swing the entire fold outcome. This instability is not "
     "a model flaw, it is the unavoidable consequence of working with a small event dataset.")
sp(1)
take("Profitability is fragile and regime-dependent. The edge is real but thin.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════
# SLIDE 11, Generalization + F-Beta
# ═══════════════════════════════════════════════════════
sl("Generalization and F-Beta", "How uncertain are these results?")
img("wf_variability.png", w=CW, h=4.5*cm,
    c="Per-fold metrics from walk-forward CV (mean shown as dashed line, \u00b11 std as grey band).")
body("F1 is the most stable metric (std=0.008), but cumulative return swings from near-zero to "
     "7% depending on the fold. Sharpe ratio varies between 0 and 0.3. Win rate hovers around "
     "50% with substantial uncertainty (\u00b19%). These wide confidence bands mean that the "
     "reported single-split results (F1=0.569, return=25.9%) represent favourable realisations "
     "from a broad distribution, not stable point estimates.")
sp(1)
img("fbeta_comparison.png", w=CW*0.55, h=4.5*cm,
    c="F-beta scores averaged over walk-forward folds.")
body("F0.5 (precision-heavy) = 0.285, F1 (balanced) = 0.282, F2 (recall-heavy) = 0.299. "
     "Precision (0.30) and recall (0.32) are nearly balanced. The model is not strongly biased "
     "toward either avoiding false alarms or capturing all opportunities. In trading terms, "
     "a precision-focused strategy (F0.5) would take fewer but higher-confidence trades; "
     "a recall-focused strategy (F2) would trade more aggressively, accepting more losses "
     "in exchange for not missing profitable setups.")
sp(1)
tbl([
    ["Error Type", "Trading Meaning", "Cost"],
    ["False Positive", "Enter trade \u2192 lose money", "Direct financial loss"],
    ["False Negative", "Skip trade \u2192 miss profit", "Opportunity cost (no loss)"],
], widths=[3*cm, 5.5*cm, 4.5*cm])
sp(1)
take("Walk-forward variance is the most important diagnostic in this project. "
     "It reveals the uncertainty that single-split evaluation hides.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════
# SLIDE 12, What We Learned
# ═══════════════════════════════════════════════════════
sl("What We Learned")
sp(2)
body("<b>1. Event filtering works.</b> Reducing 4,023 bars to 132 events concentrates "
     "the model on moments with structure. RF achieves 3.6\u00d7 F1 improvement over the "
     "stratified random baseline (0.569 vs 0.160). The 97% of bars discarded as noise "
     "would have diluted this signal.")
sp(2)
body("<b>2. Classification \u2260 profitability.</b> The best F1 configuration uses tight "
     "stops (sl=1.5\u00d7ATR, 10-bar holding), producing accurate but small trades. The "
     "best profit configuration uses wide stops (sl=3.0\u00d7ATR, 20-bar holding), "
     "allowing winners to develop despite lower classification accuracy. This divergence "
     "is confirmed across all walk-forward folds and the full optimisation heatmap.")
sp(2)
body("<b>3. Honest validation is essential.</b> Walk-forward CV reveals F1 variance of "
     "\u00b10.008 and return variance of \u00b13.8%, showing substantial regime dependence. "
     "K-fold CV hides this instability by mixing time periods. Financial ML research "
     "that relies solely on k-fold risks dangerous overconfidence.")
sp(2)
body("<b>4. The signal is real but fragile.</b> Technical patterns contain measurable "
     "predictive information, but the edge is thin. With ~17 test events per "
     "walk-forward fold, a single trade can swing the entire fold outcome. The "
     "framework is validated; the dataset is too small for production conclusions.")
sp(3)
tbl([
    ["Limitations", "Future Work"],
    ["Small dataset (~100 events)", "Multi-asset testing (ETFs, stocks)"],
    ["Single asset (SPY only)", "Transaction cost modeling"],
    ["No transaction costs", "Purged/embargo cross-validation"],
    ["Optimisation overfitting risk", "Regime-aware dynamic TP/SL"],
], widths=[7*cm, 7*cm])
sp(5)
story.append(Paragraph("Thank you. Questions?", styles["ST"]))

doc.build(story)
print(f"Presentation saved: {OUT}")
