"""Generate 10-minute scientific presentation (vertical A4, 12 slides).

Produces: reports/final/Zeineb_Turki_bem3.pdf
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings; warnings.filterwarnings("ignore")

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak,
)
from reportlab.lib import colors

BASE = os.path.dirname(__file__)
FIG = os.path.join(BASE, "final", "figures")
TF  = os.path.join(BASE, "thesis_figures")
EF  = os.path.join(BASE, "experiment_figures")
OUT = os.path.join(BASE, "final", "Zeineb_Turki_bem3.pdf")

PW, PH = A4
M = 1.8 * cm
CW = PW - 2 * M

# Colours
BLUE  = HexColor("#1B3A5C")
ACC   = HexColor("#2E86C1")
GREY  = HexColor("#808B96")
LIGHT = HexColor("#EBF5FB")
DARK  = HexColor("#1C2833")
GREEN = HexColor("#27AE60")
RED   = HexColor("#E74C3C")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("STitle", fontSize=22, leading=28, textColor=BLUE,
                          alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("SSub", fontSize=13, leading=17, textColor=ACC,
                          alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle("SInfo", fontSize=11, leading=15, alignment=TA_CENTER, spaceAfter=3))
styles.add(ParagraphStyle("SH", fontSize=17, leading=22, textColor=BLUE, spaceAfter=4,
                          fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("SH2", fontSize=12, leading=16, textColor=ACC, spaceAfter=6,
                          fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("SB", fontSize=11, leading=15, textColor=DARK, spaceAfter=4))
styles.add(ParagraphStyle("SBul", fontSize=11, leading=15, textColor=DARK,
                          leftIndent=14, bulletIndent=6, spaceAfter=3))
styles.add(ParagraphStyle("STake", fontSize=11, leading=15, textColor=GREEN,
                          fontName="Helvetica-BoldOblique", spaceAfter=6,
                          borderColor=GREEN, borderWidth=0.5, borderPadding=4,
                          backColor=HexColor("#E8F8F5")))
styles.add(ParagraphStyle("SCap", fontSize=8, leading=10, textColor=GREY, alignment=TA_CENTER,
                          spaceAfter=6))
styles.add(ParagraphStyle("SSmall", fontSize=9, leading=12, textColor=GREY))
styles.add(ParagraphStyle("SNum", fontSize=8, textColor=GREY, alignment=TA_RIGHT))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=M, rightMargin=M,
                        topMargin=M, bottomMargin=M)
story = []

def slide(title, subtitle=None):
    story.append(Paragraph(title, styles["SH"]))
    if subtitle:
        story.append(Paragraph(subtitle, styles["SH2"]))
    story.append(Spacer(1, 2 * mm))

def bul(text):
    story.append(Paragraph(f"\u2022  {text}", styles["SBul"]))

def body(text):
    story.append(Paragraph(text, styles["SB"]))

def takeaway(text):
    story.append(Paragraph(f"\u2794  {text}", styles["STake"]))

def cap(text):
    story.append(Paragraph(text, styles["SCap"]))

def img(name, w=None, h=None, caption_text=None):
    for d in [FIG, TF, EF]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            story.append(Image(p, width=w or CW*0.9, height=h))
            if caption_text:
                cap(caption_text)
            return
    story.append(Paragraph(f"<i>[Figure {name} not found]</i>", styles["SSmall"]))

def tbl(data, widths=None, hdr=BLUE):
    s = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), hdr),
        ("TEXTCOLOR", (0,0), (-1,0), white),
        ("FONTSIZE", (0,0), (-1,0), 9), ("FONTSIZE", (0,1), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.4, GREY),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, LIGHT]),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ])
    t = Table(data, colWidths=widths)
    t.setStyle(s)
    story.append(t)

# ════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ════════════════════════════════════════════════════════════════════
story.append(Spacer(1, 6 * cm))
story.append(Paragraph("Regime-Aware Machine Learning<br/>for Technical Pattern Trading", styles["STitle"]))
story.append(Spacer(1, 8 * mm))
story.append(Paragraph("Event-Based Classification and Profitability Evaluation on SPY", styles["SSub"]))
story.append(Spacer(1, 2 * cm))
story.append(Paragraph("Zeineb Turki", styles["SInfo"]))
story.append(Spacer(1, 3 * mm))
story.append(Paragraph("Supervisor: Prof. Dr. Kozlovszky Mikl\u00f3s", styles["SInfo"]))
story.append(Paragraph("\u00d3buda University \u2014 BME Independent Laboratory", styles["SInfo"]))
story.append(Paragraph("Summer Semester 2026", styles["SInfo"]))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════
# SLIDE 2 — Problem & Motivation
# ════════════════════════════════════════════════════════════════════
slide("Problem and Motivation", "Why is financial prediction so difficult?")
bul("<b>Markets are noisy:</b> most price bars contain no repeatable signal")
bul("<b>Non-stationarity:</b> statistical properties change over time (regime shifts)")
bul("<b>Naive bar-by-bar prediction</b> learns trends and noise, not trading edges")
story.append(Spacer(1, 4 * mm))
body("<b>Our approach: event-based learning</b>")
bul("Detect <i>technically meaningful moments</i> (pattern signals)")
bul("Predict trade outcomes only at those moments")
bul("Reduce 4,023 bars \u2192 ~100 focused trading candidates")
story.append(Spacer(1, 4 * mm))
takeaway("Focus the model on moments where price structure may provide an edge, "
         "rather than predicting every bar.")
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════
# SLIDE 3 — System Architecture
# ════════════════════════════════════════════════════════════════════
slide("System Architecture")
img("pipeline_vertical.png", w=CW*0.75, h=15*cm, caption_text="Figure 1: End-to-end pipeline from raw data to profitability evaluation.")
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════
# SLIDE 4 — Pattern Detectors
# ════════════════════════════════════════════════════════════════════
slide("Technical Pattern Detection", "Four detectors identify trading opportunities")
tbl([
    ["Detector", "Method", "Events", "Key Idea"],
    ["Support/Resistance", "Rolling extremes + stability", "42", "Price reacts at key levels"],
    ["Channels", "Chunk polyfit + touch validation", "12", "Price oscillates in parallel bands"],
    ["Triangles", "Pivot regression + convergence", "17", "Volatility compression \u2192 breakout"],
    ["Multiple Tops/Bots", "Rolling extreme + slope confirm", "63", "Reversal at repeated levels"],
], widths=[3.2*cm, 4.5*cm, 1.5*cm, 5*cm])
story.append(Spacer(1, 4 * mm))
bul("All detectors use <b>ATR-based thresholds</b> for scale independence")
bul("<b>Cooldown filter</b> (10 bars) prevents signal clustering")
bul("Triangles and channels excluded from training (supervisor feedback)")
story.append(Spacer(1, 3 * mm))
takeaway("132 events from 4,023 bars = 3.3% event density. "
         "The model focuses on the 3% where structure exists.")
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════
# SLIDE 5 — Triple-Barrier Labeling
# ════════════════════════════════════════════════════════════════════
slide("Triple-Barrier Labeling", "Trade-outcome-based labels (Lopez de Prado, 2018)")
img("triple_barrier.png", w=CW*0.7, h=7.5*cm,
    caption_text="Figure 2: Price walks forward; first barrier hit determines the label.")
story.append(Spacer(1, 3 * mm))
bul("<b>Upper barrier</b> (TP): entry + pt_mult \u00d7 ATR \u2192 label = <b>long</b>")
bul("<b>Lower barrier</b> (SL): entry \u2212 sl_mult \u00d7 ATR \u2192 label = <b>short</b>")
bul("<b>Time barrier</b>: max_holding bars with no hit \u2192 label = <b>no_trade</b>")
story.append(Spacer(1, 3 * mm))
takeaway("pt_mult, sl_mult, max_holding are treated as tunable hyperparameters, "
         "not fixed constants.")
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════
# SLIDE 6 — Features & Leakage Prevention
# ════════════════════════════════════════════════════════════════════
slide("Feature Engineering", "48 features with strict leakage prevention")
tbl([
    ["Group", "Count", "Examples"],
    ["Volatility & Returns", "9", "atr_ratio, ret_1/5/10/20, momentum"],
    ["Moving Average Distances", "8", "sma_10_dist \u2026 sma_200_dist, MA spreads"],
    ["Momentum (RSI, MACD)", "4", "rsi_14, macd_norm, signal, histogram"],
    ["Bollinger & Volume", "4", "bb_width, bb_pctb, volume_ratio, vol_std"],
    ["Binary Filters", "8", "BB touches, SMA crosses, RSI extremes"],
    ["Pattern Geometry", "11", "slopes, touches, containment, width, R\u00b2"],
    ["Event Type Dummies", "4", "One-hot pattern type"],
], widths=[4*cm, 1.5*cm, 8.5*cm])
story.append(Spacer(1, 3 * mm))
body("<b>Removed (leaking temporal info):</b> raw ATR, absolute SMAs, cumulative OBV, raw MACD")
story.append(Spacer(1, 2 * mm))
takeaway("Every feature at bar i uses only data up to bar i. "
         "Normalised values prevent the model from learning price level or time.")
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════
# SLIDE 7 — ML Models & Validation
# ════════════════════════════════════════════════════════════════════
slide("ML Models and Validation")
body("<b>Models:</b> Random Forest (200 trees) | Bagging (200 trees) | Stratified Baseline")
story.append(Spacer(1, 3 * mm))
img("walkforward_diagram.png", w=CW*0.85, h=5.5*cm,
    caption_text="Figure 3: Walk-forward CV — training always precedes testing in time.")
story.append(Spacer(1, 3 * mm))
tbl([
    ["Validation", "Temporal?", "Purpose"],
    ["60/20/20 split", "Yes", "Primary train/val/test evaluation"],
    ["Walk-forward CV (5 folds)", "Yes", "Realistic deployment simulation"],
    ["K-fold event CV (5 folds)", "No", "Reduce sampling noise (diagnostic)"],
], widths=[4.5*cm, 2*cm, 7.5*cm])
story.append(Spacer(1, 3 * mm))
takeaway("Walk-forward CV respects time and prevents future data from leaking into training. "
         "It is the gold standard for financial ML evaluation.")
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════
# SLIDE 8 — Profitability Evaluation
# ════════════════════════════════════════════════════════════════════
slide("Profitability Evaluation", "Why accuracy alone is insufficient")
bul("A model with 60% accuracy can <b>lose money</b> if correct predictions are small "
    "and errors are large")
bul("A model with 40% accuracy can <b>be profitable</b> if winners are much larger than losers")
story.append(Spacer(1, 3 * mm))
tbl([
    ["Metric", "What it measures"],
    ["Cumulative return", "Total profit/loss across all trades"],
    ["Sharpe ratio", "Risk-adjusted return (mean / std)"],
    ["Win rate", "Fraction of trades that are profitable"],
    ["Profit factor", "Gross profit \u00f7 gross loss"],
    ["Max drawdown", "Worst peak-to-trough decline"],
], widths=[4*cm, 10*cm])
story.append(Spacer(1, 3 * mm))
body("<b>Assumptions:</b> entry at signal-bar Close, no transaction costs, equal position sizing.")
story.append(Spacer(1, 2 * mm))
takeaway("Profitability metrics complement classification metrics. "
         "Both are needed for honest evaluation.")
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════
# SLIDE 9 — Results
# ════════════════════════════════════════════════════════════════════
slide("Experimental Results")
img("results_summary.png", w=CW*0.95, h=6.5*cm,
    caption_text="Figure 4: Best F1 configuration differs from best profitability configuration.")
story.append(Spacer(1, 3 * mm))
tbl([
    ["Config", "pt / sl / mh", "F1", "Return", "Win Rate"],
    ["Default", "2.0 / 2.0 / 10", "0.160", "3.7%", "50%"],
    ["Best F1", "2.0 / 1.5 / 10", "0.569", "8.5%", "55%"],
    ["Best Profit", "2.5 / 3.0 / 20", "0.392", "25.9%", "52%"],
], widths=[3*cm, 3*cm, 2*cm, 2.5*cm, 2.5*cm])
story.append(Spacer(1, 3 * mm))
takeaway("Classification optimum \u2260 profitability optimum. "
         "Wider stops sacrifice F1 but allow larger gains per trade.")
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════
# SLIDE 10 — Generalization & F-Beta
# ════════════════════════════════════════════════════════════════════
slide("Generalization and F-Beta Analysis")
img("wf_variability.png", w=CW*0.95, h=5.5*cm,
    caption_text="Figure 5: Per-fold variability — metrics fluctuate substantially across time periods.")
story.append(Spacer(1, 3 * mm))
img("fbeta_comparison.png", w=CW*0.65, h=5.5*cm,
    caption_text="Figure 6: F-beta scores — precision vs recall tradeoff in trading.")
story.append(Spacer(1, 2 * mm))
takeaway("High fold variance confirms that results are preliminary. "
         "F0.5 favours fewer/cleaner trades; F2.0 captures more opportunities.")
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════
# SLIDE 11 — Limitations
# ════════════════════════════════════════════════════════════════════
slide("Limitations", "Honest assessment")
bul("<b>Small dataset:</b> ~100 events \u2192 high variance, unstable hyperparameters")
bul("<b>Single asset:</b> only SPY tested; generalisation to other markets unknown")
bul("<b>No transaction costs:</b> spread, slippage, commissions not modelled")
bul("<b>Simplified entry:</b> signal-bar Close, not next-bar Open")
bul("<b>Optimisation overfitting risk:</b> 100 configs on ~100 events")
bul("<b>No regime modelling:</b> HMM planned but not integrated")
story.append(Spacer(1, 6 * mm))
takeaway("All results are preliminary research findings. "
         "The framework is validated, but the dataset is too small for production conclusions.")
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════
# SLIDE 12 — Conclusion & Future Work
# ════════════════════════════════════════════════════════════════════
slide("Conclusion and Future Work")
body("<b>What was achieved:</b>")
bul("End-to-end pipeline: data \u2192 patterns \u2192 features \u2192 labels \u2192 ML \u2192 backtest")
bul("48 leakage-free features, 4 pattern detectors, triple-barrier labeling")
bul("Barrier parameters as hyperparameters (100 configs tested)")
bul("Walk-forward CV + F-beta + profitability evaluation")
story.append(Spacer(1, 4 * mm))
body("<b>Key scientific finding:</b>")
bul("Classification accuracy and trading profitability optimise at <i>different</i> parameter settings")
story.append(Spacer(1, 4 * mm))
body("<b>Future work:</b>")
bul("Multi-asset testing, transaction costs, purged/embargo CV")
bul("Regime-aware dynamic TP/SL, HMM integration")
bul("Gradient boosting (XGBoost/LightGBM), meta-labeling")
story.append(Spacer(1, 6 * mm))
story.append(Paragraph("Thank you \u2014 Questions?", styles["STitle"]))

# ════════════════════════════════════════════════════════════════════
doc.build(story)
print(f"Presentation saved: {OUT}")
