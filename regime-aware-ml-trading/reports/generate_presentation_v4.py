"""Generate 10-minute scientific presentation (vertical A4, 12 slides).

Produces: reports/final/Zeineb_Turki_bem3.pdf (overwrites v3)
Visually rich, figure-heavy, minimal text per slide.
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
FIG = os.path.join(BASE, "final", "figures")
OUT = os.path.join(BASE, "final", "Zeineb_Turki_bem3.pdf")

PW, PH = A4
M = 1.6 * cm
CW = PW - 2 * M

BLUE  = HexColor("#1B3A5C")
ACC   = HexColor("#2E86C1")
GREY  = HexColor("#808B96")
LIGHT = HexColor("#EBF5FB")
DARK  = HexColor("#1C2833")
GREEN = HexColor("#27AE60")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("STitle", fontSize=22, leading=28, textColor=BLUE,
                          alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("SSub", fontSize=13, leading=17, textColor=ACC,
                          alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle("SInfo", fontSize=11, leading=15, alignment=TA_CENTER, spaceAfter=3))
styles.add(ParagraphStyle("SH", fontSize=17, leading=22, textColor=BLUE, spaceAfter=3,
                          fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("SH2", fontSize=11, leading=15, textColor=ACC, spaceAfter=5,
                          fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("SB", fontSize=10.5, leading=14.5, textColor=DARK, spaceAfter=3))
styles.add(ParagraphStyle("SBul", fontSize=10.5, leading=14.5, textColor=DARK,
                          leftIndent=14, bulletIndent=6, spaceAfter=2))
styles.add(ParagraphStyle("STake", fontSize=10.5, leading=14, textColor=GREEN,
                          fontName="Helvetica-BoldOblique", spaceAfter=5,
                          borderColor=GREEN, borderWidth=0.5, borderPadding=4,
                          backColor=HexColor("#E8F8F5")))
styles.add(ParagraphStyle("SCap", fontSize=8, leading=10, textColor=GREY, alignment=TA_CENTER,
                          spaceAfter=4))
styles.add(ParagraphStyle("SSmall", fontSize=9, leading=12, textColor=GREY))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=M, rightMargin=M,
                        topMargin=M, bottomMargin=M)
story = []

def slide(title, sub=None):
    story.append(Paragraph(title, styles["SH"]))
    if sub: story.append(Paragraph(sub, styles["SH2"]))
    story.append(Spacer(1, 1.5 * mm))

def bul(t):  story.append(Paragraph(f"\u2022  {t}", styles["SBul"]))
def body(t): story.append(Paragraph(t, styles["SB"]))
def take(t): story.append(Paragraph(f"\u2794  {t}", styles["STake"]))
def cap(t):  story.append(Paragraph(t, styles["SCap"]))
def sp(h=3): story.append(Spacer(1, h * mm))

def img(name, w=None, h=None, c=None):
    p = os.path.join(FIG, name)
    if os.path.exists(p):
        story.append(Image(p, width=w or CW, height=h))
        if c: cap(c)
    else:
        story.append(Paragraph(f"<i>[{name}]</i>", styles["SSmall"]))

def tbl(data, widths=None):
    s = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BLUE), ("TEXTCOLOR", (0,0), (-1,0), white),
        ("FONTSIZE", (0,0), (-1,0), 9), ("FONTSIZE", (0,1), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.4, GREY),
        ("ALIGN", (1,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, LIGHT]),
        ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ])
    t = Table(data, colWidths=widths); t.setStyle(s); story.append(t)

# ═══════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ═══════════════════════════════════════════════════════════════
story.append(Spacer(1, 5 * cm))
story.append(Paragraph("Regime-Aware Machine Learning<br/>for Technical Pattern Trading", styles["STitle"]))
sp(8)
story.append(Paragraph("Event-Based Classification and Profitability Evaluation on SPY", styles["SSub"]))
story.append(Spacer(1, 2 * cm))
story.append(Paragraph("Zeineb Turki", styles["SInfo"]))
sp(2)
story.append(Paragraph("Supervisor: Prof. Dr. Kozlovszky Mikl\u00f3s", styles["SInfo"]))
story.append(Paragraph("\u00d3buda University \u2014 BME Independent Laboratory", styles["SInfo"]))
story.append(Paragraph("Summer Semester 2026", styles["SInfo"]))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# SLIDE 2 — Problem
# ═══════════════════════════════════════════════════════════════
slide("Problem and Motivation", "Why is financial prediction so difficult?")
bul("<b>Markets are noisy</b> \u2014 most bars contain no repeatable signal")
bul("<b>Non-stationarity</b> \u2014 statistical properties change (regime shifts, crises)")
bul("<b>Naive prediction</b> learns trends and time artefacts, not trading edges")
sp(3)
body("<b>Our approach: event-based learning</b>")
bul("Detect technically meaningful patterns (S/R, channels, triangles, reversal)")
bul("Predict trade outcomes <i>only</i> at those ~100 events out of 4,023 bars")
sp(4)
img("spy_events.png", w=CW, h=7*cm, c="Figure 1: SPY 2010\u20132025 with 104 detected events (red dots).")
sp(2)
take("Focus on the 3% of bars where price structure suggests a trading opportunity.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# SLIDE 3 — Architecture
# ═══════════════════════════════════════════════════════════════
slide("System Architecture")
img("pipeline_vertical.png", w=CW*0.7, h=14*cm, c="Figure 2: End-to-end pipeline \u2014 7 stages from raw data to profitability evaluation.")
sp(2)
take("Modular design: each stage can be improved independently.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# SLIDE 4 — Detectors + Case Study
# ═══════════════════════════════════════════════════════════════
slide("Pattern Detection", "4 complementary detectors + case study")
img("detection_breakdown.png", w=CW*0.75, h=5.5*cm, c="Figure 3: Event counts by detector type.")
sp(2)
img("case_sr.png", w=CW, h=8*cm, c="Figure 4: Case study \u2014 S/R event with triple-barrier overlay.")
sp(2)
take("132 events from 4,023 bars. Each event gets barriers, features, and a label.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# SLIDE 5 — Labeling
# ═══════════════════════════════════════════════════════════════
slide("Triple-Barrier Labeling", "Trade-outcome labels (Lopez de Prado, 2018)")
img("triple_barrier.png", w=CW*0.65, h=7*cm, c="Figure 5: Price walks forward \u2014 first barrier hit determines the label.")
sp(2)
img("label_dist.png", w=CW*0.6, h=5.5*cm, c="Figure 6: Label distribution (best-F1 config).")
sp(2)
take("pt_mult, sl_mult, max_holding are hyperparameters \u2014 changing them redefines the task.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# SLIDE 6 — Features
# ═══════════════════════════════════════════════════════════════
slide("Feature Engineering", "48 features with strict leakage prevention")
img("feature_importance.png", w=CW, h=8*cm, c="Figure 7: Top 15 features by RF importance.")
sp(2)
body("<b>Removed (leaking temporal info):</b> raw ATR, absolute SMAs, cumulative OBV, raw MACD")
sp(1)
take("Every feature at bar i uses only data up to bar i. No future information leaks.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# SLIDE 7 — Validation
# ═══════════════════════════════════════════════════════════════
slide("Validation Strategy", "Walk-forward CV: the gold standard for financial ML")
img("walkforward_diagram.png", w=CW*0.85, h=5.5*cm, c="Figure 8: Walk-forward CV \u2014 training always precedes testing in time.")
sp(3)
tbl([
    ["Method", "Temporal?", "Purpose"],
    ["60/20/20 split", "Yes", "Primary evaluation"],
    ["Walk-forward CV (4 folds)", "Yes", "Realistic deployment simulation"],
    ["K-fold CV (5 folds)", "No", "Reduce sampling noise (diagnostic)"],
], widths=[4.5*cm, 2.5*cm, 7*cm])
sp(3)
take("Standard k-fold leaks future data. Walk-forward CV simulates real trading conditions.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# SLIDE 8 — Results
# ═══════════════════════════════════════════════════════════════
slide("Experimental Results", "Classification optimum \u2260 profitability optimum")
img("results_summary.png", w=CW, h=6.5*cm, c="Figure 9: Best F1 config differs from best profitability config.")
sp(2)
img("heatmap_annotated.png", w=CW, h=7.5*cm, c="Figure 10: Optimization landscape \u2014 F1 and return across 25 (pt, sl) combinations.")
sp(2)
take("Wider stops sacrifice classification accuracy but allow winning trades to develop. "
     "Best F1=0.569 at pt=2.0/sl=1.5; best return=25.9% at pt=2.5/sl=3.0.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# SLIDE 9 — Profitability
# ═══════════════════════════════════════════════════════════════
slide("Profitability Analysis")
img("equity_drawdown.png", w=CW, h=8.5*cm, c="Figure 11: Equity curve and drawdown on test set (best-F1 config).")
sp(2)
img("confusion_matrix_large.png", w=CW*0.55, h=6.5*cm, c="Figure 12: RF confusion matrix (test set).")
sp(2)
take("Profitability depends on trade magnitude, not just prediction frequency.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# SLIDE 10 — Generalization & F-Beta
# ═══════════════════════════════════════════════════════════════
slide("Generalization and F-Beta Analysis")
img("wf_variability.png", w=CW, h=5.5*cm, c="Figure 13: Walk-forward CV \u2014 per-fold variability (mean \u00b1 std band).")
sp(2)
img("fbeta_comparison.png", w=CW*0.65, h=6*cm, c="Figure 14: F0.5 (precision) vs F1 vs F2 (recall).")
sp(2)
take("High variance across folds = small-dataset uncertainty. "
     "F-beta reveals precision\u2013recall tradeoff relevant to trading risk appetite.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# SLIDE 11 — Limitations
# ═══════════════════════════════════════════════════════════════
slide("Limitations", "Honest scientific assessment")
bul("<b>Small dataset:</b> ~100 events \u2192 high variance, unstable optimisation")
bul("<b>Single asset:</b> only SPY \u2014 generalisation unknown")
bul("<b>No transaction costs:</b> spread, slippage, commissions omitted")
bul("<b>Simplified entry:</b> signal-bar Close (slight optimism)")
bul("<b>Optimisation overfitting:</b> 100 configs on ~100 events")
bul("<b>Regime blind:</b> HMM module planned but not integrated")
sp(4)
body("The walk-forward CV variance confirms that reported point estimates "
     "(F1=0.569, return=25.9%) sit within a broad confidence band. "
     "The framework is validated; the dataset is too small for production conclusions.")
sp(3)
take("All results are preliminary. The contribution is the framework and the finding, "
     "not a deployable trading strategy.")
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# SLIDE 12 — Conclusion
# ═══════════════════════════════════════════════════════════════
slide("Conclusion and Future Work")
body("<b>Achieved:</b>")
bul("End-to-end event-based ML pipeline for SPY (data \u2192 backtest)")
bul("48 leakage-free features, 4 pattern detectors, triple-barrier labels")
bul("Barrier parameters as hyperparameters (100 configs, grid search)")
bul("Walk-forward CV + F-beta + profitability evaluation")
sp(4)
body("<b>Core finding:</b>")
body("Classification accuracy and trading profitability optimise at "
     "<b>different parameter settings</b>. This finding is robust across CV folds "
     "and confirms that profitability metrics are essential for trading model evaluation.")
sp(4)
body("<b>Future work:</b>")
bul("Multi-asset, transaction costs, purged CV, regime-aware TP/SL, gradient boosting")
sp(8)
story.append(Paragraph("Thank you \u2014 Questions?", styles["STitle"]))

doc.build(story)
print(f"Presentation saved: {OUT}")
