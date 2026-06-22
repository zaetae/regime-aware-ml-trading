"""Generate 10-minute research defense presentation (vertical A4, 12 slides).

Narrative structure: problem → limitation → solution → finding → next problem.
Every slide answers: What was the challenge? What did we discover?

Produces: reports/final/Zeineb_Turki_bem3.pdf (overwrites previous)
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
PW, PH = A4; M = 1.6*cm; CW = PW - 2*M

BLUE  = HexColor("#1B3A5C"); ACC = HexColor("#2E86C1"); GREY = HexColor("#808B96")
LIGHT = HexColor("#EBF5FB"); DARK = HexColor("#1C2833")
GREEN = HexColor("#27AE60"); RED  = HexColor("#E74C3C"); ORANGE = HexColor("#E67E22")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("STitle", fontSize=22, leading=28, textColor=BLUE,
                          alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("SSub", fontSize=13, leading=17, textColor=ACC,
                          alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle("SInfo", fontSize=11, leading=15, alignment=TA_CENTER, spaceAfter=3))
styles.add(ParagraphStyle("SH", fontSize=17, leading=22, textColor=BLUE, spaceAfter=2,
                          fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("SH2", fontSize=11, leading=14, textColor=ACC, spaceAfter=4,
                          fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("SB", fontSize=10.5, leading=14.5, textColor=DARK, spaceAfter=3))
styles.add(ParagraphStyle("SBul", fontSize=10.5, leading=14, textColor=DARK,
                          leftIndent=14, bulletIndent=6, spaceAfter=2))
# "Key Insight" green box
styles.add(ParagraphStyle("STake", fontSize=10.5, leading=14, textColor=GREEN,
                          fontName="Helvetica-BoldOblique", spaceAfter=5,
                          borderColor=GREEN, borderWidth=0.5, borderPadding=5,
                          backColor=HexColor("#E8F8F5")))
# "Problem" red box
styles.add(ParagraphStyle("SProb", fontSize=10.5, leading=14, textColor=RED,
                          fontName="Helvetica-BoldOblique", spaceAfter=5,
                          borderColor=RED, borderWidth=0.5, borderPadding=5,
                          backColor=HexColor("#FDEDEC")))
styles.add(ParagraphStyle("SCap", fontSize=8, leading=10, textColor=GREY, alignment=TA_CENTER,
                          spaceAfter=4))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=M, rightMargin=M,
                        topMargin=M, bottomMargin=M)
story = []

def slide(t, s=None):
    story.append(Paragraph(t, styles["SH"]))
    if s: story.append(Paragraph(s, styles["SH2"]))
    story.append(Spacer(1, 1*mm))
def bul(t):  story.append(Paragraph(f"\u2022  {t}", styles["SBul"]))
def body(t): story.append(Paragraph(t, styles["SB"]))
def take(t): story.append(Paragraph(f"\u2794  {t}", styles["STake"]))
def prob(t): story.append(Paragraph(f"\u26a0  {t}", styles["SProb"]))
def cap(t):  story.append(Paragraph(t, styles["SCap"]))
def sp(h=3): story.append(Spacer(1, h*mm))
def img(name, w=None, h=None, c=None):
    p = os.path.join(FIG, name)
    if os.path.exists(p):
        story.append(Image(p, width=w or CW, height=h))
        if c: cap(c)

def tbl(data, widths=None):
    s = TableStyle([
        ("BACKGROUND",(0,0),(-1,0),BLUE),("TEXTCOLOR",(0,0),(-1,0),white),
        ("FONTSIZE",(0,0),(-1,0),9),("FONTSIZE",(0,1),(-1,-1),9),
        ("GRID",(0,0),(-1,-1),0.4,GREY),("ALIGN",(1,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[white,LIGHT]),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ])
    t = Table(data, colWidths=widths); t.setStyle(s); story.append(t)

# ════════════════════════════════════════════════════
# SLIDE 1 — Title
# ════════════════════════════════════════════════════
story.append(Spacer(1, 4.5*cm))
story.append(Paragraph("Regime-Aware Machine Learning<br/>for Technical Pattern Trading", styles["STitle"]))
sp(8)
story.append(Paragraph("An Investigation into Event-Based Classification<br/>"
                        "and Profitability Evaluation on SPY Daily Data", styles["SSub"]))
story.append(Spacer(1, 1.5*cm))
story.append(Paragraph("Zeineb Turki", styles["SInfo"]))
sp(2)
story.append(Paragraph("Supervisor: Prof. Dr. Hadh\u00e1zi D\u00e1niel", styles["SInfo"]))
story.append(Paragraph("\u00d3buda University \u2014 BME Independent Laboratory", styles["SInfo"]))
story.append(Paragraph("Summer Semester 2026", styles["SInfo"]))
story.append(Spacer(1, 2*cm))
# Research questions preview
body("<b>Research questions:</b>")
bul("Can pattern-filtered ML outperform a random baseline on trade-outcome prediction?")
bul("Do classification-optimal parameters also maximise trading profit?")
bul("How stable are results across different time periods?")
story.append(PageBreak())

# ════════════════════════════════════════════════════
# SLIDE 2 — The Problem: Markets Are Noisy
# ════════════════════════════════════════════════════
slide("The Core Problem", "Financial markets are hostile to machine learning")
img("noise_regime.png", w=CW, h=9.5*cm,
    c="Figure 1: SPY during COVID \u2014 a 34% crash in 23 trading days, followed by a V-shaped recovery. "
      "Daily returns swing from \u221212% to +9%. Most bars are pure noise.")
sp(2)
prob("Most ML models trained on daily bars learn price trends or temporal artefacts, "
     "not genuine trading edges. Performance collapses on unseen regimes.")
sp(2)
img("volatility_regimes.png", w=CW, h=5.5*cm,
    c="Figure 2: Annualised volatility shifts from 5% (2017) to 90% (COVID). "
      "A model trained in one regime may be useless in another.")
sp(2)
take("Solution: instead of predicting every bar, detect technically meaningful "
     "moments and predict only at those events.")
story.append(PageBreak())

# ════════════════════════════════════════════════════
# SLIDE 3 — The Approach: Problem→Solution Chain
# ════════════════════════════════════════════════════
slide("Research Approach", "Each solution creates the next problem")
img("problem_solution_chain.png", w=CW*0.85, h=14*cm,
    c="Figure 3: The logical chain that shaped the pipeline. "
      "Every design decision responds to a concrete limitation of the previous step.")
story.append(PageBreak())

# ════════════════════════════════════════════════════
# SLIDE 4 — Event Detection + Case Study
# ════════════════════════════════════════════════════
slide("Step 1: Event Detection", "Filtering 4,023 bars down to 132 meaningful moments")
img("spy_events.png", w=CW, h=5.5*cm,
    c="Figure 4: SPY close price with 104 detected events (red). Only 3.3% of bars are flagged.")
sp(2)
img("case_sr.png", w=CW, h=7*cm,
    c="Figure 5: A real support/resistance event. Red triangle = signal bar. "
      "Green/red dashed = TP/SL barriers. The S/R level (green line) acts as a price floor.")
sp(2)
take("Four algorithmic detectors (S/R, channels, triangles, reversals) reduce noise "
     "by >96%. The model only predicts at moments where price structure suggests an opportunity.")
story.append(PageBreak())

# ════════════════════════════════════════════════════
# SLIDE 5 — Labeling + Features
# ════════════════════════════════════════════════════
slide("Step 2: Labels and Features", "Triple-barrier labeling + 48 leakage-free features")
img("triple_barrier.png", w=CW*0.55, h=6*cm,
    c="Figure 6: Triple-barrier method \u2014 label = first barrier hit (TP/SL/time).")
sp(2)
img("feature_importance.png", w=CW, h=6.5*cm,
    c="Figure 7: Top 15 features. MA distances and momentum dominate \u2014 the model "
      "relies on trend-position and short-term dynamics, not absolute price levels.")
sp(2)
prob("Raw features like ATR, SMA, OBV would let the model distinguish early vs late "
     "data by magnitude alone \u2014 inflating test results. All removed and replaced with "
     "normalised alternatives.")
story.append(PageBreak())

# ════════════════════════════════════════════════════
# SLIDE 6 — Validation Strategy
# ════════════════════════════════════════════════════
slide("Step 3: Honest Validation", "Walk-forward CV prevents temporal leakage")
img("walkforward_diagram.png", w=CW*0.85, h=5*cm,
    c="Figure 8: Walk-forward CV \u2014 training always precedes testing. "
      "K-fold (diagnostic only) violates time order.")
sp(3)
body("Standard k-fold shuffles data across time, allowing the model to see future "
     "patterns during training. In finance, this produces dangerously optimistic results.")
sp(2)
body("Walk-forward CV simulates real deployment: train on the past, test on the next "
     "unseen period. It is harder to pass \u2014 but honest.")
sp(3)
take("Walk-forward CV is the gold standard for financial ML. "
     "Our 4-fold results show substantially more variance than k-fold \u2014 "
     "confirming that temporal structure matters.")
story.append(PageBreak())

# ════════════════════════════════════════════════════
# SLIDE 7 — The Core Finding
# ════════════════════════════════════════════════════
slide("Central Finding", "Classification accuracy \u2260 trading profitability")
img("results_summary.png", w=CW, h=6*cm,
    c="Figure 9: Best F1 config (pt=2.0/sl=1.5) differs from best profit config (pt=2.5/sl=3.0).")
sp(2)
body("The best-F1 configuration uses tight stops (sl=1.5\u00d7ATR) and short holding (10 bars). "
     "This classifies labels accurately because tight barriers are easier to predict.")
sp(1)
body("The best-profit configuration uses wide stops (sl=3.0\u00d7ATR) and long holding (20 bars). "
     "This <i>misclassifies more events</i> but allows winning trades room to develop larger gains. "
     "The asymmetric reward outweighs the higher error rate.")
sp(3)
take("A model optimised for accuracy takes small, frequent bets. "
     "A model optimised for profit takes fewer, larger bets. "
     "These are fundamentally different strategies.")
story.append(PageBreak())

# ════════════════════════════════════════════════════
# SLIDE 8 — Optimization Landscape
# ════════════════════════════════════════════════════
slide("Hyperparameter Landscape", "100 configurations reveal divergent optima")
img("heatmap_annotated.png", w=CW, h=7.5*cm,
    c="Figure 10: F1 and cumulative return across 25 (pt, sl) combinations. "
      "The brightest F1 cells do NOT overlap with the highest return cells.")
sp(2)
body("The heatmaps make the divergence visually obvious. High F1 concentrates "
     "around moderate pt and tight sl. High returns require wider sl values, "
     "which the F1 metric penalises because they increase label noise.")
sp(2)
take("Barrier parameters redefine the classification task itself. "
     "Optimising TP/SL is not hyperparameter tuning \u2014 it is task design.")
story.append(PageBreak())

# ════════════════════════════════════════════════════
# SLIDE 9 — Profitability Deep Dive
# ════════════════════════════════════════════════════
slide("Profitability Analysis", "Equity curve, drawdown, and trade-level behaviour")
img("equity_drawdown.png", w=CW, h=7.5*cm,
    c="Figure 11: Equity curve (top) and drawdown (bottom) on the test set. "
      "Green/red dots = individual winning/losing trades.")
sp(2)
img("confusion_matrix_large.png", w=CW*0.5, h=5.5*cm,
    c="Figure 12: Confusion matrix. The model struggles most with no_trade "
      "and short predictions \u2014 reflecting the long-biased nature of SPY.")
sp(2)
take("Profitability is fragile: a few large trades dominate the equity curve. "
     "The model's edge is thin and regime-dependent.")
story.append(PageBreak())

# ════════════════════════════════════════════════════
# SLIDE 10 — Generalization Reality Check
# ════════════════════════════════════════════════════
slide("Generalization Reality Check", "How stable are these results across time?")
img("wf_variability.png", w=CW, h=5.5*cm,
    c="Figure 13: Walk-forward folds. F1 is stable (~0.28), but cumulative return "
      "and Sharpe swing dramatically. One bad fold erases gains.")
sp(2)
img("fbeta_comparison.png", w=CW*0.6, h=5.5*cm,
    c="Figure 14: F-beta scores. Precision barely exceeds recall (0.30 vs 0.32). "
      "The model's signal is weak but consistent across beta weightings.")
sp(2)
take("The walk-forward variance is the most important result in this project. "
     "It means the point estimates (F1=0.569, return=25.9%) come from a broad, "
     "uncertain distribution. Honest evaluation requires reporting this uncertainty.")
story.append(PageBreak())

# ════════════════════════════════════════════════════
# SLIDE 11 — Limitations
# ════════════════════════════════════════════════════
slide("What We Cannot Claim", "Limitations and honest assessment")
bul("<b>~100 events</b> is not enough for statistically robust conclusions")
bul("<b>Single asset (SPY)</b> \u2014 the most efficient market, hardest to predict")
bul("<b>No transaction costs</b> \u2014 real returns would be lower")
bul("<b>Optimisation overfitting</b> \u2014 100 configs on 100 events risks false discovery")
bul("<b>Regime blindness</b> \u2014 the model does not know which regime it is in")
bul("<b>Detector imperfections</b> \u2014 triangles/channels excluded after supervisor review")
sp(4)
body("The walk-forward CV confirms this honestly: across 4 folds, cumulative return "
     "averages 3.3% with a standard deviation of 3.8%. The \"25.9% return\" is a "
     "validation-set peak, not a stable out-of-sample estimate.")
sp(3)
take("The contribution is the framework and the finding about classification vs profitability, "
     "not a deployable trading strategy.")
story.append(PageBreak())

# ════════════════════════════════════════════════════
# SLIDE 12 — Conclusion
# ════════════════════════════════════════════════════
slide("What We Learned")
sp(2)
body("<b>1. Event filtering works.</b>  Reducing 4,023 bars to 132 events concentrates the "
     "model on moments with measurable structure. The RF achieves 3.6\u00d7 F1 improvement "
     "over a random baseline on these events.")
sp(2)
body("<b>2. Classification \u2260 profitability.</b>  The optimal barrier parameters for F1 "
     "(tight stops) differ from those for return (wide stops). This is the central finding "
     "and it is robust across walk-forward folds.")
sp(2)
body("<b>3. Honest validation matters.</b>  Walk-forward CV reveals instability that k-fold "
     "hides. Financial ML research that skips temporal validation risks dangerous overconfidence.")
sp(2)
body("<b>4. The signal is real but weak.</b>  Technical patterns contain measurable information, "
     "but the edge is thin, regime-dependent, and statistically fragile on small datasets.")
sp(5)
body("<b>Future work:</b> more assets, transaction costs, purged CV, regime-aware TP/SL, "
     "gradient boosting, meta-labeling.")
sp(8)
story.append(Paragraph("Thank you \u2014 Questions?", styles["STitle"]))

doc.build(story)
print(f"Presentation saved: {OUT}")
