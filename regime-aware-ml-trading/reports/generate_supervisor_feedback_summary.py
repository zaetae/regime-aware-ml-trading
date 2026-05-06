"""Generate a short PDF summary of supervisor feedback fixes.

Produces a 3-6 page report documenting each issue raised by the
supervisor, what was changed, and why the change is academically valid.

Usage:
    python reports/generate_supervisor_feedback_summary.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether,
)
from reportlab.lib import colors

# ── Output path ──
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_PDF = REPORTS_DIR / "supervisor_feedback_fixes_summary.pdf"

# ── Styles ──
_ss = getSampleStyleSheet()
STYLE_TITLE = ParagraphStyle("T", parent=_ss["Title"], fontSize=18, leading=24,
                             spaceAfter=12, alignment=TA_CENTER)
STYLE_SUBTITLE = ParagraphStyle("ST", parent=_ss["Normal"], fontSize=11,
                                leading=14, spaceAfter=6, alignment=TA_CENTER,
                                textColor=HexColor("#444444"))
STYLE_H1 = ParagraphStyle("H1", parent=_ss["Heading1"], fontSize=14, leading=18,
                           spaceBefore=14, spaceAfter=8, textColor=HexColor("#1a1a2e"))
STYLE_H2 = ParagraphStyle("H2", parent=_ss["Heading2"], fontSize=12, leading=15,
                           spaceBefore=10, spaceAfter=6, textColor=HexColor("#16213e"))
STYLE_BODY = ParagraphStyle("B", parent=_ss["Normal"], fontSize=10, leading=13,
                            spaceAfter=5, alignment=TA_JUSTIFY)
STYLE_SMALL = ParagraphStyle("S", parent=_ss["Normal"], fontSize=8, leading=10)

PAGE_W, PAGE_H = A4
MARGIN = 2.5 * cm


def P(text, style=STYLE_BODY):
    return Paragraph(text, style)

def H1(text):
    return Paragraph(text, STYLE_H1)

def H2(text):
    return Paragraph(text, STYLE_H2)

def SP(h=0.3 * cm):
    return Spacer(1, h)

def make_table(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#f0f4f8")]),
    ]
    if header:
        cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(cmds))
    return t


# ══════════════════════════════════════════════════════════════════
# BUILD DOCUMENT
# ══════════════════════════════════════════════════════════════════
story = []

# ── Title page ──
story.append(SP(3 * cm))
story.append(P("Supervisor Feedback Fixes — Summary Report", STYLE_TITLE))
story.append(SP(0.5 * cm))
story.append(P("Regime-Aware ML Trading Project", STYLE_SUBTITLE))
story.append(P("Zeineb Turki — May 2026", STYLE_SUBTITLE))
story.append(SP(2 * cm))
story.append(P(
    "This document summarises the changes made in response to supervisor feedback. "
    "Each issue is listed with the corresponding fix, academic justification, and "
    "affected files. No drastic redesign was performed — all changes are targeted "
    "improvements within the existing architecture.",
    STYLE_BODY,
))
story.append(PageBreak())

# ── 1. Motivation ──
story.append(H1("1. Motivation"))
story.append(P(
    "Following the supervisor review of the thesis draft, eleven areas were identified "
    "for targeted improvement. These range from documentation fixes (signal localization, "
    "triple-barrier explanation) to methodological enhancements (feature normalization, "
    "additional validation methods, diagnostic metrics). The goal is to strengthen the "
    "academic rigor of the project without redesigning the existing pipeline."
))
story.append(SP())

# ── 2. Issues and fixes table ──
story.append(H1("2. Summary Table of Issues and Fixes"))
issues = [
    ["#", "Issue", "Fix", "Files Modified"],
    ["1", "Signal localization unclear",
     "Added per-detector documentation of how the signal bar (yellow diamond) "
     "is selected. Diamond always plotted at Close of event bar.",
     "triangles.py, channels.py, support_resistance.py, multiple_tops_bottoms.py, "
     "generate_thesis.py"],
    ["2", "Triple-barrier explanation incomplete",
     "Documented: entry=Close, TP=entry+2.0xATR, SL=entry-2.0xATR, max_holding=10, "
     "first barrier hit determines label, same-bar tiebreak via Close.",
     "generate_thesis.py (Section 8.1)"],
    ["3", "Triangle latent bug (missing atr_i)",
     "Fixed _make_detail() call for desc_triangle_upper_test — added missing "
     "atr_i parameter.",
     "triangles.py (line 161)"],
    ["4", "Trend-leaking features in model input",
     "Removed: atr_14 (raw ATR), event_atr, obv_norm (cumulative OBV). "
     "Normalized: MACD features divided by Close (macd_norm, macd_signal_norm, "
     "macd_hist_norm). Kept atr_ratio as normalized ATR.",
     "indicators.py, build_features.py, generate_thesis.py"],
    ["5", "No simple technical filter features",
     "Added 8 binary features: bb_touch_upper/lower, sma50_cross_above/below_sma200, "
     "close_touch_sma50/200, rsi_oversold/overbought.",
     "indicators.py (binary_technical_filters)"],
    ["6", "No training-set confusion matrices",
     "Added evaluate_model on train set. Results dict now has 'train_results'. "
     "Train-vs-test gap table measures overfitting.",
     "train.py, generate_thesis.py, notebook 10"],
    ["7", "No individual tree diagnostics",
     "Added individual_tree_diagnostics(): per-tree test accuracy, mean/min/max, "
     "ensemble improvement, histogram.",
     "train.py, generate_thesis.py, notebook 10"],
    ["8", "No complementary k-fold CV",
     "Added kfold_event_cv(): 5-fold contiguous disjoint folds, 80/20 "
     "train/val within remaining folds. Reports mean/std accuracy and F1.",
     "train.py, generate_thesis.py, notebook 10"],
    ["9", "Walk-forward CV lacks per-fold confusion matrices",
     "Added per-fold confusion matrices and aggregate CM to walk_forward_cv output.",
     "train.py"],
    ["10", "Notebooks and thesis text reference old feature counts",
     "Updated all feature group tables, feature lists, conclusion text, and "
     "limitations to reflect new feature set and validation methods.",
     "generate_thesis.py, notebooks 09/10"],
    ["11", "No summary PDF of changes",
     "Created this document (supervisor_feedback_fixes_summary.pdf).",
     "generate_supervisor_feedback_summary.py"],
]
story.append(make_table(issues, col_widths=[0.8 * cm, 3.5 * cm, 6 * cm, 5 * cm]))
story.append(SP())
story.append(PageBreak())

# ── 3. Detailed justifications ──
story.append(H1("3. Academic Justifications"))

story.append(H2("3.1 Removing raw ATR and event_atr"))
story.append(P(
    "Raw ATR(14) values scale linearly with SPY's price level: ATR was ~$1 in 2010 "
    "and ~$6 in 2025. Including it as a feature effectively encodes the calendar date, "
    "leaking temporal information that inflates model performance on chronological splits. "
    "The normalized version, <b>atr_ratio = ATR / Close</b>, measures volatility as a "
    "fraction of price and is stationary across different price regimes. event_atr has "
    "the same issue and was removed for the same reason."
))
story.append(SP())

story.append(H2("3.2 Normalizing MACD features"))
story.append(P(
    "The MACD line is defined as EMA(12) - EMA(26). When SPY trades at $100, a MACD "
    "value of 2.0 represents a 2% divergence; when SPY trades at $600, the same 2.0 "
    "represents only 0.33%. Dividing by Close (<b>macd_norm = MACD / Close</b>) makes "
    "values comparable across different price regimes. The same normalization is applied "
    "to the signal line and histogram."
))
story.append(SP())

story.append(H2("3.3 Removing obv_norm"))
story.append(P(
    "On-balance volume (OBV) is a cumulative indicator: it starts at zero and accumulates "
    "signed volume over the entire history. Even after normalizing by recent volume, "
    "the cumulative sum trends upward or downward over time, encoding temporal position. "
    "This makes it a proxy for the calendar date rather than a useful trading feature."
))
story.append(SP())

story.append(H2("3.4 Adding binary technical filter features"))
story.append(P(
    "Eight binary (0/1) features encode simple, interpretable technical conditions: "
    "Bollinger Band touches flag extreme price positions; SMA(50)/SMA(200) crossovers "
    "capture the well-known golden cross/death cross signals; Close proximity to key "
    "SMAs flags potential support/resistance tests; RSI extremes flag overbought/oversold "
    "conditions. These add low-complexity interpretable signals that any practitioner "
    "would recognize, without introducing new detectors or increasing model complexity."
))
story.append(SP())

story.append(H2("3.5 Training-set confusion matrices"))
story.append(P(
    "Evaluating on the training set is standard practice for diagnosing overfitting "
    "(Hastie, Tibshirani & Friedman, 2009). A model that achieves near-perfect training "
    "accuracy but poor test accuracy is memorizing noise. The train-test accuracy gap "
    "quantifies this risk. Including train confusion matrices alongside val/test matrices "
    "provides a complete picture of model behaviour across all data partitions."
))
story.append(SP())

story.append(H2("3.6 Individual tree diagnostics"))
story.append(P(
    "Ensemble methods derive their power from aggregating diverse weak learners. If the "
    "ensemble accuracy is only marginally better than the mean individual tree accuracy, "
    "the voting mechanism adds little value. If a few strong trees dominate while most "
    "are weak, the ensemble is fragile. Reporting per-tree accuracy statistics (mean, "
    "std, min, max) and ensemble improvement quantifies ensemble stability and helps "
    "determine whether the ensemble size (200 trees) is sufficient."
))
story.append(SP())

story.append(H2("3.7 Five-fold event-level CV"))
story.append(P(
    "With only ~137 labeled events and a 60/20/20 split producing ~28 test events, "
    "single-split metrics have high variance. K-fold cross-validation reduces this "
    "variance by averaging over multiple test folds (Kohavi, 1995). Using contiguous, "
    "pairwise disjoint folds preserves some temporal structure while maximizing each "
    "fold's effective test size. This is explicitly presented as a <b>complementary</b> "
    "diagnostic, not a replacement for walk-forward CV which respects strict temporal "
    "ordering."
))
story.append(SP())
story.append(PageBreak())

# ── 4. Before/After summary ──
story.append(H1("4. Before / After Summary"))
ba_data = [
    ["Aspect", "Before", "After"],
    ["Feature count", "~49 (with atr_14, event_atr, obv_norm, raw MACD)",
     "~54 (with atr_ratio, normalized MACD, 8 binary filters, no leakers)"],
    ["MACD features", "Raw: macd, macd_signal, macd_hist",
     "Normalized: macd_norm, macd_signal_norm, macd_hist_norm"],
    ["Signal localization docs", "Implicit in code",
     "Explicit per-detector documentation in docstrings and thesis"],
    ["Triple-barrier explanation", "Brief paragraph",
     "Explicit bullet list: entry, TP, SL, max_holding, label rule"],
    ["Triangle bug", "Missing atr_i in desc_triangle_upper_test",
     "Fixed: atr_i passed correctly"],
    ["Train confusion matrices", "Not included",
     "Train/val/test CMs with train-test gap table"],
    ["Individual tree diagnostics", "Not included",
     "Per-tree accuracy stats + histogram + ensemble improvement"],
    ["K-fold CV", "Not included",
     "5-fold event-level CV with mean/std accuracy and F1"],
    ["Walk-forward per-fold CMs", "Not included",
     "Per-fold and aggregate confusion matrices"],
    ["Validation methods", "Chrono split + walk-forward",
     "Chrono split + walk-forward + 5-fold event CV + train CMs"],
]
story.append(make_table(ba_data, col_widths=[3 * cm, 5.5 * cm, 6.5 * cm]))
story.append(SP())
story.append(PageBreak())

# ── 5. Files modified ──
story.append(H1("5. Files Modified"))
files = [
    ["File", "Changes"],
    ["src/patterns/triangles.py", "Fixed missing atr_i bug; added signal localization docstring"],
    ["src/patterns/channels.py", "Added signal localization docstring"],
    ["src/patterns/support_resistance.py", "Added signal localization docstring"],
    ["src/patterns/multiple_tops_bottoms.py", "Added signal localization docstring"],
    ["src/features/indicators.py",
     "Removed atr_14, obv_norm from compute_all_indicators; "
     "normalized MACD; added binary_technical_filters()"],
    ["src/features/build_features.py", "Removed event_atr from feature matrix"],
    ["src/models/train.py",
     "Added individual_tree_diagnostics(), kfold_event_cv(); "
     "added train-set evaluation; added per-fold CMs to walk-forward"],
    ["reports/generate_thesis.py",
     "Updated feature tables, signal localization section, "
     "triple-barrier explanation, train CMs, tree diagnostics, k-fold CV, "
     "updated conclusion and limitations"],
    ["notebooks/09_feature_engineering.ipynb", "Updated conclusion, feature descriptions"],
    ["notebooks/10_model_training.ipynb",
     "Updated imports, feature description; added cells for train CMs, "
     "tree diagnostics, 5-fold CV"],
    ["reports/generate_supervisor_feedback_summary.py", "New file (this report)"],
]
story.append(make_table(files, col_widths=[5 * cm, 10 * cm]))
story.append(SP())

# ── 6. Remaining limitations ──
story.append(H1("6. Remaining Limitations"))
story.append(P(
    "&bull; <b>Small sample size</b> remains the primary constraint (~137 events). "
    "No amount of cross-validation can overcome limited data.<br/>"
    "&bull; <b>No real Alpha Vantage validation</b> — the comparison still uses "
    "simulated noise, not a real API key.<br/>"
    "&bull; <b>No hyperparameter optimization</b> — max_depth=8 and n_estimators=200 "
    "are fixed defaults.<br/>"
    "&bull; <b>5-fold event-level CV does not respect temporal ordering</b> — it is "
    "a complementary diagnostic, not a replacement for walk-forward CV.<br/>"
    "&bull; <b>No regime conditioning</b> (HMM) or backtesting with transaction costs."
))
story.append(SP())
story.append(P(
    "These limitations are honestly documented in the thesis and do not invalidate "
    "the methodological improvements made in this round of fixes.",
))

# ── Build PDF ──
print(f"Writing {OUTPUT_PDF} ...")
doc = SimpleDocTemplate(
    str(OUTPUT_PDF), pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN,
    title="Supervisor Feedback Fixes — Summary Report",
    author="Zeineb Turki",
)

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(PAGE_W / 2, 1.5 * cm, f"Page {doc.page}")
    canvas.restoreState()

doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print(f"Done! Summary saved to {OUTPUT_PDF}")
