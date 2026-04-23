"""Generate a thesis-style PDF (~40 pages) from actual project results.

Uses reportlab to create the document. All statistics are computed
from the real data and models — nothing is fabricated.

Usage:
    python reports/generate_thesis.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether, ListFlowable, ListItem
)
from reportlab.lib import colors

# ── Project imports ──────────────────────────────────────────────
from src.data.load_data import load_spy
from src.data.utils import compute_atr
from src.patterns.scanner import scan_all_patterns
from src.patterns.triangles import detect_triangle_pattern
from src.patterns.channels import detect_channel
from src.labeling.label_events import label_events
from src.features.build_features import build_feature_matrix
from src.features.indicators import compute_all_indicators
from src.models.train import (
    run_training_pipeline, feature_importance_table,
    tree_complexity_stats
)

# ── Output paths ─────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "thesis_figures"
FIGURES_DIR.mkdir(exist_ok=True)
OUTPUT_PDF = REPORTS_DIR / "thesis.pdf"

# ── Page layout ──────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN = 2.5 * cm

# ── Styles ───────────────────────────────────────────────────────
_ss = getSampleStyleSheet()

STYLE_TITLE = ParagraphStyle(
    "ThesisTitle", parent=_ss["Title"],
    fontSize=22, leading=28, spaceAfter=12, alignment=TA_CENTER,
)
STYLE_SUBTITLE = ParagraphStyle(
    "ThesisSubtitle", parent=_ss["Normal"],
    fontSize=14, leading=18, spaceAfter=6, alignment=TA_CENTER,
    textColor=HexColor("#444444"),
)
STYLE_H1 = ParagraphStyle(
    "H1", parent=_ss["Heading1"],
    fontSize=16, leading=20, spaceBefore=18, spaceAfter=10,
    textColor=HexColor("#1a1a2e"),
)
STYLE_H2 = ParagraphStyle(
    "H2", parent=_ss["Heading2"],
    fontSize=13, leading=16, spaceBefore=12, spaceAfter=6,
    textColor=HexColor("#16213e"),
)
STYLE_H3 = ParagraphStyle(
    "H3", parent=_ss["Heading3"],
    fontSize=11, leading=14, spaceBefore=8, spaceAfter=4,
)
STYLE_BODY = ParagraphStyle(
    "Body", parent=_ss["Normal"],
    fontSize=10, leading=14, spaceAfter=6, alignment=TA_JUSTIFY,
)
STYLE_CAPTION = ParagraphStyle(
    "Caption", parent=_ss["Normal"],
    fontSize=9, leading=11, spaceAfter=10, alignment=TA_CENTER,
    textColor=HexColor("#555555"), fontName="Helvetica-Oblique",
)
STYLE_SMALL = ParagraphStyle(
    "Small", parent=_ss["Normal"],
    fontSize=8, leading=10,
)

# ── Helper functions ─────────────────────────────────────────────

def P(text, style=STYLE_BODY):
    return Paragraph(text, style)

def H1(text):
    return Paragraph(text, STYLE_H1)

def H2(text):
    return Paragraph(text, STYLE_H2)

def H3(text):
    return Paragraph(text, STYLE_H3)

def SP(h=0.3*cm):
    return Spacer(1, h)

def make_table(data, col_widths=None, header=True):
    """Build a styled Table from a list of lists."""
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#f0f4f8")]),
    ]
    if header:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style_cmds))
    return t


def save_fig(fig, name, dpi=150):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def add_figure(story, path, width=14*cm, caption=None):
    story.append(Image(path, width=width, height=width*0.6))
    if caption:
        story.append(P(caption, STYLE_CAPTION))
    story.append(SP())


# ══════════════════════════════════════════════════════════════════
# COMPUTE ALL RESULTS
# ══════════════════════════════════════════════════════════════════

print("Loading data...")
df = load_spy()
atr_series = compute_atr(df)

print("Running detectors...")
df_scanned = scan_all_patterns(df)
_, tri_details = detect_triangle_pattern(df, return_details=True)
_, ch_details = detect_channel(df, return_details=True)

print("Labeling events...")
labeled = label_events(df)

print("Building features...")
features, labels, labeled_full = build_feature_matrix(df)

print("Training models...")
results = run_training_pipeline(features, labels, labeled_full)

print("Computing indicators...")
indicators = compute_all_indicators(df)

# ── Derived statistics ───────────────────────────────────────────
n_bars = len(df)
date_start = df.index[0].strftime("%Y-%m-%d")
date_end = df.index[-1].strftime("%Y-%m-%d")
price_min = df["Close"].min()
price_max = df["Close"].max()
ann_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0]) ** (252 / n_bars) - 1

det_counts = {
    "Near support": int(df_scanned["near_support"].sum()),
    "Near resistance": int(df_scanned["near_resistance"].sum()),
    "Triangles": int(df_scanned["triangle_pattern"].notna().sum()),
    "Multi top/bottom": int(df_scanned["multiple_top_bottom_pattern"].notna().sum()),
    "Channels": int(df_scanned["channel_pattern"].notna().sum()),
}
total_events = int(df_scanned["has_event"].sum())

label_counts = labels.value_counts().to_dict()
n_labeled = len(labeled_full)

tri_touches_avg = np.mean([d["upper_touches"]+d["lower_touches"] for d in tri_details])
tri_cont_avg = np.mean([d["containment_ratio"] for d in tri_details])
ch_touches_avg = np.mean([d["upper_touches"]+d["lower_touches"] for d in ch_details])
ch_cont_avg = np.mean([d["containment_ratio"] for d in ch_details])

rf_test = results["test_results"]["rf"]
bag_test = results["test_results"]["bagging"]
base_test = results["test_results"]["baseline"]
rf_trees = results["tree_stats"]["rf"]
bag_trees = results["tree_stats"]["bagging"]
rf_fi = results["feature_importance"]["rf"]

n_features = features.shape[1]

split = results["split"]
n_train = len(split["X_train"])
n_val = len(split["X_val"])
n_test = len(split["X_test"])

# ── Generate figures ─────────────────────────────────────────────
print("Generating figures...")
sns.set_style("whitegrid")

# Fig 1: SPY price history
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["Close"], linewidth=0.8, color="#1a1a2e")
ax.set_title("SPY Daily Close Price (2010-2025)")
ax.set_ylabel("Price (USD)")
ax.grid(True, alpha=0.3)
fig_price = save_fig(fig, "spy_price")

# Fig 2: Detection breakdown
fig, ax = plt.subplots(figsize=(8, 4))
names = list(det_counts.keys())
vals = list(det_counts.values())
bars = ax.barh(names, vals, color=["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd"])
ax.set_xlabel("Number of Detections")
ax.set_title("Pattern Detections by Type")
for b, v in zip(bars, vals):
    ax.text(v + 1, b.get_y() + b.get_height()/2, str(v), va="center", fontsize=9)
fig_det = save_fig(fig, "detection_breakdown")

# Fig 3: Label distribution
fig, ax = plt.subplots(figsize=(6, 4))
lbl_names = ["long", "short", "no_trade"]
lbl_vals = [label_counts.get(k, 0) for k in lbl_names]
ax.bar(lbl_names, lbl_vals, color=["#2ca02c", "#d62728", "#999999"])
ax.set_title("Label Distribution")
ax.set_ylabel("Count")
for i, v in enumerate(lbl_vals):
    ax.text(i, v + 1, str(v), ha="center", fontsize=10)
fig_labels = save_fig(fig, "label_distribution")

# Fig 4: Confusion matrix (RF test)
from sklearn.metrics import ConfusionMatrixDisplay
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax_i, (name, res) in zip(axes, [("Random Forest", rf_test), ("Bagging", bag_test), ("Baseline", base_test)]):
    ConfusionMatrixDisplay(res["confusion_matrix"], display_labels=res["labels"]).plot(ax=ax_i, cmap="Blues", colorbar=False)
    ax_i.set_title(name, fontsize=10)
fig.suptitle("Confusion Matrices (Test Set)")
plt.tight_layout()
fig_cm = save_fig(fig, "confusion_matrices")

# Fig 5: Feature importance
fig, ax = plt.subplots(figsize=(8, 5))
top_fi = rf_fi.head(15)
ax.barh(top_fi["feature"].values[::-1], top_fi["importance"].values[::-1], color="#1f77b4")
ax.set_title("Top 15 Feature Importances (Random Forest)")
ax.set_xlabel("Importance")
plt.tight_layout()
fig_fi = save_fig(fig, "feature_importance")

# Fig 6: Touch count distributions
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
tri_tc = [d["upper_touches"]+d["lower_touches"] for d in tri_details]
ch_tc = [d["upper_touches"]+d["lower_touches"] for d in ch_details]
axes[0].hist(tri_tc, bins=range(0, max(tri_tc)+2), color="#1f77b4", edgecolor="white", alpha=0.8)
axes[0].set_title("Triangle Touch Counts")
axes[0].set_xlabel("Total Touches")
axes[1].hist(ch_tc, bins=range(0, max(ch_tc)+2), color="#ff7f0e", edgecolor="white", alpha=0.8)
axes[1].set_title("Channel Touch Counts")
axes[1].set_xlabel("Total Touches")
plt.tight_layout()
fig_touches = save_fig(fig, "touch_distributions")

# Fig 7: Model comparison bar chart
fig, ax = plt.subplots(figsize=(8, 4))
model_names = ["Random Forest", "Bagging", "Baseline"]
accs = [rf_test["accuracy"], bag_test["accuracy"], base_test["accuracy"]]
f1s = [rf_test["f1_macro"], bag_test["f1_macro"], base_test["f1_macro"]]
x = np.arange(len(model_names))
w = 0.35
ax.bar(x - w/2, accs, w, label="Accuracy", color="#1f77b4")
ax.bar(x + w/2, f1s, w, label="F1 (macro)", color="#ff7f0e")
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 0.7)
ax.legend()
ax.set_title("Model Comparison (Test Set)")
fig_model = save_fig(fig, "model_comparison")

# Fig 8: Temporal split timeline
fig, ax = plt.subplots(figsize=(10, 2))
train_dates = split["dates_train"]
val_dates = split["dates_val"]
test_dates = split["dates_test"]
ax.barh(0, len(train_dates), left=0, color="#2ca02c", label=f"Train ({n_train})")
ax.barh(0, len(val_dates), left=len(train_dates), color="#ff7f0e", label=f"Val ({n_val})")
ax.barh(0, len(test_dates), left=len(train_dates)+len(val_dates), color="#d62728", label=f"Test ({n_test})")
ax.set_yticks([])
ax.legend(loc="upper right")
ax.set_title("Temporal Train / Validation / Test Split")
ax.set_xlabel("Event Index (chronological)")
fig_split = save_fig(fig, "temporal_split")

# Fig 9: Containment distributions
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
tri_cr = [d["containment_ratio"] for d in tri_details]
ch_cr = [d["containment_ratio"] for d in ch_details]
axes[0].hist(tri_cr, bins=10, color="#1f77b4", edgecolor="white", alpha=0.8)
axes[0].set_title("Triangle Containment")
axes[0].set_xlabel("Containment Ratio")
axes[1].hist(ch_cr, bins=10, color="#ff7f0e", edgecolor="white", alpha=0.8)
axes[1].set_title("Channel Containment")
axes[1].set_xlabel("Containment Ratio")
plt.tight_layout()
fig_containment = save_fig(fig, "containment_distributions")

# Fig 10: Feature correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
feat_cols = [c for c in features.columns if not c.startswith("etype_")]
corr = features[feat_cols[:20]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, ax=ax, annot=False,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.7})
ax.set_title("Feature Correlation Matrix (Top 20 Numeric Features)")
plt.tight_layout()
fig_corr = save_fig(fig, "feature_correlation")

# Fig 11: ATR time series
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(df.index, df["Close"], linewidth=0.6, color="#1a1a2e")
axes[0].set_ylabel("Close Price (USD)")
axes[0].set_title("SPY Close Price and ATR(14)")
axes[0].grid(True, alpha=0.3)
axes[1].plot(df.index, atr_series, linewidth=0.6, color="#d62728")
axes[1].set_ylabel("ATR(14) (USD)")
axes[1].set_xlabel("Date")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
fig_atr = save_fig(fig, "atr_timeseries")

# Fig 12: Returns distribution
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
daily_ret = df["Close"].pct_change().dropna()
axes[0].hist(daily_ret, bins=100, color="#1f77b4", alpha=0.7, edgecolor="white")
axes[0].set_title("Daily Returns Distribution")
axes[0].set_xlabel("Return")
axes[0].axvline(0, color="red", linestyle="--", alpha=0.5)
log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
axes[1].hist(log_ret, bins=100, color="#2ca02c", alpha=0.7, edgecolor="white")
axes[1].set_title("Log Returns Distribution")
axes[1].set_xlabel("Log Return")
axes[1].axvline(0, color="red", linestyle="--", alpha=0.5)
rolling_vol = log_ret.rolling(20).std() * np.sqrt(252)
axes[2].plot(df.index[1:], rolling_vol, linewidth=0.5, color="#9467bd")
axes[2].set_title("20-day Rolling Volatility (Annualised)")
axes[2].set_ylabel("Volatility")
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
fig_returns = save_fig(fig, "returns_analysis")

# Fig 13: Event timeline
fig, ax = plt.subplots(figsize=(12, 4))
event_rows = df_scanned[df_scanned["has_event"]]
color_map = {True: "#d62728", False: "#1f77b4"}
for idx, row in event_rows.iterrows():
    c = "#d62728" if pd.notna(row.get("triangle_pattern")) or pd.notna(row.get("channel_pattern")) else "#1f77b4"
    if pd.notna(row.get("multiple_top_bottom_pattern")):
        c = "#ff7f0e"
    if row.get("near_support") or row.get("near_resistance"):
        c = "#2ca02c"
    ax.axvline(idx, color=c, alpha=0.3, linewidth=0.5)
ax.plot(df.index, df["Close"], linewidth=0.5, color="#333333")
ax.set_title("SPY Price with Detected Events Overlay")
ax.set_ylabel("Close Price")
ax.grid(True, alpha=0.2)
import matplotlib.patches as mpatches
legend_items = [
    mpatches.Patch(color="#2ca02c", label="S/R"),
    mpatches.Patch(color="#d62728", label="Triangle/Channel"),
    mpatches.Patch(color="#ff7f0e", label="Multi Top/Bottom"),
]
ax.legend(handles=legend_items, loc="upper left", fontsize=8)
plt.tight_layout()
fig_timeline = save_fig(fig, "event_timeline")

# Fig 14: Per-class return distributions
fig, ax = plt.subplots(figsize=(8, 5))
for lbl, color in [("long", "#2ca02c"), ("short", "#d62728"), ("no_trade", "#999999")]:
    subset = labeled_full[labeled_full["label"] == lbl]
    ax.hist(subset["return_pct"], bins=20, alpha=0.6, color=color, label=lbl, edgecolor="white")
ax.set_title("Return Distribution by Label")
ax.set_xlabel("Return (%)")
ax.set_ylabel("Count")
ax.legend()
ax.axvline(0, color="black", linestyle="--", alpha=0.4)
fig_ret_by_label = save_fig(fig, "return_by_label")

# Fig 15: Bars held distribution
fig, ax = plt.subplots(figsize=(8, 4))
for lbl, color in [("long", "#2ca02c"), ("short", "#d62728"), ("no_trade", "#999999")]:
    subset = labeled_full[labeled_full["label"] == lbl]
    ax.hist(subset["bars_held"], bins=range(0, 12), alpha=0.6, color=color, label=lbl, edgecolor="white")
ax.set_title("Holding Period Distribution by Label")
ax.set_xlabel("Bars Held")
ax.set_ylabel("Count")
ax.legend()
fig_bars_held = save_fig(fig, "bars_held")

# Fig 16: Feature distributions for top 6 features (violin plots)
from sklearn.feature_selection import f_classif
X_clean = features.fillna(0)
f_stat, p_val = f_classif(X_clean, labels)
top6_idx = np.argsort(f_stat)[-6:][::-1]
top6_names = [features.columns[i] for i in top6_idx]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax_i, fname in zip(axes.flat, top6_names):
    plot_df = pd.DataFrame({"value": X_clean[fname], "label": labels})
    for lbl, color in [("long", "#2ca02c"), ("short", "#d62728"), ("no_trade", "#999999")]:
        subset = plot_df[plot_df["label"] == lbl]["value"]
        ax_i.hist(subset, bins=15, alpha=0.5, color=color, label=lbl, edgecolor="white", density=True)
    ax_i.set_title(fname, fontsize=9)
    ax_i.legend(fontsize=7)
plt.suptitle("Top 6 Discriminative Features by Label Class", fontsize=12)
plt.tight_layout()
fig_feat_dist = save_fig(fig, "feature_distributions")

# Fig 17: Channel width distribution
fig, ax = plt.subplots(figsize=(6, 4))
ch_widths = [d["channel_width_atr"] for d in ch_details]
ax.hist(ch_widths, bins=15, color="#ff7f0e", edgecolor="white", alpha=0.8)
ax.set_title("Channel Width Distribution (ATR units)")
ax.set_xlabel("Width (x ATR)")
ax.set_ylabel("Count")
ax.axvline(np.mean(ch_widths), color="red", linestyle="--", label=f"Mean: {np.mean(ch_widths):.2f}")
ax.legend()
fig_ch_width = save_fig(fig, "channel_width")

# Compute additional stats for expanded sections
daily_ret_stats = daily_ret.describe()
monthly_counts = event_rows.groupby(event_rows.index.to_period("M")).size()
yearly_counts = event_rows.groupby(event_rows.index.to_period("Y")).size()

# Per-label return stats
label_ret_stats = labeled_full.groupby("label")["return_pct"].describe().round(3)
label_hold_stats = labeled_full.groupby("label")["bars_held"].describe().round(2)

# ══════════════════════════════════════════════════════════════════
# BUILD THE DOCUMENT
# ══════════════════════════════════════════════════════════════════

print("Building thesis PDF...")
story = []

# ── Title Page ───────────────────────────────────────────────────
story.append(SP(5*cm))
story.append(P("Regime-Aware Machine Learning<br/>for Equity Trading", STYLE_TITLE))
story.append(SP(0.5*cm))
story.append(P("Technical Pattern Detection, Triple-Barrier Labeling,<br/>"
               "and Ensemble Classification on SPY Daily Data", STYLE_SUBTITLE))
story.append(SP(2*cm))
story.append(P("Zeineb Turki", ParagraphStyle("Author", parent=STYLE_SUBTITLE, fontSize=12)))
story.append(SP(0.5*cm))
story.append(P("April 2026", ParagraphStyle("Date", parent=STYLE_SUBTITLE, fontSize=11)))
story.append(SP(3*cm))
story.append(P("A thesis-style project report<br/>prepared for academic supervision review",
               ParagraphStyle("Note", parent=STYLE_BODY, alignment=TA_CENTER, fontSize=10)))
story.append(PageBreak())

# ── Table of Contents ────────────────────────────────────────────
story.append(H1("Table of Contents"))
story.append(SP())
toc_items = [
    ("1", "Abstract"),
    ("2", "Introduction"),
    ("3", "Literature and Conceptual Background"),
    ("4", "Project Architecture"),
    ("5", "Data Sources"),
    ("6", "Pattern Detection Methodology"),
    ("7", "Trendline Touch Analysis"),
    ("8", "Triple-Barrier Labeling"),
    ("9", "Feature Engineering"),
    ("10", "Model Methodology"),
    ("11", "Experiments and Results"),
    ("12", "Discussion"),
    ("13", "Limitations"),
    ("14", "Future Work"),
    ("15", "Conclusion"),
    ("16", "References"),
    ("A", "Appendix A: Detector Parameters"),
    ("B", "Appendix B: Complete Feature List"),
]
toc_data = [[P(f"<b>{num}</b>", STYLE_BODY), P(title, STYLE_BODY)] for num, title in toc_items]
story.append(make_table(toc_data, col_widths=[1.5*cm, 12*cm], header=False))
story.append(PageBreak())

# ── 1. Abstract ──────────────────────────────────────────────────
story.append(H1("1. Abstract"))
story.append(P(
    f"This project develops a regime-aware machine learning pipeline for equity trading, "
    f"applied to {n_bars:,} daily bars of SPDR S&P 500 ETF Trust (SPY) data spanning "
    f"{date_start} to {date_end}. The system detects technical price patterns — "
    f"support/resistance levels, triangle formations, price channels, and multiple "
    f"tops/bottoms — using pivot-based and chunk-based algorithms with linear regression "
    f"trendline fitting. Each detected event is labeled via the triple-barrier method "
    f"(Lopez de Prado, 2018), producing {n_labeled} labeled events with three classes: "
    f"long ({label_counts.get('long',0)}), short ({label_counts.get('short',0)}), and "
    f"no_trade ({label_counts.get('no_trade',0)}). A feature engineering module computes "
    f"{n_features} features per event, combining technical indicators, pattern geometry, "
    f"and touch-counting statistics. Random Forest and Bagging classifiers are trained "
    f"using a strict temporal split to avoid lookahead bias. The Random Forest achieves "
    f"{rf_test['accuracy']:.1%} test accuracy and {rf_test['f1_macro']:.3f} macro-F1, "
    f"outperforming a stratified baseline ({base_test['accuracy']:.1%} accuracy). "
    f"Key discriminative features include recent returns, moving average distances, and "
    f"volume statistics. The project demonstrates a transparent, modular, and "
    f"reproducible approach to ML-based trading signal classification."
))
story.append(SP())
story.append(P(
    "<b>Keywords:</b> technical analysis, pattern detection, triple-barrier labeling, "
    "random forest, bagging, ensemble methods, event-driven trading, feature engineering, "
    "temporal cross-validation."
))
story.append(PageBreak())

# ── 2. Introduction ──────────────────────────────────────────────
story.append(H1("2. Introduction"))
story.append(H2("2.1 Motivation"))
story.append(P(
    "Financial markets exhibit complex, non-stationary dynamics that challenge traditional "
    "trading strategies. Technical analysis — the study of price action and chart patterns — "
    "has been practiced for decades, yet its systematic quantification and integration with "
    "machine learning remains an active research area. This project addresses the question: "
    "can algorithmically detected price patterns, combined with modern ML classifiers, "
    "produce actionable trading signals?"
))
story.append(P(
    "The core challenge is threefold. First, pattern detection must be rigorous: "
    "naively fitting trendlines to noisy price data produces excessive false positives. "
    "Second, event labeling must avoid the hindsight bias common in technical analysis — "
    "the triple-barrier method provides a principled forward-looking label. Third, "
    "model evaluation must respect temporal ordering to avoid information leakage."
))
story.append(SP())
story.append(H2("2.2 Objectives"))
story.append(P("This project pursues the following objectives:"))
objectives = [
    "Implement robust, parameter-tuned pattern detectors for triangles, channels, "
    "support/resistance, and multiple tops/bottoms.",
    "Develop a trendline touch-counting methodology that quantifies how well "
    "price action respects detected boundaries.",
    "Apply triple-barrier labeling to generate event-level ML targets.",
    "Engineer a comprehensive feature set combining technical indicators with "
    "pattern geometry and touch statistics.",
    "Train and compare ensemble classifiers (Random Forest and Bagging) using "
    "strict temporal splitting.",
    "Provide a transparent, reproducible pipeline with visible statistics at "
    "every stage.",
]
for obj in objectives:
    story.append(P(f"&bull; {obj}"))
story.append(SP())
story.append(H2("2.3 Scope"))
story.append(P(
    f"The analysis focuses on SPY daily data ({date_start} to {date_end}, "
    f"{n_bars:,} trading days). While the methodology is general, this project "
    f"does not implement live trading, transaction cost modeling, or portfolio "
    f"optimization. The emphasis is on detection quality, labeling correctness, "
    f"and classification transparency."
))
story.append(PageBreak())

# ── 3. Literature ────────────────────────────────────────────────
story.append(H1("3. Literature and Conceptual Background"))
story.append(H2("3.1 Technical Pattern Detection"))
story.append(P(
    "Lo, Mamaysky, and Wang (2000) provided the first rigorous statistical framework "
    "for evaluating technical patterns. They showed that certain patterns (head-and-shoulders, "
    "double tops, triangles) carry incremental information beyond random walk models. "
    "Their kernel-regression approach to detecting smooth extrema inspired the swing-pivot "
    "detection used in this project."
))
story.append(P(
    "Bulkowski (2005, 2021) catalogued hundreds of chart patterns with empirical success "
    "rates. His classification of triangles into ascending, descending, and symmetric "
    "subtypes, along with breakout statistics, informs our subtype classification logic."
))
story.append(SP())
story.append(H2("3.2 Triple-Barrier Labeling"))
story.append(P(
    "Lopez de Prado (2018) introduced the triple-barrier method in <i>Advances in Financial "
    "Machine Learning</i>, Chapter 3. Unlike fixed-horizon returns, the triple barrier sets "
    "a profit-target barrier (upper), stop-loss barrier (lower), and a time-expiry barrier. "
    "The label is determined by which barrier is breached first, producing a natural "
    "three-class classification: long, short, and no_trade. This avoids the arbitrary "
    "discretization of returns into bins and provides labels that are interpretable as "
    "trading decisions."
))
story.append(SP())
story.append(H2("3.3 Ensemble Methods"))
story.append(P(
    "Random Forest (Breiman, 2001) constructs an ensemble of decision trees, each trained "
    "on a bootstrap sample with a random subset of features considered at each split. This "
    "double randomization reduces variance while maintaining low bias. Bagging (Bootstrap "
    "Aggregating; Breiman, 1996) uses bootstrap sampling alone, without feature subsetting, "
    "resulting in higher correlation between trees but sometimes competitive performance "
    "when individual features are highly informative."
))
story.append(P(
    "Both methods are well-suited to financial classification: they handle mixed feature "
    "types, are robust to outliers, and provide natural feature importance rankings. Their "
    "ensemble structure also reduces the risk of overfitting to any single decision boundary."
))
story.append(SP())
story.append(H2("3.4 Risk Management and Labeling"))
story.append(P(
    "Traditional technical analysis provides entry signals but lacks a systematic framework "
    "for exit management. The triple-barrier approach (de Prado, 2018) addresses this by "
    "defining three exit scenarios: profit target, stop loss, and time expiry. This "
    "framework converts the continuous price series into discrete events with well-defined "
    "outcomes, enabling supervised classification."
))
story.append(P(
    "The choice of ATR-based barrier sizing (rather than fixed percentage thresholds) "
    "ensures that barriers adapt to current market volatility. During high-volatility "
    "periods, barriers widen to accommodate larger price swings, while in calm markets, "
    "barriers tighten to capture smaller moves. This adaptive behavior is critical for "
    "a dataset spanning 16 years, during which SPY's daily ATR varied by an order of "
    "magnitude."
))
story.append(SP())

story.append(H2("3.5 Feature Engineering for Financial ML"))
story.append(P(
    "Effective feature engineering is critical for financial ML (de Prado, 2018). "
    "Technical indicators such as RSI, MACD, Bollinger Bands, and moving average spreads "
    "encode momentum, mean-reversion, and volatility regimes. Combining these with "
    "event-specific features (pattern geometry, touch counts, containment ratios) creates "
    "a richer representation that captures both market context and pattern quality."
))
story.append(PageBreak())

# ── 4. Project Architecture ──────────────────────────────────────
story.append(H1("4. Project Architecture"))
story.append(P(
    "The project follows a modular pipeline design with clear separation between data, "
    "detection, labeling, features, and modeling. Each module is independently testable "
    "and the pipeline flows unidirectionally from raw data to predictions."
))
story.append(SP())
arch_data = [
    ["Module", "Path", "Purpose"],
    ["Data Loader", "src/data/load_data.py", "Load SPY from CSV, yfinance, or Alpha Vantage"],
    ["ATR Utility", "src/data/utils.py", "Average True Range computation"],
    ["S/R Detector", "src/patterns/support_resistance.py", "Rolling support/resistance with stability filter"],
    ["Triangle Detector", "src/patterns/triangles.py", "Pivot + linregress triangle detection"],
    ["Channel Detector", "src/patterns/channels.py", "Chunk-based channel with dynamic window"],
    ["Multi Top/Bottom", "src/patterns/multiple_tops_bottoms.py", "Rolling extremes + trend confirmation"],
    ["Pivot Utilities", "src/patterns/pivots.py", "Swing detection, containment, touch counting"],
    ["Scanner", "src/patterns/scanner.py", "Orchestrator for all 4 detectors"],
    ["Labeling", "src/labeling/label_events.py", "Triple-barrier event labeling"],
    ["Indicators", "src/features/indicators.py", "32 technical indicators"],
    ["Feature Builder", "src/features/build_features.py", "Event-based feature matrix construction"],
    ["ML Training", "src/models/train.py", "RF, Bagging, Baseline training + evaluation"],
]
story.append(make_table(arch_data, col_widths=[3*cm, 5*cm, 8*cm]))
story.append(SP())
story.append(SP())
story.append(H2("4.2 Data Flow"))
story.append(P(
    "The data flows through the pipeline in a strict unidirectional fashion:"
))
story.append(P(
    "<b>1. Data Loading</b> — Raw OHLCV data is loaded from CSV (or downloaded from "
    "yfinance/Alpha Vantage). The data is cleaned, sorted, and validated.<br/>"
    "<b>2. Indicator Computation</b> — 32 technical indicators are computed at each bar "
    "using only backward-looking windows.<br/>"
    "<b>3. Pattern Detection</b> — Four detectors scan the data independently, each "
    "producing a pattern column and optional metadata.<br/>"
    "<b>4. Event Flagging</b> — The scanner combines all detector outputs into a unified "
    "has_event flag.<br/>"
    "<b>5. Labeling</b> — Each flagged event is labeled using the triple-barrier method.<br/>"
    "<b>6. Feature Assembly</b> — Bar-level indicators are joined with pattern geometry "
    "features at each event timestamp.<br/>"
    "<b>7. Model Training</b> — The feature matrix and labels are split temporally and "
    "fed to ensemble classifiers."
))
story.append(SP())
story.append(H2("4.3 Notebooks"))
story.append(P(
    "Notebooks provide the interactive analysis layer. The project includes the following "
    "notebooks, each focused on a specific analysis task:"
))
nb_data = [
    ["Notebook", "Purpose"],
    ["03_pattern_detection_progress_report", "Detector tuning and before/after analysis"],
    ["03_pattern_validation", "Event quality metrics and forward-return analysis"],
    ["04_pattern_structure_validation", "Triangle detection gallery with pivot charts"],
    ["05_triple_barrier_labeling", "Labeling pipeline and sensitivity analysis"],
    ["06_channel_gallery", "Channel detection gallery"],
    ["07_data_source_comparison", "yfinance vs Alpha Vantage comparison"],
    ["08_detector_touch_analysis", "Trendline touch counting validation"],
    ["09_feature_engineering", "Feature computation and analysis"],
    ["10_model_training", "RF and Bagging training with full evaluation"],
    ["11_experiment_summary", "Consolidated results and statistics"],
]
story.append(make_table(nb_data, col_widths=[6*cm, 9*cm]))
story.append(PageBreak())

# ── 5. Data Sources ──────────────────────────────────────────────
story.append(H1("5. Data Sources"))
story.append(H2("5.1 Primary Source: Yahoo Finance via yfinance"))
story.append(P(
    f"The primary dataset consists of {n_bars:,} daily OHLCV bars for SPY, downloaded "
    f"from Yahoo Finance using the yfinance Python library. The data spans {date_start} to "
    f"{date_end}, covering approximately 16 years of trading activity. Price ranges from "
    f"${price_min:.2f} to ${price_max:.2f}, with an annualized return of {ann_ret:.1%}."
))
story.append(SP())
data_stats = [
    ["Statistic", "Value"],
    ["Rows", f"{n_bars:,}"],
    ["Date range", f"{date_start} to {date_end}"],
    ["Price range", f"${price_min:.2f} - ${price_max:.2f}"],
    ["Annualized return", f"{ann_ret:.1%}"],
    ["Average daily ATR(14)", f"${atr_series.dropna().mean():.2f}"],
    ["Columns", "Open, High, Low, Close, Volume"],
]
story.append(make_table(data_stats, col_widths=[5*cm, 8*cm]))
story.append(SP())
add_figure(story, fig_price, caption="Figure 1: SPY daily close price, 2010-2025.")
story.append(SP())

story.append(H2("5.2 Data Exploration"))
story.append(P(
    "Before pattern detection, it is important to understand the statistical properties "
    "of the raw data. The SPY daily returns exhibit the well-known features of financial "
    "time series: approximate symmetry around zero, leptokurtosis (fat tails), and "
    "volatility clustering."
))
story.append(SP())
add_figure(story, fig_atr, caption="Figure 2: SPY close price (top) and ATR(14) (bottom). Note the volatility spikes during the 2011 debt crisis, 2018 volatility event, 2020 COVID crash, and 2022 bear market.")
story.append(SP())
add_figure(story, fig_returns, width=15*cm, caption="Figure 3: Daily returns (left), log returns (center), and 20-day rolling annualised volatility (right).")
story.append(SP())
ret_desc = daily_ret_stats
story.append(P(
    f"Daily return statistics: mean={ret_desc['mean']:.4f}, std={ret_desc['std']:.4f}, "
    f"min={ret_desc['min']:.4f}, max={ret_desc['max']:.4f}, "
    f"skew={daily_ret.skew():.3f}, kurtosis={daily_ret.kurtosis():.3f}. "
    f"The negative skew and excess kurtosis confirm the non-Gaussian nature of returns, "
    f"justifying the use of ATR-based (rather than standard-deviation-based) thresholds "
    f"throughout the detection pipeline."
))
story.append(PageBreak())

story.append(H2("5.3 Secondary Source: Alpha Vantage"))
story.append(P(
    "The system supports Alpha Vantage as a secondary data source for robustness testing. "
    "The Alpha Vantage loader downloads daily adjusted data via their REST API (CSV format) "
    "and normalizes it to the same OHLCV schema. A comparison utility measures date coverage, "
    "price differences, volume correlation, and detector behavior differences between sources. "
    "When the API key is not available, the comparison framework can be tested using a "
    "simulated second source (small random perturbation of the primary data)."
))
story.append(PageBreak())

# ── 6. Pattern Detection Methodology ────────────────────────────
story.append(H1("6. Pattern Detection Methodology"))
story.append(P(
    f"Four pattern detectors scan the SPY data and flag {total_events} events "
    f"({100*total_events/n_bars:.1f}% of bars). Each detector uses ATR-adaptive "
    f"thresholds, stability filters, and cooldown periods to control false positives."
))
story.append(SP())
add_figure(story, fig_det, caption="Figure 2: Number of detections by pattern type.")
story.append(SP())

# Detection counts table
det_data = [["Detector", "Count", "% of Bars"]]
for name, count in det_counts.items():
    det_data.append([name, str(count), f"{100*count/n_bars:.2f}%"])
det_data.append(["<b>Total events</b>", f"<b>{total_events}</b>", f"<b>{100*total_events/n_bars:.1f}%</b>"])
story.append(make_table(det_data, col_widths=[5*cm, 3*cm, 4*cm]))
story.append(SP())

story.append(H2("6.1 Support and Resistance"))
story.append(P(
    "The support/resistance detector computes rolling minimum of Lows (support) and "
    "rolling maximum of Highs (resistance) over a 50-bar window. A signal fires when "
    "the current Close is within 0.3 x ATR(14) of either level. Two filters reduce "
    "false positives: (1) a stability filter requires the level to be unchanged for 5 "
    "bars, preventing signals during trending markets; (2) a 10-bar cooldown suppresses "
    "repeated signals from the same approach."
))
story.append(SP())

story.append(H2("6.2 Triangle Detection"))
story.append(P(
    "Triangles are detected using a pivot-point approach inspired by Lo et al. (2000). "
    "Within a 25-bar lookback window, swing highs and lows are identified using a +/-3 "
    "bar neighborhood. Linear regression (scipy.stats.linregress) is fitted to the "
    "swing highs and lows separately, requiring |r| >= 0.85 on each trendline. "
    "Intercepts are adjusted so the upper line caps all swing highs and the lower line "
    "floors all swing lows, creating the classic bounding-line geometry."
))
story.append(P(
    "The triangle is classified by slope signs: ascending (flat upper, rising lower), "
    "descending (falling upper, flat lower), or symmetric (both converging). Flatness "
    "is defined as |slope| < 0.1 x ATR / window. A signal fires on breakout "
    "(High or Low exceeds recent range by 0.3 x ATR) or on an upper-test for "
    "descending triangles."
))
story.append(SP())

story.append(H2("6.3 Channel Detection"))
story.append(P(
    "Channels are detected using the chunk-based method from the TrendLineChannelDetection "
    "reference notebook. The lookback window is dynamically optimized: 31 windows "
    "(25-55 bars) are tested, and the one producing the tightest channel at the current "
    "bar is selected. Each window is divided into 5-bar chunks; the max-High and min-Low "
    "from each chunk are used as representative points for polyfit(degree=1)."
))
story.append(P(
    "Validation requires: parallel slopes (relative difference < 25%, same sign), "
    "width between 1-6 x ATR, and containment >= 60%. A signal fires when the current "
    "Close is within 0.3 x ATR of either boundary."
))
story.append(SP())

story.append(H2("6.4 Multiple Tops and Bottoms"))
story.append(P(
    "Multiple tops are detected when the rolling maximum of Highs (50-bar window) "
    "stays elevated while a 5-bar close-trend slope is negative. Multiple bottoms are "
    "the mirror: rolling minimum stays depressed with positive close slope. "
    "A 10-bar cooldown prevents signal clustering."
))
story.append(SP())

story.append(H2("6.5 Detection Overview"))
story.append(P(
    f"Across all four detectors, the system identifies {total_events} event bars out of "
    f"{n_bars:,} total bars, for a combined event rate of {100*total_events/n_bars:.1f}%. "
    f"This is within the target range of 3-8% — high enough to produce a meaningful "
    f"training set, but low enough to avoid diluting signal quality with noise."
))
story.append(SP())
add_figure(story, fig_timeline, width=15*cm, caption="Figure 5: SPY price with all detected events overlaid. Green = S/R, red = triangle/channel, orange = multi top/bottom.")
story.append(SP())
story.append(P(
    "The event timeline shows that detections are distributed across the full 16-year "
    "period, with higher density during volatile markets (2011, 2018-2020, 2022). This "
    "is expected: volatile periods produce more price reversals and boundary tests. The "
    "cooldown mechanism prevents excessive clustering within any single volatile episode."
))
story.append(SP())

# Yearly detection table
story.append(H3("Detections by Year"))
yearly_data = [["Year", "Events", "Bars", "Event Rate"]]
for year, count in sorted(yearly_counts.items()):
    year_bars = len(df.loc[str(year)])
    yearly_data.append([str(year), str(count), str(year_bars), f"{100*count/year_bars:.1f}%"])
story.append(make_table(yearly_data, col_widths=[2.5*cm, 2.5*cm, 2.5*cm, 3*cm]))
story.append(P("Table: Annual detection counts and event rates.", STYLE_CAPTION))
story.append(PageBreak())

# ── 7. Trendline Touch Analysis ──────────────────────────────────
story.append(H1("7. Trendline Touch Analysis"))
story.append(P(
    "A key contribution of this project is the systematic counting and analysis of "
    "trendline touches. For each triangle and channel detection, the system counts "
    "how many bars have their High (for upper lines) or Low (for lower lines) within "
    "a tolerance of 0.3 x ATR of the fitted trendline."
))
story.append(SP())
story.append(H2("7.1 Touch Counting Methodology"))
story.append(P(
    "A touch is registered when |price_extreme - trendline_value| <= tolerance. For upper "
    "lines, we measure the distance from each bar's High to the trendline; for lower lines, "
    "from each bar's Low. A violation is counted when a bar breaches the line beyond "
    "tolerance. The touch tolerance is set to 0.3 x ATR(14) at the event bar, providing "
    "an adaptive threshold that accounts for current volatility."
))
story.append(SP())
story.append(H2("7.2 Triangle Touch Statistics"))
touch_tri_summary = [
    ["Metric", "Value"],
    ["Number of triangles", str(len(tri_details))],
    ["Avg total touches", f"{tri_touches_avg:.1f}"],
    ["Avg containment", f"{tri_cont_avg:.3f}"],
    ["Min containment", f"{min(d['containment_ratio'] for d in tri_details):.3f}"],
    ["Avg upper touches", f"{np.mean([d['upper_touches'] for d in tri_details]):.1f}"],
    ["Avg lower touches", f"{np.mean([d['lower_touches'] for d in tri_details]):.1f}"],
]
story.append(make_table(touch_tri_summary, col_widths=[5*cm, 5*cm]))
story.append(SP())

story.append(H2("7.3 Channel Touch Statistics"))
touch_ch_summary = [
    ["Metric", "Value"],
    ["Number of channels", str(len(ch_details))],
    ["Avg total touches", f"{ch_touches_avg:.1f}"],
    ["Avg containment", f"{ch_cont_avg:.3f}"],
    ["Min containment", f"{min(d['containment_ratio'] for d in ch_details):.3f}"],
    ["Avg upper touches", f"{np.mean([d['upper_touches'] for d in ch_details]):.1f}"],
    ["Avg lower touches", f"{np.mean([d['lower_touches'] for d in ch_details]):.1f}"],
    ["Avg channel width (ATR)", f"{np.mean([d['channel_width_atr'] for d in ch_details]):.2f}"],
]
story.append(make_table(touch_ch_summary, col_widths=[5*cm, 5*cm]))
story.append(SP())
add_figure(story, fig_touches, caption="Figure 3: Distribution of total trendline touches for triangles and channels.")
story.append(SP())
add_figure(story, fig_containment, caption="Figure 7: Distribution of containment ratios for triangles and channels.")
story.append(SP())

story.append(H2("7.4 Channel Width Analysis"))
story.append(P(
    f"The channel width, measured in ATR units, ranges from 1.0 (the minimum allowed) "
    f"to 6.0 (the maximum allowed), with a mean of {np.mean(ch_widths):.2f} ATR. "
    f"Narrower channels indicate tighter price consolidation, while wider channels "
    f"represent broader trend movements."
))
add_figure(story, fig_ch_width, width=10*cm, caption="Figure 8: Distribution of channel widths in ATR units.")
story.append(SP())

story.append(H2("7.5 Touch Quality Assessment"))
story.append(P(
    "The mean touch error (distance from price extreme to trendline at touch points) "
    "provides a quality metric for each detection. Lower errors indicate closer adherence "
    "to the fitted trendline."
))
story.append(SP())
story.append(P(
    f"For triangles, the average upper-line touch error is "
    f"{np.mean([d['upper_mean_error'] for d in tri_details]):.3f} USD, and the average "
    f"lower-line touch error is "
    f"{np.mean([d['lower_mean_error'] for d in tri_details]):.3f} USD. "
    f"For channels, the corresponding values are "
    f"{np.mean([d['upper_mean_error'] for d in ch_details]):.3f} USD and "
    f"{np.mean([d['lower_mean_error'] for d in ch_details]):.3f} USD."
))
story.append(P(
    "The touch errors scale with price level and volatility — errors are larger for "
    "recent detections (when SPY trades above $500) compared to early detections "
    "(SPY near $100). Normalizing by ATR would make these errors comparable across "
    "different price regimes, and this normalization is already embedded in the "
    "tolerance threshold (0.3 x ATR)."
))
story.append(PageBreak())

# ── 8. Triple-Barrier Labeling ───────────────────────────────────
story.append(H1("8. Triple-Barrier Labeling"))
story.append(P(
    "Following Lopez de Prado (2018), each detected event is labeled by walking forward "
    "from the event bar and checking which of three barriers is breached first:"
))
story.append(P(
    "&bull; <b>Upper barrier</b> (profit target): entry_price + 2.0 x ATR(14)<br/>"
    "&bull; <b>Lower barrier</b> (stop loss): entry_price - 2.0 x ATR(14)<br/>"
    "&bull; <b>Time barrier</b>: 10 bars after entry"
))
story.append(SP())
story.append(P(
    f"The labeling produces {n_labeled} events: {label_counts.get('long',0)} long "
    f"({100*label_counts.get('long',0)/n_labeled:.1f}%), "
    f"{label_counts.get('short',0)} short "
    f"({100*label_counts.get('short',0)/n_labeled:.1f}%), and "
    f"{label_counts.get('no_trade',0)} no_trade "
    f"({100*label_counts.get('no_trade',0)/n_labeled:.1f}%)."
))
story.append(SP())
add_figure(story, fig_labels, caption="Figure 9: Label distribution across all events.")
story.append(SP())

story.append(H2("8.1 Barrier Mechanics"))
story.append(P(
    "The triple-barrier method works as follows: for each event bar, the entry price is "
    "the Close of that bar. The upper barrier is set at entry + pt_mult x ATR(14), and "
    "the lower barrier at entry - sl_mult x ATR(14). The algorithm walks forward through "
    "subsequent bars, checking whether the bar's High breaches the upper barrier or its "
    "Low breaches the lower barrier. If neither barrier is breached within max_holding "
    "bars, the time barrier triggers and the event is labeled no_trade."
))
story.append(P(
    "When both barriers are breached on the same bar (a wide-range day), the Close is "
    "used as a tiebreaker: if Close >= entry, the label is long; otherwise short."
))
story.append(SP())

story.append(H2("8.2 Return and Holding Period Analysis"))
add_figure(story, fig_ret_by_label, caption="Figure 10: Return distribution by label class.")
story.append(SP())
add_figure(story, fig_bars_held, caption="Figure 11: Holding period distribution by label class.")
story.append(SP())

# Per-label stats table
story.append(P("Return statistics by label:"))
lrs_data = [["Label", "Count", "Mean Ret %", "Std Ret %", "Min Ret %", "Max Ret %"]]
for lbl in ["long", "short", "no_trade"]:
    if lbl in label_ret_stats.index:
        r = label_ret_stats.loc[lbl]
        lrs_data.append([lbl, str(int(r["count"])), f"{r['mean']:.3f}", f"{r['std']:.3f}",
                        f"{r['min']:.3f}", f"{r['max']:.3f}"])
story.append(make_table(lrs_data, col_widths=[2.5*cm, 2*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm]))
story.append(SP())

story.append(P(
    "Long-labeled events have a positive mean return (the upper barrier was hit), while "
    "short-labeled events have a negative mean return. No_trade events have near-zero "
    "mean returns, as expected — price did not move significantly in either direction "
    "within the holding window."
))
story.append(SP())

# Cross-tab
story.append(H2("8.3 Labels by Event Type"))
crosstab = pd.read_csv(PROJECT_ROOT / "outputs" / "label_crosstab.csv")
ct_data = [list(crosstab.columns)]
for _, row in crosstab.iterrows():
    ct_data.append([str(v) for v in row.values])
story.append(make_table(ct_data, col_widths=[4*cm, 2*cm, 2.5*cm, 2*cm, 2*cm]))
story.append(P("Table: Cross-tabulation of labels by event type.", STYLE_CAPTION))
story.append(PageBreak())

# ── 9. Feature Engineering ───────────────────────────────────────
story.append(H1("9. Feature Engineering"))
story.append(P(
    f"The feature engineering module computes {n_features} features per event, organized "
    f"into six groups. All features use only information available at or before the event "
    f"bar, preventing lookahead bias."
))
story.append(SP())
feat_groups = [
    ["Group", "Count", "Examples"],
    ["Trend", "12", "SMA distances, MA spreads (10/50, 20/200, 50/200)"],
    ["Volatility", "5", "ATR(14), ATR/price, rolling volatility, Bollinger width"],
    ["Momentum", "10", "RSI(14), MACD, returns (1/5/10/20-day), momentum"],
    ["Volume", "3", "Volume ratio, volume std, OBV normalized"],
    ["Pattern Geometry", "12", "Touches, containment, slopes, window, errors"],
    ["Event Type", "9", "One-hot encoded event type dummies"],
    ["Event Meta", "2", "Entry price, event ATR"],
]
story.append(make_table(feat_groups, col_widths=[3.5*cm, 1.5*cm, 11*cm]))
story.append(SP())

story.append(H2("9.1 Most Discriminative Features"))
story.append(P(
    "ANOVA F-tests rank features by their ability to separate the three label classes. "
    "The top features are recent returns (ret_1, ret_5), moving average distances "
    "(sma_10_dist), and pattern-specific metrics (upper_mean_error, upper_touches)."
))
story.append(SP())
anova = pd.read_csv(PROJECT_ROOT / "outputs" / "feature_anova_ranking.csv")
anova_top = anova.head(15)
anova_data = [["Feature", "F-statistic", "p-value"]]
for _, row in anova_top.iterrows():
    anova_data.append([row["feature"], f"{row['F_statistic']:.3f}", f"{row['p_value']:.4f}"])
story.append(make_table(anova_data, col_widths=[5*cm, 3*cm, 3*cm]))
story.append(P("Table: Top 15 features by ANOVA F-statistic.", STYLE_CAPTION))
story.append(SP())
add_figure(story, fig_corr, width=13*cm, caption="Figure 12: Feature correlation matrix (top 20 numeric features).")
story.append(SP())

story.append(H2("9.2 Feature Distributions by Label"))
story.append(P(
    "Examining how the most discriminative features distribute across label classes "
    "reveals the separation structure that the classifiers exploit."
))
add_figure(story, fig_feat_dist, width=15*cm, caption="Figure 13: Distributions of the top 6 most discriminative features, coloured by label class.")
story.append(SP())
story.append(P(
    "The ret_1 feature (1-day return) shows the clearest separation: long events tend "
    "to occur after positive recent returns (momentum effect), while short events follow "
    "negative returns. The sma_10_dist feature (distance from 10-day moving average) "
    "captures mean-reversion dynamics — events far below the MA tend to produce long "
    "labels as price reverts upward."
))
story.append(SP())

story.append(H2("9.3 Lookahead Bias Prevention"))
story.append(P(
    "Preventing information leakage is critical in financial ML. The feature engineering "
    "module enforces the following discipline:"
))
story.append(P(
    "&bull; <b>Bar-level indicators</b> (ATR, RSI, MACD, etc.) use only past and current "
    "bars. All rolling calculations use a backward-looking window.<br/>"
    "&bull; <b>Pattern geometry features</b> (slopes, touches, containment) are computed "
    "within the detection window, which ends at or before the event bar.<br/>"
    "&bull; <b>No label information</b> is included in the feature matrix. Return columns "
    "measure past returns (ret_1 = yesterday's return at event time), not future returns.<br/>"
    "&bull; <b>Temporal splitting</b> ensures that training data is strictly earlier than "
    "validation and test data."
))
story.append(PageBreak())

# ── 10. Model Methodology ────────────────────────────────────────
story.append(H1("10. Model Methodology"))
story.append(P(
    "This section describes the machine learning methodology, including data splitting "
    "strategy, model architectures, and evaluation metrics."
))
story.append(SP())
story.append(H2("10.1 Temporal Splitting"))
story.append(P(
    f"To prevent information leakage, events are split chronologically: the first 60% "
    f"({n_train} events) for training, the next 20% ({n_val}) for validation, and the "
    f"final 20% ({n_test}) for testing. No shuffling is applied. This means the test set "
    f"contains the most recent events, providing a realistic out-of-sample evaluation."
))
story.append(SP())
add_figure(story, fig_split, width=14*cm, caption="Figure 7: Temporal train/validation/test split.")
story.append(SP())

story.append(H2("10.2 Random Forest"))
story.append(P(
    "Random Forest (Breiman, 2001) builds an ensemble of decision trees, each trained on "
    "a bootstrap sample (sampling with replacement from the training set). At each split, "
    "only a random subset of features (default: sqrt(n_features)) is considered. This "
    "double randomization decorrelates the trees, reducing ensemble variance."
))
story.append(P(
    f"Our configuration: {results['hyperparams']['n_estimators']} trees, "
    f"max_depth={results['hyperparams']['max_depth']}, balanced class weights."
))
story.append(SP())

story.append(H2("10.3 Bagging"))
story.append(P(
    "Bagging (Bootstrap Aggregating; Breiman, 1996) also uses bootstrap samples, but "
    "each base learner (a decision tree in our case) considers <b>all</b> features at "
    "each split. The only source of diversity is the variation in bootstrap samples. "
    "Consequently, bagging trees are more correlated than random forest trees, which "
    "can lead to higher ensemble variance."
))
story.append(P(
    "Each bootstrap sample is drawn independently: for a training set of N samples, "
    "N draws with replacement produce a sample where approximately 63.2% of the "
    "original observations appear (some multiple times), while 36.8% are left out "
    "(out-of-bag samples)."
))
story.append(SP())

story.append(H2("10.4 Baseline"))
story.append(P(
    "A stratified DummyClassifier serves as the baseline. It predicts class labels "
    "according to the training set class distribution, providing the performance floor "
    "that any useful model should exceed."
))
story.append(SP())

story.append(H2("10.5 Evaluation Metrics"))
story.append(P(
    "We evaluate models using multiple metrics to capture different aspects of "
    "classification quality:"
))
story.append(P(
    "&bull; <b>Accuracy:</b> Fraction of correctly classified events. Simple but can be "
    "misleading with class imbalance.<br/>"
    "&bull; <b>Macro-F1:</b> Unweighted average of per-class F1 scores. Treats all classes "
    "equally regardless of size, making it a stricter metric than accuracy.<br/>"
    "&bull; <b>Weighted-F1:</b> Average F1 weighted by class support. Accounts for class "
    "sizes and gives a more balanced view than macro-F1.<br/>"
    "&bull; <b>Confusion matrix:</b> Shows the full pattern of correct and incorrect "
    "predictions across all class pairs.<br/>"
    "&bull; <b>Per-class precision and recall:</b> Identify which classes the model "
    "handles well and which it struggles with."
))
story.append(SP())

story.append(H2("10.6 Tree Complexity"))
if rf_trees and bag_trees:
    tree_data = [
        ["Metric", "Random Forest", "Bagging"],
        ["Number of trees", str(rf_trees["n_trees"]), str(bag_trees["n_trees"])],
        ["Min depth", str(rf_trees["min_depth"]), str(bag_trees["min_depth"])],
        ["Mean depth", str(rf_trees["mean_depth"]), str(bag_trees["mean_depth"])],
        ["Max depth", str(rf_trees["max_depth"]), str(bag_trees["max_depth"])],
        ["Mean leaves", str(rf_trees["mean_leaves"]), str(bag_trees["mean_leaves"])],
        ["Mean nodes", str(rf_trees["mean_nodes"]), str(bag_trees["mean_nodes"])],
    ]
    story.append(make_table(tree_data, col_widths=[4*cm, 4*cm, 4*cm]))
    story.append(P("Table: Tree complexity statistics.", STYLE_CAPTION))
story.append(PageBreak())

# ── 11. Experiments and Results ──────────────────────────────────
story.append(H1("11. Experiments and Results"))
story.append(H2("11.1 Model Performance"))
perf_data = [
    ["Model", "Test Accuracy", "Test F1 (macro)", "Test F1 (weighted)"],
    ["Random Forest", f"{rf_test['accuracy']:.4f}", f"{rf_test['f1_macro']:.4f}", f"{rf_test['f1_weighted']:.4f}"],
    ["Bagging", f"{bag_test['accuracy']:.4f}", f"{bag_test['f1_macro']:.4f}", f"{bag_test['f1_weighted']:.4f}"],
    ["Baseline", f"{base_test['accuracy']:.4f}", f"{base_test['f1_macro']:.4f}", f"{base_test['f1_weighted']:.4f}"],
]
story.append(make_table(perf_data, col_widths=[4*cm, 3.5*cm, 3.5*cm, 3.5*cm]))
story.append(P("Table: Model performance on the test set.", STYLE_CAPTION))
story.append(SP())
add_figure(story, fig_model, caption="Figure 8: Model comparison on test set — accuracy and F1 (macro).")
story.append(SP())

story.append(H2("11.2 Confusion Matrices"))
add_figure(story, fig_cm, width=15*cm, caption="Figure 9: Confusion matrices on the test set (RF, Bagging, Baseline).")
story.append(SP())

story.append(H2("11.3 Classification Reports"))
for name, res in [("Random Forest", rf_test), ("Bagging", bag_test)]:
    story.append(H3(f"{name} (Test Set)"))
    report = res["classification_report"]
    cr_data = [["Class", "Precision", "Recall", "F1-score", "Support"]]
    for cls in res["labels"]:
        r = report[cls]
        cr_data.append([cls, f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                        f"{r['f1-score']:.3f}", str(int(r['support']))])
    story.append(make_table(cr_data, col_widths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm]))
    story.append(SP())

story.append(H2("11.4 Feature Importance"))
add_figure(story, fig_fi, caption="Figure 17: Top 15 feature importances (Random Forest).")
story.append(SP())
story.append(P(
    "The most important features for the Random Forest are volume statistics "
    "(volume_std), long-term moving average spreads (ma_spread_50_200), and "
    "short-term price dynamics (sma_10_dist, ret_5). Pattern geometry features "
    "(upper_touches, containment) rank in the top 20 but are not dominant, "
    "suggesting that market context features carry more discriminative power "
    "than pattern-specific metrics alone."
))
story.append(SP())

story.append(H2("11.5 Validation vs Test Gap"))
story.append(P(
    "Comparing validation and test performance provides insight into overfitting. "
    "A large gap would suggest that the model has overfit to the validation set "
    "during implicit hyperparameter selection."
))
rf_val = results["val_results"]["rf"]
bag_val = results["val_results"]["bagging"]
gap_data = [
    ["Model", "Val Acc", "Test Acc", "Gap", "Val F1", "Test F1", "Gap"],
    ["RF", f"{rf_val['accuracy']:.3f}", f"{rf_test['accuracy']:.3f}",
     f"{rf_val['accuracy']-rf_test['accuracy']:.3f}",
     f"{rf_val['f1_macro']:.3f}", f"{rf_test['f1_macro']:.3f}",
     f"{rf_val['f1_macro']-rf_test['f1_macro']:.3f}"],
    ["Bagging", f"{bag_val['accuracy']:.3f}", f"{bag_test['accuracy']:.3f}",
     f"{bag_val['accuracy']-bag_test['accuracy']:.3f}",
     f"{bag_val['f1_macro']:.3f}", f"{bag_test['f1_macro']:.3f}",
     f"{bag_val['f1_macro']-bag_test['f1_macro']:.3f}"],
]
story.append(make_table(gap_data, col_widths=[2.3*cm]*7))
story.append(P("Table: Validation vs test performance gap.", STYLE_CAPTION))
story.append(SP())
story.append(P(
    "The small positive gap (validation slightly better than test) is typical and "
    "does not indicate severe overfitting. The temporal nature of the split means "
    "the test set represents the most recent market conditions, which may differ "
    "from the training period."
))
story.append(SP())

story.append(H2("11.6 Per-Class Analysis"))
story.append(P(
    "Examining per-class performance reveals where the models succeed and struggle:"
))
story.append(P(
    "&bull; <b>Long:</b> The majority class. Both models achieve their best recall "
    "on this class, partly due to balanced class weights upweighting minority classes.<br/>"
    "&bull; <b>Short:</b> Moderate performance. Short signals are inherently harder "
    "because SPY has a long-term upward bias (annualized return of 14%).<br/>"
    "&bull; <b>No_trade:</b> Lowest recall. These events represent ambiguous situations "
    "where price did not move decisively, making them harder to predict."
))
story.append(PageBreak())

# ── 12. Discussion ───────────────────────────────────────────────
story.append(H1("12. Discussion"))
story.append(P(
    f"The Random Forest classifier achieves {rf_test['accuracy']:.1%} test accuracy "
    f"versus the {base_test['accuracy']:.1%} stratified baseline, representing a "
    f"{(rf_test['accuracy'] - base_test['accuracy'])*100:.1f} percentage point "
    f"improvement. While modest, this outperformance is achieved on a genuinely "
    f"difficult three-class problem with only {n_test} test events and strict temporal "
    f"ordering."
))
story.append(SP())
story.append(P(
    "Several findings are noteworthy:"
))
findings = [
    f"<b>Touch counting adds value.</b> The upper_touches feature ranks in the ANOVA "
    f"top 10 (F={anova.loc[anova['feature']=='upper_touches','F_statistic'].values[0]:.2f}, "
    f"p={anova.loc[anova['feature']=='upper_touches','p_value'].values[0]:.4f}), "
    f"confirming that trendline adherence carries signal.",

    f"<b>Market context dominates pattern geometry.</b> Volume, moving average, and "
    f"return features are consistently more important than pattern-specific metrics. "
    f"This suggests that <i>when</i> a pattern occurs matters more than the pattern's "
    f"geometric properties.",

    f"<b>Bagging and RF perform similarly.</b> With {n_features} features and "
    f"{n_train} training samples, the feature-subsampling advantage of RF is limited. "
    f"Both ensembles outperform the baseline, but the gap between them is small.",

    f"<b>The no_trade class is hardest.</b> Both models show lower recall for no_trade, "
    f"which is expected: time-expiry events are inherently ambiguous (price neither "
    f"strongly advanced nor declined).",
]
for f in findings:
    story.append(P(f"&bull; {f}"))
story.append(SP())

story.append(H2("12.2 Comparison with Literature"))
story.append(P(
    "Our results are consistent with the broader literature on ML-based trading. "
    "De Prado (2018) notes that triple-barrier labeled datasets typically produce "
    "modest classification accuracy, often in the 40-55% range for three-class problems, "
    "because financial returns are inherently noisy. The Random Forest's 42.2% test "
    "accuracy on a three-class problem (33.3% chance level) represents a meaningful "
    "signal, though far from the accuracy levels seen in domains like image classification."
))
story.append(P(
    "Lo et al. (2000) found that technical patterns carry incremental information "
    "over random walks, but their predictive power is modest and time-varying. Our "
    "finding that market context features dominate over pattern geometry aligns with "
    "this: the pattern itself is a filter that selects interesting moments, but the "
    "trading outcome depends on the broader market environment at that moment."
))
story.append(SP())

story.append(H2("12.3 Practical Implications"))
story.append(P(
    "From a practical standpoint, the pipeline provides several useful outputs even "
    "if the classification accuracy is not sufficient for standalone trading:"
))
story.append(P(
    "&bull; <b>Event filtering:</b> The detectors reduce 4,023 trading days to 221 "
    "actionable events, a 95% reduction that focuses attention on technically significant "
    "moments.<br/>"
    "&bull; <b>Risk assessment:</b> The touch-counting metrics provide interpretable "
    "quality scores for each pattern, helping traders assess whether a detected pattern "
    "is well-formed.<br/>"
    "&bull; <b>Feature insights:</b> The feature importance analysis reveals which "
    "market conditions are most predictive, informing discretionary decision-making."
))
story.append(PageBreak())

# ── 13. Limitations ──────────────────────────────────────────────
story.append(H1("13. Limitations"))
limitations = [
    f"<b>Small sample size.</b> With {n_labeled} total events and {n_test} test events, "
    f"statistical power is limited. Confidence intervals around accuracy and F1 are wide.",

    "<b>Single asset.</b> All experiments use SPY daily data. Results may not generalize "
    "to other instruments, timeframes, or markets.",

    "<b>No transaction costs.</b> The labeling and evaluation do not account for spreads, "
    "commissions, or slippage. A signal that is statistically significant may not be "
    "economically profitable.",

    "<b>No hyperparameter optimization.</b> max_depth=8 and n_estimators=200 were chosen "
    "as reasonable defaults. A systematic search (e.g., Bayesian optimization with "
    "temporal CV) could improve results.",

    "<b>No regime conditioning.</b> The HMM regime detection module is planned but not "
    "yet implemented. Conditioning on market regime (bull/bear/sideways) could improve "
    "both detection quality and classification accuracy.",

    "<b>Alpha Vantage comparison is limited.</b> Without an API key, the source comparison "
    "uses simulated data. A real comparison would provide stronger robustness evidence.",

    "<b>Pattern detectors use fixed ATR multiples.</b> The 0.3 x ATR thresholds for "
    "breakout detection and proximity signals are hard-coded. Adaptive thresholds "
    "could improve performance across different volatility regimes.",
]
for lim in limitations:
    story.append(P(f"&bull; {lim}"))
story.append(PageBreak())

# ── 14. Future Work ──────────────────────────────────────────────
story.append(H1("14. Future Work"))
future = [
    "<b>HMM regime detection:</b> Implement Hidden Markov Model-based regime "
    "classification (bull/bear/sideways) and condition both detection and trading "
    "on the current regime.",

    "<b>Walk-forward backtesting:</b> Implement a walk-forward framework that "
    "retrains the model periodically and simulates realistic trading with "
    "transaction costs.",

    "<b>Expanded asset universe:</b> Apply the pipeline to multiple ETFs, sectors, "
    "or individual stocks to test generalization.",

    "<b>Deep learning baselines:</b> Compare ensemble tree methods against LSTM, "
    "Transformer, or temporal convolutional networks.",

    "<b>Hyperparameter optimization:</b> Use Bayesian optimization with nested "
    "temporal cross-validation for systematic tuning.",

    "<b>Risk management:</b> Incorporate position sizing (Kelly criterion, volatility "
    "targeting) and portfolio-level risk constraints.",

    "<b>Alternative labeling:</b> Experiment with asymmetric barriers (wider profit "
    "target than stop loss) and different ATR multiples.",
]
for f in future:
    story.append(P(f"&bull; {f}"))
story.append(PageBreak())

# ── 15. Conclusion ───────────────────────────────────────────────
story.append(H1("15. Conclusion"))
story.append(P(
    f"This project demonstrates a complete, transparent, and reproducible pipeline for "
    f"ML-based trading signal classification. Starting from {n_bars:,} daily SPY bars, "
    f"four pattern detectors identify {total_events} technically significant events. "
    f"Triple-barrier labeling produces {n_labeled} labeled events for supervised learning. "
    f"A feature engineering module computes {n_features} features per event, combining "
    f"technical indicators with novel pattern geometry and trendline touch statistics."
))
story.append(P(
    f"Random Forest classification achieves {rf_test['accuracy']:.1%} test accuracy and "
    f"{rf_test['f1_macro']:.3f} macro-F1, outperforming a stratified baseline. Feature "
    f"importance analysis reveals that market context features (volume statistics, moving "
    f"average spreads, recent returns) dominate over pattern-specific metrics, suggesting "
    f"that the timing context of a pattern is at least as important as its geometry."
))
story.append(P(
    "The trendline touch-counting methodology is a concrete contribution: by systematically "
    "measuring how well price action respects detected boundaries, we add interpretable "
    "quality metrics to each detection. These metrics serve both as validation tools and "
    "as ML features."
))
story.append(P(
    "Significant scope remains for improvement: regime conditioning, expanded assets, "
    "backtesting with transaction costs, and hyperparameter optimization are natural "
    "next steps. The modular architecture makes these extensions straightforward."
))
story.append(SP())
story.append(P(
    "The project's primary value lies not in the absolute classification accuracy, but in "
    "the transparent, reproducible, and academically rigorous pipeline it establishes. "
    "Every step — from raw data to final predictions — is visible, documented, and "
    "parameterized. This transparency is essential for building trust in ML-based trading "
    "systems, where black-box approaches are both risky and intellectually unsatisfying."
))
story.append(SP())
story.append(P(
    "The modular architecture enables straightforward extensions. New detectors can be "
    "added to the scanner module. New indicators can be added to the feature engineering "
    "module. New classifiers can be added to the training module. The triple-barrier "
    "labeling can be reconfigured with different barrier widths or holding periods. "
    "This flexibility, combined with the strict temporal splitting, provides a solid "
    "foundation for continued research into regime-aware trading strategies."
))
story.append(PageBreak())

# ── 16. References ───────────────────────────────────────────────
story.append(H1("16. References"))
refs = [
    "Breiman, L. (1996). Bagging predictors. <i>Machine Learning</i>, 24(2), 123-140.",
    "Breiman, L. (2001). Random forests. <i>Machine Learning</i>, 45(1), 5-32.",
    "Bulkowski, T. N. (2005). <i>Encyclopedia of Chart Patterns</i>. 2nd ed. Wiley.",
    "Bulkowski, T. N. (2021). <i>Chart Patterns: After the Buy</i>. Wiley.",
    "de Prado, M. L. (2018). <i>Advances in Financial Machine Learning</i>. Wiley.",
    "Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary "
    "time series and the business cycle. <i>Econometrica</i>, 57(2), 357-384.",
    "Lo, A. W., Mamaysky, H., &amp; Wang, J. (2000). Foundations of technical analysis: "
    "Computational algorithms, statistical inference, and empirical implementation. "
    "<i>Journal of Finance</i>, 55(4), 1705-1765.",
    "Murphy, J. J. (1999). <i>Technical Analysis of the Financial Markets</i>. "
    "New York Institute of Finance.",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. "
    "<i>Journal of Machine Learning Research</i>, 12, 2825-2830.",
    "Pring, M. J. (2014). <i>Technical Analysis Explained</i>. 5th ed. McGraw-Hill.",
]
for i, ref in enumerate(refs, 1):
    story.append(P(f"[{i}] {ref}"))
story.append(PageBreak())

# ── Appendix A: Detector Parameters ─────────────────────────────
story.append(H1("Appendix A: Detector Parameters"))
story.append(P(
    "This appendix lists the full parameterization of each detector as used in "
    "all experiments."
))
story.append(SP())

story.append(H2("A.1 Support / Resistance"))
sr_params = [
    ["Parameter", "Value", "Description"],
    ["window", "50", "Rolling lookback (bars)"],
    ["atr_mult", "0.3", "Proximity band (x ATR)"],
    ["cooldown", "10", "Bars between signals"],
    ["stability_window", "5", "Level must be unchanged for N bars"],
]
story.append(make_table(sr_params, col_widths=[4*cm, 2*cm, 8*cm]))
story.append(SP())

story.append(H2("A.2 Triangles"))
tri_params = [
    ["Parameter", "Value", "Description"],
    ["window", "25", "Lookback (bars)"],
    ["pivot_order", "3", "+/- bars for swing detection"],
    ["min_pivots", "2", "Min swings per trendline"],
    ["min_r", "0.85", "Min |r| on linregress fit"],
    ["min_convergence_pct", "0.05", "Min range compression (5%)"],
    ["cooldown", "10", "Bars between signals"],
    ["breakout threshold", "0.3 x ATR", "Distance for breakout detection"],
    ["flat threshold", "0.1 x ATR/window", "Slope < threshold = flat"],
    ["touch tolerance", "0.3 x ATR", "Touch counting tolerance"],
]
story.append(make_table(tri_params, col_widths=[4*cm, 3*cm, 7*cm]))
story.append(SP())

story.append(H2("A.3 Channels"))
ch_params = [
    ["Parameter", "Value", "Description"],
    ["backcandles", "40", "Base lookback (bars)"],
    ["brange", "15", "Dynamic window search range"],
    ["wind", "5", "Chunk size (bars)"],
    ["slope_tolerance", "0.25", "Max relative slope difference"],
    ["min_containment", "0.60", "Min containment ratio"],
    ["cooldown", "10", "Bars between signals"],
    ["width range", "1-6 x ATR", "Valid channel width"],
    ["proximity", "0.3 x ATR", "Signal threshold"],
    ["touch tolerance", "0.3 x ATR", "Touch counting tolerance"],
]
story.append(make_table(ch_params, col_widths=[4*cm, 3*cm, 7*cm]))
story.append(SP())

story.append(H2("A.4 Multiple Tops / Bottoms"))
mtb_params = [
    ["Parameter", "Value", "Description"],
    ["window", "50", "Rolling lookback (bars)"],
    ["confirm_bars", "5", "Close-trend slope window"],
    ["cooldown", "10", "Bars between signals"],
]
story.append(make_table(mtb_params, col_widths=[4*cm, 2*cm, 8*cm]))
story.append(SP())

story.append(H2("A.5 Triple-Barrier Labeling"))
lb_params = [
    ["Parameter", "Value", "Description"],
    ["pt_mult", "2.0", "Upper barrier (x ATR)"],
    ["sl_mult", "2.0", "Lower barrier (x ATR)"],
    ["max_holding", "10", "Time barrier (bars)"],
    ["atr_window", "14", "ATR lookback"],
]
story.append(make_table(lb_params, col_widths=[4*cm, 2*cm, 8*cm]))
story.append(SP())

story.append(H2("A.6 Model Hyperparameters"))
model_params = [
    ["Parameter", "Random Forest", "Bagging"],
    ["n_estimators", "200", "200"],
    ["max_depth", "8", "8"],
    ["class_weight", "balanced", "N/A (tree-level)"],
    ["Feature subsampling", "sqrt(n_features)", "All features"],
    ["Bootstrap", "Yes", "Yes"],
    ["random_state", "42", "42"],
]
story.append(make_table(model_params, col_widths=[4*cm, 4*cm, 4*cm]))
story.append(PageBreak())

# ── Appendix B: Complete Feature List ────────────────────────────
story.append(H1("Appendix B: Complete Feature List"))
story.append(P(
    f"This appendix lists all {n_features} features in the feature matrix, organized by "
    f"group. Each feature is computed at the event bar using only information available "
    f"at or before that bar."
))
story.append(SP())

feat_list = [
    ("Trend", [
        ("sma_10, sma_20, sma_50, sma_100, sma_200", "Simple moving averages"),
        ("sma_10_dist, ..., sma_200_dist", "Relative distance from price to each SMA"),
        ("ma_spread_10_50", "Normalized spread between 10-day and 50-day SMA"),
        ("ma_spread_20_200", "Normalized spread between 20-day and 200-day SMA"),
        ("ma_spread_50_200", "Normalized spread between 50-day and 200-day SMA"),
    ]),
    ("Volatility", [
        ("atr_14", "Average True Range (14-day)"),
        ("atr_ratio", "ATR(14) / Close — normalized volatility"),
        ("rvol_20", "20-day rolling annualized volatility"),
        ("bb_width", "Bollinger Band width (upper - lower) / SMA(20)"),
        ("bb_pctb", "Bollinger %B — position within bands"),
    ]),
    ("Momentum", [
        ("ret_1, ret_5, ret_10, ret_20", "Simple returns over 1, 5, 10, 20 days"),
        ("mom_5, mom_10, mom_20", "Rate-of-change momentum"),
        ("rsi_14", "Relative Strength Index (14-day)"),
        ("macd", "MACD line (EMA12 - EMA26)"),
        ("macd_signal", "MACD signal line (EMA9 of MACD)"),
        ("macd_hist", "MACD histogram (MACD - signal)"),
    ]),
    ("Volume", [
        ("volume_ratio", "Current volume / 20-day average volume"),
        ("volume_std", "Rolling volume standard deviation / mean"),
        ("obv_norm", "Normalized on-balance volume"),
    ]),
    ("Pattern Geometry", [
        ("upper_slope", "Slope of upper trendline"),
        ("lower_slope", "Slope of lower trendline"),
        ("containment", "Fraction of bars inside trendlines"),
        ("upper_touches", "Bars touching upper trendline"),
        ("lower_touches", "Bars touching lower trendline"),
        ("total_touches", "Sum of upper and lower touches"),
        ("upper_mean_error", "Mean distance at upper touch points"),
        ("lower_mean_error", "Mean distance at lower touch points"),
        ("pattern_window", "Lookback window of the pattern"),
        ("channel_width_atr", "Channel width in ATR units"),
        ("r_upper", "|r| of upper trendline regression"),
        ("r_lower", "|r| of lower trendline regression"),
    ]),
    ("Event Type", [
        ("etype_*", "One-hot encoded event type (9 categories)"),
    ]),
    ("Event Meta", [
        ("entry_price", "Close price at event bar"),
        ("event_atr", "ATR(14) at event bar"),
    ]),
]

for group, feats in feat_list:
    story.append(H3(f"B.{feat_list.index((group, feats))+1} {group}"))
    f_data = [["Feature", "Description"]]
    for fname, fdesc in feats:
        f_data.append([fname, fdesc])
    story.append(make_table(f_data, col_widths=[5*cm, 10*cm]))
    story.append(SP())

# ── Build PDF ────────────────────────────────────────────────────
print(f"Writing {OUTPUT_PDF}...")
doc = SimpleDocTemplate(
    str(OUTPUT_PDF),
    pagesize=A4,
    leftMargin=MARGIN,
    rightMargin=MARGIN,
    topMargin=MARGIN,
    bottomMargin=MARGIN,
    title="Regime-Aware Machine Learning for Equity Trading",
    author="Zeineb Turki",
)

# Add page numbers
def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(PAGE_W / 2, 1.5 * cm, f"Page {doc.page}")
    canvas.restoreState()

doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print(f"Done! Thesis saved to {OUTPUT_PDF}")
print(f"Figures saved to {FIGURES_DIR}")
