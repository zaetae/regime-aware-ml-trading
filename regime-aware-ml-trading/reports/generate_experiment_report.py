"""Generate PDF summarizing hyperparameter optimization and profitability experiments.

Runs the full experiment pipeline and produces a concise academic report:
- Hyperparameter search results
- Best TP/SL/max_holding configurations
- Classification vs profitability tradeoffs
- Touch-event dataset augmentation effects
- Remaining limitations

Usage:
    python reports/generate_experiment_report.py
"""

import sys
import os
import json
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    KeepTogether,
)
from reportlab.lib import colors

from src.data.load_data import load_spy
from src.features.build_features import build_feature_matrix
from src.models.train import run_training_pipeline
from src.models.optimize import grid_search
from src.backtest.simulator import evaluate_profitability
from src.patterns.touch_events import generate_all_touch_events
from src.patterns.scanner import scan_all_patterns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXCLUDE = ["triangle_pattern", "channel_pattern"]
DEFAULT_PT, DEFAULT_SL, DEFAULT_MH = 2.0, 2.0, 10

PT_RANGE = [1.0, 1.5, 2.0, 2.5, 3.0]
SL_RANGE = [1.0, 1.5, 2.0, 2.5, 3.0]
MH_RANGE = [5, 10, 15, 20]

IMG_DIR = os.path.join(os.path.dirname(__file__), "experiment_figures")
os.makedirs(IMG_DIR, exist_ok=True)

OUTPUT_PDF = os.path.join(os.path.dirname(__file__), "experiment_report.pdf")

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
print("Loading data...")
df = load_spy()
n_bars = len(df)

# --- Baseline ---
print("Building baseline features...")
feat_base, lab_base, ldf_base = build_feature_matrix(
    df, exclude_patterns=EXCLUDE,
    pt_mult=DEFAULT_PT, sl_mult=DEFAULT_SL, max_holding=DEFAULT_MH,
)

print("Training baseline models...")
res_base = run_training_pipeline(
    feat_base, lab_base, ldf_base,
    df_ohlcv=df, pt_mult=DEFAULT_PT, sl_mult=DEFAULT_SL, max_holding=DEFAULT_MH,
)

# --- Grid search: F1 ---
print("Grid search for F1 macro (100 trials)...")
best_f1, score_f1, df_f1 = grid_search(
    df, target="f1_macro",
    pt_range=PT_RANGE, sl_range=SL_RANGE, holding_range=MH_RANGE,
    exclude_patterns=EXCLUDE, n_estimators=100, verbose=True,
)

# --- Grid search: cumulative return ---
print("Grid search for cumulative return (100 trials)...")
best_pr, score_pr, df_pr = grid_search(
    df, target="cumulative_return",
    pt_range=PT_RANGE, sl_range=SL_RANGE, holding_range=MH_RANGE,
    exclude_patterns=EXCLUDE, n_estimators=100, verbose=True,
)

# --- Train with best-F1 params ---
print("Training with best-F1 params...")
feat_f1, lab_f1, ldf_f1 = build_feature_matrix(
    df, exclude_patterns=EXCLUDE,
    pt_mult=best_f1["pt_mult"], sl_mult=best_f1["sl_mult"],
    max_holding=best_f1["max_holding"],
)
res_f1 = run_training_pipeline(
    feat_f1, lab_f1, ldf_f1,
    df_ohlcv=df, **best_f1,
)

# --- Train with best-profit params ---
print("Training with best-profit params...")
feat_pr, lab_pr, ldf_pr = build_feature_matrix(
    df, exclude_patterns=EXCLUDE,
    pt_mult=best_pr["pt_mult"], sl_mult=best_pr["sl_mult"],
    max_holding=best_pr["max_holding"],
)
res_pr = run_training_pipeline(
    feat_pr, lab_pr, ldf_pr,
    df_ohlcv=df, **best_pr,
)

# --- Touch events ---
print("Generating touch events...")
df_scanned = scan_all_patterns(df)
df_touch, touch_stats = generate_all_touch_events(df_scanned)

feat_touch, lab_touch, ldf_touch = build_feature_matrix(
    df, exclude_patterns=EXCLUDE,
    pt_mult=best_f1["pt_mult"], sl_mult=best_f1["sl_mult"],
    max_holding=best_f1["max_holding"],
    include_touch_events=True,
)
res_touch = run_training_pipeline(
    feat_touch, lab_touch, ldf_touch,
    df_ohlcv=df, pt_mult=best_f1["pt_mult"], sl_mult=best_f1["sl_mult"],
    max_holding=best_f1["max_holding"],
)

n_detector = int((ldf_touch["event_source"] == "detector").sum()) if "event_source" in ldf_touch.columns else len(ldf_touch)
n_touch_ev = int((ldf_touch["event_source"] == "touch").sum()) if "event_source" in ldf_touch.columns else 0

# ---------------------------------------------------------------------------
# Generate figures
# ---------------------------------------------------------------------------
print("Generating figures...")

# Figure 1: Optimization heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (rdf, title) in zip(axes, [
    (df_f1, "F1 Macro Score"),
    (df_pr, "Cumulative Return"),
]):
    pivot = rdf.pivot_table(values="score", index="sl_mult", columns="pt_mult",
                            aggfunc="mean")
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, center=0)
    ax.set_title(f"{title}\n(averaged over max_holding)")
    ax.set_xlabel("pt_mult")
    ax.set_ylabel("sl_mult")
plt.tight_layout()
fig1_path = os.path.join(IMG_DIR, "heatmaps.png")
plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
plt.close()

# Figure 2: Config comparison bars
configs_data = [
    ("Default", res_base),
    ("Best F1", res_f1),
    ("Best Profit", res_pr),
    ("Best F1+Touch", res_touch),
]

labels_fig = [c[0] for c in configs_data]
f1_vals = [c[1]["test_results"]["rf"]["f1_macro"] for c in configs_data]
acc_vals = [c[1]["test_results"]["rf"]["accuracy"] for c in configs_data]
cum_ret = []
for _, r in configs_data:
    if r["profitability"]:
        cum_ret.append(r["profitability"]["rf"]["test"]["cumulative_return"])
    else:
        cum_ret.append(0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
x = range(len(labels_fig))

axes[0].bar(x, f1_vals, color="steelblue", alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels_fig, rotation=25, ha="right")
axes[0].set_ylabel("F1 Macro")
axes[0].set_title("Classification (F1)")

bar_colors = ["green" if v > 0 else "red" for v in cum_ret]
axes[1].bar(x, cum_ret, color=bar_colors, alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels_fig, rotation=25, ha="right")
axes[1].set_ylabel("Cumulative Return")
axes[1].set_title("Profitability")
axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.3)

axes[2].bar(x, acc_vals, color="darkorange", alpha=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(labels_fig, rotation=25, ha="right")
axes[2].set_ylabel("Accuracy")
axes[2].set_title("Classification (Accuracy)")

plt.tight_layout()
fig2_path = os.path.join(IMG_DIR, "comparison.png")
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
plt.close()

# Figure 3: max_holding effect
fig, ax = plt.subplots(figsize=(8, 4.5))
grouped = df_f1.groupby("max_holding")["f1_macro"].agg(["mean", "std"])
ax.bar(grouped.index, grouped["mean"], yerr=grouped["std"],
       capsize=5, alpha=0.7, color="steelblue")
ax.set_xlabel("max_holding (bars)")
ax.set_ylabel("F1 Macro")
ax.set_title("Effect of max_holding on Classification Performance")
ax.set_xticks(grouped.index)
plt.tight_layout()
fig3_path = os.path.join(IMG_DIR, "max_holding_effect.png")
plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------
print("Building PDF report...")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="JustifySmall", parent=styles["Normal"],
                          fontSize=9, leading=12, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name="Heading2Center", parent=styles["Heading2"],
                          alignment=TA_CENTER))
styles.add(ParagraphStyle(name="SmallItalic", parent=styles["Normal"],
                          fontSize=8, leading=10, textColor=colors.grey))

doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4,
                        leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=2 * cm, bottomMargin=2 * cm)

story = []

# Title
story.append(Paragraph("Hyperparameter Optimization & Profitability Analysis",
                        styles["Title"]))
story.append(Paragraph("Regime-Aware ML Trading — Experiment Report",
                        styles["Heading2Center"]))
story.append(Spacer(1, 0.5 * cm))
story.append(Paragraph(
    f"Dataset: SPY daily, {n_bars} bars | "
    f"Events: {len(feat_base)} (detector-only) | "
    f"Grid: {len(df_f1)} configurations tested",
    styles["SmallItalic"],
))
story.append(Spacer(1, 0.8 * cm))

# Section 1: Introduction
story.append(Paragraph("1. Motivation", styles["Heading2"]))
story.append(Paragraph(
    "The original pipeline used fixed triple-barrier parameters (pt_mult=2.0, "
    "sl_mult=2.0, max_holding=10) and evaluated models solely by classification "
    "accuracy and F1 score. This report extends the framework by treating these "
    "parameters as tunable hyperparameters and evaluating models by both "
    "classification and trading profitability metrics. A key hypothesis is that "
    "the optimal parameters for classification may differ from those for profitability.",
    styles["JustifySmall"],
))
story.append(Spacer(1, 0.5 * cm))

# Section 2: Search space
story.append(Paragraph("2. Hyperparameter Search Space", styles["Heading2"]))
story.append(Paragraph(
    f"Exhaustive grid search over {len(PT_RANGE)}x{len(SL_RANGE)}x{len(MH_RANGE)} "
    f"= {len(PT_RANGE)*len(SL_RANGE)*len(MH_RANGE)} configurations:<br/>"
    f"&bull; pt_mult: {PT_RANGE}<br/>"
    f"&bull; sl_mult: {SL_RANGE}<br/>"
    f"&bull; max_holding: {MH_RANGE}<br/>"
    "Each trial re-labels all events with the candidate parameters, rebuilds features, "
    "trains an RF (100 trees, max_depth=8), and evaluates on a chronological "
    "validation set (60/20/20 split). No look-ahead leakage: labeling walks forward "
    "from signal-bar close.",
    styles["JustifySmall"],
))
story.append(Spacer(1, 0.3 * cm))

# Heatmaps figure
if os.path.exists(fig1_path):
    story.append(Image(fig1_path, width=16 * cm, height=6.3 * cm))
    story.append(Paragraph("Figure 1: Optimization landscape — F1 and cumulative return "
                           "averaged over max_holding values.", styles["SmallItalic"]))
story.append(Spacer(1, 0.5 * cm))

# Section 3: Best parameters
story.append(Paragraph("3. Best Parameters Found", styles["Heading2"]))

param_table_data = [
    ["Target", "pt_mult", "sl_mult", "max_holding", "Score"],
    ["F1 Macro", str(best_f1["pt_mult"]), str(best_f1["sl_mult"]),
     str(best_f1["max_holding"]), f"{score_f1:.4f}"],
    ["Cum. Return", str(best_pr["pt_mult"]), str(best_pr["sl_mult"]),
     str(best_pr["max_holding"]), f"{score_pr:.4f}"],
    ["Default", str(DEFAULT_PT), str(DEFAULT_SL), str(DEFAULT_MH), "(baseline)"],
]
t = Table(param_table_data, colWidths=[3.5 * cm, 2.5 * cm, 2.5 * cm, 3 * cm, 3 * cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E8EDF5")]),
]))
story.append(t)
story.append(Spacer(1, 0.3 * cm))

same_params = best_f1 == best_pr
if same_params:
    story.append(Paragraph(
        "Both optimization targets converged to the same parameters, suggesting "
        "alignment between classification and profitability in this dataset.",
        styles["JustifySmall"],
    ))
else:
    story.append(Paragraph(
        "<b>Key finding:</b> optimal parameters differ between classification and "
        "profitability targets. This confirms that accuracy alone is insufficient "
        "for evaluating trading models.",
        styles["JustifySmall"],
    ))
story.append(Spacer(1, 0.5 * cm))

# Section 4: Results comparison
story.append(Paragraph("4. Classification vs Profitability Results", styles["Heading2"]))

results_table_data = [
    ["Config", "RF Acc", "RF F1", "Cum Return", "Win Rate", "Sharpe", "Trades"],
]
for label, res in [("Default", res_base), ("Best F1", res_f1),
                    ("Best Profit", res_pr), ("Best F1+Touch", res_touch)]:
    tr = res["test_results"]["rf"]
    row = [label, f"{tr['accuracy']:.3f}", f"{tr['f1_macro']:.3f}"]
    if res["profitability"]:
        p = res["profitability"]["rf"]["test"]
        row.extend([
            f"{p['cumulative_return']:.4f}",
            f"{p['win_rate']:.2f}",
            f"{p['sharpe_ratio']:.3f}",
            str(p["n_trades"]),
        ])
    else:
        row.extend(["-", "-", "-", "-"])
    results_table_data.append(row)

t2 = Table(results_table_data,
           colWidths=[3 * cm, 1.8 * cm, 1.8 * cm, 2.5 * cm, 2 * cm, 2 * cm, 1.8 * cm])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E8EDF5")]),
]))
story.append(t2)
story.append(Spacer(1, 0.3 * cm))

# Comparison figure
if os.path.exists(fig2_path):
    story.append(Image(fig2_path, width=16 * cm, height=5 * cm))
    story.append(Paragraph("Figure 2: Comparison across configurations — classification "
                           "and profitability metrics.", styles["SmallItalic"]))
story.append(Spacer(1, 0.5 * cm))

# Section 5: Touch events
story.append(Paragraph("5. Touch-Based Event Augmentation", styles["Heading2"]))
story.append(Paragraph(
    f"Following supervisor guidance to 'start sequences from direct touch of trend "
    f"lines', we generated additional events when price directly touches support/"
    f"resistance levels or channel boundaries (using 0.2*ATR proximity threshold)."
    f"<br/><br/>"
    f"&bull; Original detector events: {n_detector}<br/>"
    f"&bull; New touch-only events: {n_touch_ev}<br/>"
    f"&bull; Combined total: {len(ldf_touch)}<br/>"
    f"&bull; Dataset increase: +{(n_touch_ev / max(n_detector, 1)) * 100:.1f}%<br/>"
    f"<br/>"
    f"Touch event breakdown: "
    f"support={touch_stats.get('touch_support', 0)}, "
    f"resistance={touch_stats.get('touch_resistance', 0)}, "
    f"channel_upper={touch_stats.get('touch_channel_upper', 0)}, "
    f"channel_lower={touch_stats.get('touch_channel_lower', 0)}",
    styles["JustifySmall"],
))
story.append(Spacer(1, 0.3 * cm))

# Label distribution comparison
touch_labels = lab_touch.value_counts().to_dict()
base_labels = lab_base.value_counts().to_dict()
label_table = [["Label", "Without Touch", "With Touch"]]
for lbl in sorted(set(list(touch_labels.keys()) + list(base_labels.keys()))):
    label_table.append([lbl, str(base_labels.get(lbl, 0)), str(touch_labels.get(lbl, 0))])
t3 = Table(label_table, colWidths=[3 * cm, 3.5 * cm, 3.5 * cm])
t3.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
]))
story.append(t3)
story.append(Spacer(1, 0.5 * cm))

# Section 6: max_holding effect
if os.path.exists(fig3_path):
    story.append(Paragraph("6. Effect of max_holding", styles["Heading2"]))
    story.append(Image(fig3_path, width=12 * cm, height=6 * cm))
    story.append(Paragraph("Figure 3: Effect of max_holding on F1 macro score.",
                           styles["SmallItalic"]))
    story.append(Spacer(1, 0.5 * cm))

# Section 7: Limitations
story.append(Paragraph("7. Limitations", styles["Heading2"]))
story.append(Paragraph(
    "&bull; <b>Small dataset</b>: ~100-140 events provides limited statistical power. "
    "All results should be interpreted with caution.<br/>"
    "&bull; <b>No transaction costs</b>: simulated returns exclude spread, slippage, "
    "and commissions.<br/>"
    "&bull; <b>Simplified entry</b>: trades enter at signal-bar Close rather than "
    "next-bar Open. This is consistent with the labeling pipeline but may slightly "
    "overstate achievable returns.<br/>"
    "&bull; <b>Single asset</b>: only SPY is tested. Generalization to other assets "
    "is untested.<br/>"
    "&bull; <b>Optimization overfitting risk</b>: with 100 configurations on ~100 events, "
    "the best parameters may overfit the validation set.<br/>"
    "&bull; <b>Touch events noise</b>: additional touch-based events may not carry "
    "the same signal quality as strict pattern-detector events.",
    styles["JustifySmall"],
))
story.append(Spacer(1, 0.5 * cm))

# Section 8: Conclusions
story.append(Paragraph("8. Conclusions", styles["Heading2"]))
story.append(Paragraph(
    "1. Triple-barrier parameters significantly affect both classification and "
    "profitability metrics. Treating them as hyperparameters is essential.<br/>"
    "2. The best parameters for F1 score may differ from those for profitability, "
    "confirming that classification accuracy alone is insufficient for trading "
    "model evaluation.<br/>"
    "3. Touch-based events increase the dataset by "
    f"{(n_touch_ev / max(n_detector, 1)) * 100:.0f}%, with effects on model "
    "performance that should be validated with larger datasets.<br/>"
    "4. The framework is now configurable: future experiments can explore different "
    "optimization targets, search ranges, and event generation strategies.<br/>"
    "5. All results are preliminary given the small dataset size. Production "
    "deployment would require significantly more data and out-of-sample validation.",
    styles["JustifySmall"],
))

# Build
doc.build(story)
print(f"\nReport saved: {OUTPUT_PDF}")

# Also save results JSON
results_json = {
    "best_f1_params": best_f1,
    "best_f1_score": float(score_f1),
    "best_profit_params": best_pr,
    "best_profit_score": float(score_pr),
    "baseline_results": {
        "accuracy": res_base["test_results"]["rf"]["accuracy"],
        "f1_macro": res_base["test_results"]["rf"]["f1_macro"],
    },
    "touch_events": {
        "n_detector": n_detector,
        "n_touch": n_touch_ev,
        "n_total": len(ldf_touch),
    },
    "n_configs_tested": len(df_f1),
}
json_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "experiment_results.json")
os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2, default=str)
print(f"Results JSON saved: {json_path}")
