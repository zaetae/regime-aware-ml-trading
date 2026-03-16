"""
Generate comprehensive thesis-style strategy & research PDF.
Includes: project vision, literature review, architecture UML,
use cases, current state, roadmap, and visual diagrams.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import matplotlib.patheffects as pe

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
)
from reportlab.lib import colors

from src.data.load_data import load_spy
from src.data.utils import compute_atr
from src.patterns.support_resistance import calculate_support_resistance
from src.patterns.triangles import detect_triangle_pattern
from src.patterns.channels import detect_channel
from src.patterns.multiple_tops_bottoms import detect_multiple_tops_bottoms

# ── helpers ──────────────────────────────────────────────────────────────
IMG_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMG_DIR, exist_ok=True)

DARK = "#1a1a2e"
MID = "#16213e"
ACCENT = "#0f3460"
HIGHLIGHT = "#e94560"
LIGHT_BG = "#f8f8f8"

# ── load data & run detectors ────────────────────────────────────────────
df = load_spy()
n_bars = len(df)
date_start = df.index[0].strftime("%Y-%m-%d")
date_end = df.index[-1].strftime("%Y-%m-%d")

sr_df = calculate_support_resistance(df)
sr = sr_df["near_support"] | sr_df["near_resistance"]
tri_df = detect_triangle_pattern(df)
tri = tri_df["triangle_pattern"].notna()
ch_df = detect_channel(df)
ch = ch_df["channel_pattern"].notna()
mtb_df = detect_multiple_tops_bottoms(df)
mtb = mtb_df["multiple_top_bottom_pattern"].notna()
combined = sr | tri | ch | mtb

# =====================================================================
#  DIAGRAM 1 — System Architecture (UML Component Diagram)
# =====================================================================
def draw_architecture():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.4", linewidth=1.5)

    def draw_box(x, y, w, h, label, color, sub=None, fontsize=9):
        rect = FancyBboxPatch((x, y), w, h, **{**box_style, "edgecolor": color, "facecolor": color + "18"})
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.3, label, ha="center", va="top",
                fontsize=fontsize, fontweight="bold", color=color)
        if sub:
            for i, s in enumerate(sub):
                ax.text(x + 0.25, y + h - 0.7 - i * 0.32, f"- {s}",
                        fontsize=7, color="#444444", va="top")

    def arrow(x1, y1, x2, y2, label="", color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.1, my + 0.12, label, fontsize=6.5, color=color,
                    fontstyle="italic", ha="center")

    # Title
    ax.text(7, 8.7, "System Architecture — Regime-Aware ML Trading",
            ha="center", fontsize=13, fontweight="bold", color=DARK)

    # Layer 1: Data
    draw_box(0.3, 7, 3, 1.3, "Data Layer", ACCENT,
             ["yfinance download", "CSV loader", "ATR utility"])

    # Layer 2: Feature Engineering
    draw_box(0.3, 5.2, 3, 1.4, "Feature Engineering", "#00695c",
             ["Frac. differentiation", "Momentum / Vol / Volume", "CUSUM event filter"])

    # Layer 3: Pattern Detection
    draw_box(4, 5.2, 3.2, 1.4, "Pattern Detection", "#bf360c",
             ["Support / Resistance", "Triangles (breakout)", "Channels / Multi T-B"])

    # Layer 4: Regime Detection
    draw_box(7.8, 5.2, 3.2, 1.4, "Regime Detection", "#4a148c",
             ["HMM (2-3 states)", "Jump models", "VIX thresholds"])

    # Layer 5: Signal Model
    draw_box(2, 3.2, 3.5, 1.4, "Signal Model", "#1565c0",
             ["XGBoost / LightGBM", "Triple-barrier labels", "Regime-conditional"])

    # Layer 6: Meta-labeling / Sizing
    draw_box(6.2, 3.2, 3.5, 1.4, "Meta-Labeling & Sizing", "#c62828",
             ["Bet confidence model", "Kelly / fractional Kelly", "Regime-scaled vol target"])

    # Layer 7: Risk Management
    draw_box(0.3, 1.2, 4, 1.4, "Risk Management", "#e65100",
             ["ATR stops (regime-scaled)", "Drawdown brakes", "VaR / CVaR limits"])

    # Layer 8: Backtest & Evaluation
    draw_box(5, 1.2, 4, 1.4, "Backtest & Evaluation", "#2e7d32",
             ["Walk-forward validation", "CPCV hyperparameter tuning", "Sharpe / MDD / Calmar"])

    # Layer 9: Reporting
    draw_box(9.8, 1.2, 3.8, 1.4, "Reporting", "#37474f",
             ["PDF reports (reportlab)", "Per-regime decomposition", "Overfitting diagnostics"])

    # Arrows
    arrow(1.8, 7, 1.8, 6.6)
    arrow(1.8, 5.2, 3, 4.6, "features")
    arrow(5.6, 5.2, 4.5, 4.6, "patterns")
    arrow(9.4, 5.2, 8.5, 4.6, "regime")
    arrow(9.4, 5.5, 5, 4.2, "regime labels")
    arrow(4.5, 3.2, 3.5, 2.6, "signals")
    arrow(8.5, 3.2, 7.5, 2.6, "sizes")
    arrow(3, 1.2, 4, 0.6)  # risk arrow stub down
    arrow(7, 1.2, 8, 0.6)  # eval arrow stub down

    # Status badges
    def badge(x, y, text, color):
        ax.text(x, y, text, fontsize=6, color="white", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="none"))

    badge(3, 8.05, "DONE", "#2e7d32")
    badge(6.8, 6.35, "DONE", "#2e7d32")
    badge(2.8, 6.35, "NEXT", "#ff8f00")
    badge(10.6, 6.35, "NEXT", "#ff8f00")
    badge(4.8, 4.35, "PLANNED", "#757575")
    badge(8.8, 4.35, "PLANNED", "#757575")
    badge(3.3, 2.35, "PLANNED", "#757575")
    badge(8, 2.35, "PLANNED", "#757575")
    badge(12.5, 2.35, "PLANNED", "#757575")

    fig.tight_layout()
    path = os.path.join(IMG_DIR, "architecture.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# =====================================================================
#  DIAGRAM 2 — Pipeline Flowchart
# =====================================================================
def draw_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    steps = [
        ("OHLCV\nData", "#546e7a"),
        ("Feature\nEngineering", "#00695c"),
        ("Pattern\nDetection", "#bf360c"),
        ("Regime\nDetection", "#4a148c"),
        ("Signal\nModel", "#1565c0"),
        ("Risk &\nSizing", "#c62828"),
        ("Backtest\n& Report", "#2e7d32"),
    ]
    x_start = 0.4
    box_w = 1.3
    gap = 0.25
    y_center = 2.25

    for i, (label, color) in enumerate(steps):
        x = x_start + i * (box_w + gap)
        rect = FancyBboxPatch((x, y_center - 0.65), box_w, 1.3,
                              boxstyle="round,pad=0.15", edgecolor=color,
                              facecolor=color + "22", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y_center, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=color)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + box_w + gap - 0.05, y_center),
                        xytext=(x + box_w + 0.05, y_center),
                        arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.5))

    # Phase labels below
    phases = ["Phase 0", "Phase 1", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5"]
    statuses = ["Done", "Done", "Done", "Next", "Planned", "Planned", "Planned"]
    status_colors = {"Done": "#2e7d32", "Next": "#ff8f00", "Planned": "#757575"}
    for i, (phase, status) in enumerate(zip(phases, statuses)):
        x = x_start + i * (box_w + gap) + box_w / 2
        ax.text(x, y_center - 0.95, phase, ha="center", fontsize=7, color="#666")
        ax.text(x, y_center - 1.2, status, ha="center", fontsize=7,
                fontweight="bold", color=status_colors[status])

    ax.set_title("End-to-End Pipeline Phases", fontsize=12, fontweight="bold",
                 color=DARK, pad=10)
    fig.tight_layout()
    path = os.path.join(IMG_DIR, "pipeline.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# =====================================================================
#  DIAGRAM 3 — Use Case Diagram
# =====================================================================
def draw_use_cases():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.set_title("Use Case Diagram — Regime-Aware ML Trading System",
                 fontsize=13, fontweight="bold", color=DARK, pad=10)

    # System boundary
    rect = mpatches.FancyBboxPatch((2.5, 0.4), 7, 6.1,
                                    boxstyle="round,pad=0.3",
                                    edgecolor=ACCENT, facecolor="#f0f4ff",
                                    linewidth=2, linestyle="--")
    ax.add_patch(rect)
    ax.text(6, 6.25, "Regime-Aware ML Trading System", ha="center",
            fontsize=11, fontweight="bold", color=ACCENT)

    # Actors
    def draw_actor(x, y, label):
        ax.plot(x, y + 0.35, "o", color=DARK, markersize=8)
        ax.plot([x, x], [y + 0.25, y - 0.05], color=DARK, lw=1.5)
        ax.plot([x - 0.15, x + 0.15], [y + 0.15, y + 0.15], color=DARK, lw=1.5)
        ax.plot([x, x - 0.15], [y - 0.05, y - 0.25], color=DARK, lw=1.5)
        ax.plot([x, x + 0.15], [y - 0.05, y - 0.25], color=DARK, lw=1.5)
        ax.text(x, y - 0.45, label, ha="center", fontsize=8, fontweight="bold", color=DARK)

    draw_actor(1.2, 5.2, "Quant\nResearcher")
    draw_actor(1.2, 2.8, "Portfolio\nManager")
    draw_actor(11, 4.5, "Market\nData API")
    draw_actor(11, 2.5, "Scheduler\n(Cron)")

    # Use case ellipses
    def use_case(x, y, label, color=ACCENT):
        el = mpatches.Ellipse((x, y), 2.8, 0.7, edgecolor=color,
                              facecolor=color + "15", linewidth=1.3)
        ax.add_patch(el)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, color=color, fontweight="bold")

    cases = [
        (4.5, 5.5, "Download & Load Data"),
        (7.5, 5.5, "Compute Features"),
        (4.5, 4.5, "Detect Patterns"),
        (7.5, 4.5, "Detect Regimes"),
        (4.5, 3.5, "Train Signal Model"),
        (7.5, 3.5, "Backtest Strategy"),
        (4.5, 2.3, "Generate Reports"),
        (7.5, 2.3, "Manage Risk & Size"),
        (6, 1.2, "Walk-Forward Validate"),
    ]
    for x, y, label in cases:
        use_case(x, y, label)

    # Connections (actor to use case)
    def connect(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color="#999", lw=0.8)

    # Researcher connections
    for x, y, _ in cases[:7]:
        connect(1.5, 5.2 if y > 3 else 2.8, x - 1.2, y)

    # PM connections
    connect(1.5, 2.8, 3.2, 2.3)
    connect(1.5, 2.8, 6.2, 2.3)
    connect(1.5, 2.8, 6.2, 1.2)

    # External actors
    connect(10.7, 4.5, 5.9, 5.5)
    connect(10.7, 2.5, 6.2, 1.2)

    fig.tight_layout()
    path = os.path.join(IMG_DIR, "use_cases.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# =====================================================================
#  DIAGRAM 4 — Regime concept illustration
# =====================================================================
def draw_regime_concept():
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), gridspec_kw={"height_ratios": [3, 1]})

    close = df["Close"].values
    dates = df.index

    # Simulated regime labels (volatility-based for illustration)
    vol = df["Close"].pct_change().rolling(50).std() * np.sqrt(252)
    regime = pd.Series("Bull", index=df.index)
    regime[vol > vol.quantile(0.7)] = "Bear"
    regime[(vol > vol.quantile(0.3)) & (vol <= vol.quantile(0.7))] = "Sideways"

    cmap = {"Bull": "#4CAF50", "Bear": "#F44336", "Sideways": "#FF9800"}

    ax = axes[0]
    ax.plot(dates, close, color="#333", lw=0.7)
    for r, color in cmap.items():
        mask = regime == r
        segments = mask.astype(int).diff().fillna(0)
        starts = dates[segments == 1]
        ends = dates[segments == -1]
        if mask.iloc[0]:
            starts = starts.insert(0, dates[0])
        if mask.iloc[-1]:
            ends = ends.append(pd.DatetimeIndex([dates[-1]]))
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, alpha=0.12, color=color)

    patches = [mpatches.Patch(color=c, alpha=0.3, label=r) for r, c in cmap.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    ax.set_title("SPY Price with Illustrative Market Regimes (Volatility-Based)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.fill_between(dates, vol * 100, alpha=0.5, color="#7986CB")
    ax2.axhline(vol.quantile(0.3) * 100, color="#4CAF50", ls="--", lw=0.8, label="30th pctl")
    ax2.axhline(vol.quantile(0.7) * 100, color="#F44336", ls="--", lw=0.8, label="70th pctl")
    ax2.set_ylabel("Ann. Vol (%)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(IMG_DIR, "regime_concept.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# =====================================================================
#  DIAGRAM 5 — Current event distribution
# =====================================================================
def draw_event_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pie chart
    labels = ["S/R", "Triangles", "Channels", "Multi T/B"]
    counts = [sr.sum(), tri.sum(), ch.sum(), mtb.sum()]
    clrs = ["#2196F3", "#E91E63", "#FF9800", "#4CAF50"]
    axes[0].pie(counts, labels=labels, colors=clrs, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 9})
    axes[0].set_title("Event Distribution by Detector", fontsize=11, fontweight="bold")

    # Monthly event rate
    monthly = combined.resample("ME").mean() * 100
    axes[1].bar(monthly.index, monthly.values, width=25, color="#5C6BC0", alpha=0.8)
    axes[1].set_title("Monthly Combined Event Rate (%)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Event Rate (%)")
    axes[1].axhline(10, color="#2e7d32", ls="--", lw=0.8, label="Target floor (10%)")
    axes[1].axhline(30, color="#c62828", ls="--", lw=0.8, label="Target ceiling (30%)")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(IMG_DIR, "event_dist.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# =====================================================================
#  DIAGRAM 6 — HMM State Diagram
# =====================================================================
def draw_hmm_diagram():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.set_title("Hidden Markov Model — State Transition Diagram",
                 fontsize=12, fontweight="bold", color=DARK, pad=10)

    states = [
        (2, 3.5, "Bull\n(Low Vol)", "#4CAF50"),
        (6, 3.5, "Bear\n(High Vol)", "#F44336"),
        (4, 1.2, "Sideways\n(Medium Vol)", "#FF9800"),
    ]
    for x, y, label, color in states:
        circle = plt.Circle((x, y), 0.8, edgecolor=color, facecolor=color + "22", lw=2.5)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    # Transition arrows
    def curved_arrow(x1, y1, x2, y2, label, offset=0.3):
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 + offset
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2,
                                    connectionstyle=f"arc3,rad={offset * 0.5}"))
        ax.text(mid_x, mid_y, label, ha="center", fontsize=7, color="#555", fontstyle="italic")

    curved_arrow(2.7, 3.8, 5.3, 3.8, "P=0.05", offset=0.5)
    curved_arrow(5.3, 3.2, 2.7, 3.2, "P=0.10", offset=-0.5)
    curved_arrow(2.3, 2.75, 3.5, 1.8, "P=0.03", offset=0.3)
    curved_arrow(3.5, 1.5, 2.2, 2.7, "P=0.15", offset=-0.3)
    curved_arrow(5.7, 2.75, 4.5, 1.8, "P=0.08", offset=-0.3)
    curved_arrow(4.5, 1.5, 5.8, 2.7, "P=0.07", offset=0.3)

    # Self-loops (text only)
    ax.text(1, 4.2, "P=0.92", fontsize=7, color="#4CAF50", fontstyle="italic")
    ax.text(6.5, 4.2, "P=0.82", fontsize=7, color="#F44336", fontstyle="italic")
    ax.text(4, 0.3, "P=0.78", fontsize=7, color="#FF9800", fontstyle="italic")

    fig.tight_layout()
    path = os.path.join(IMG_DIR, "hmm_states.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# =====================================================================
#  DIAGRAM 7 — Triple Barrier illustration
# =====================================================================
def draw_triple_barrier():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    np.random.seed(42)
    n = 60
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    entry = price[10]
    entry_idx = 10
    tp = entry + 3
    sl = entry - 2
    exp = 45

    ax.plot(range(n), price, color="#333", lw=1.2)
    ax.axhline(tp, color="#4CAF50", ls="--", lw=1.5, label=f"Profit Take ({tp:.0f})")
    ax.axhline(sl, color="#F44336", ls="--", lw=1.5, label=f"Stop Loss ({sl:.0f})")
    ax.axvline(exp, color="#FF9800", ls="--", lw=1.5, label=f"Expiration (bar {exp})")
    ax.axvline(entry_idx, color="#2196F3", ls=":", lw=1.2, label=f"Entry (bar {entry_idx})")

    ax.fill_between(range(entry_idx, exp + 1), sl, tp, alpha=0.06, color="#2196F3")
    ax.scatter([entry_idx], [entry], color="#2196F3", s=80, zorder=5)

    # Mark where TP is hit
    hit_idx = None
    for i in range(entry_idx, exp + 1):
        if price[i] >= tp:
            hit_idx = i
            break
        if price[i] <= sl:
            hit_idx = i
            break
    if hit_idx:
        ax.scatter([hit_idx], [price[hit_idx]], color="#4CAF50" if price[hit_idx] >= tp else "#F44336",
                   s=120, zorder=5, marker="*", edgecolors="black")

    ax.set_title("Triple Barrier Method — Label Generation", fontsize=11, fontweight="bold")
    ax.set_xlabel("Bar Index")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(IMG_DIR, "triple_barrier.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# =====================================================================
#  DIAGRAM 8 — SPY returns distribution by simulated regime
# =====================================================================
def draw_regime_returns():
    fig, ax = plt.subplots(figsize=(10, 4))
    returns = df["Close"].pct_change().dropna()
    vol = returns.rolling(50).std() * np.sqrt(252)
    vol = vol.reindex(returns.index)

    bull = returns[vol <= vol.quantile(0.3)]
    bear = returns[vol > vol.quantile(0.7)]
    side = returns[(vol > vol.quantile(0.3)) & (vol <= vol.quantile(0.7))]

    bins = np.linspace(-0.06, 0.06, 80)
    ax.hist(bull, bins=bins, alpha=0.5, color="#4CAF50", label=f"Bull (n={len(bull)})", density=True)
    ax.hist(bear, bins=bins, alpha=0.5, color="#F44336", label=f"Bear (n={len(bear)})", density=True)
    ax.hist(side, bins=bins, alpha=0.3, color="#FF9800", label=f"Sideways (n={len(side)})", density=True)

    ax.set_title("Daily Return Distributions by Regime (Volatility-Based Proxy)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    path = os.path.join(IMG_DIR, "regime_returns.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Generate all diagrams ────────────────────────────────────────────────
print("Generating diagrams...")
arch_path = draw_architecture()
pipe_path = draw_pipeline()
uc_path = draw_use_cases()
regime_path = draw_regime_concept()
event_path = draw_event_distribution()
hmm_path = draw_hmm_diagram()
tb_path = draw_triple_barrier()
rr_path = draw_regime_returns()
print("Diagrams done.")

# =====================================================================
#  PDF GENERATION
# =====================================================================
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "strategy_research_report.pdf")

doc = SimpleDocTemplate(OUTPUT_PATH, pagesize=A4,
                        topMargin=2.2*cm, bottomMargin=2.2*cm,
                        leftMargin=2.2*cm, rightMargin=2.2*cm)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("TitlePage", parent=styles["Title"], fontSize=26, leading=32, spaceAfter=6, alignment=TA_CENTER))
styles.add(ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=14, leading=18, alignment=TA_CENTER, textColor=HexColor("#555"), spaceAfter=4))
styles.add(ParagraphStyle("SH", parent=styles["Heading1"], fontSize=16, leading=20, spaceBefore=22, spaceAfter=10, textColor=HexColor(DARK)))
styles.add(ParagraphStyle("SS", parent=styles["Heading2"], fontSize=13, leading=16, spaceBefore=14, spaceAfter=8, textColor=HexColor(MID)))
styles.add(ParagraphStyle("SSS", parent=styles["Heading3"], fontSize=11, leading=14, spaceBefore=10, spaceAfter=6, textColor=HexColor(ACCENT)))
styles.add(ParagraphStyle("B", parent=styles["Normal"], fontSize=10.5, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=7))
styles.add(ParagraphStyle("BL", parent=styles["Normal"], fontSize=10.5, leading=14.5, leftIndent=24, bulletIndent=12, spaceAfter=4, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle("CodeCustom", parent=styles["Code"], fontSize=8, leading=10, leftIndent=12, backColor=HexColor("#f4f4f4"), spaceAfter=8, spaceBefore=4))
styles.add(ParagraphStyle("Ref", parent=styles["Normal"], fontSize=9, leading=12, leftIndent=18, spaceAfter=3, textColor=HexColor("#333")))

story = []

def SH(t): return Paragraph(t, styles["SH"])
def SS(t): return Paragraph(t, styles["SS"])
def SSS(t): return Paragraph(t, styles["SSS"])
def B(t): return Paragraph(t, styles["B"])
def BL(t): return Paragraph(t, styles["BL"], bulletText="\u2022")
def Ref(t): return Paragraph(t, styles["Ref"])
def Img(p, w=16, h=None):
    if h is None:
        # estimate
        h = w * 0.55
    return Image(p, width=w*cm, height=h*cm)

def make_table(data, widths, header_color=HexColor(DARK)):
    t = Table(data, colWidths=[w*cm for w in widths])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor(LIGHT_BG), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    t.setStyle(TableStyle(style))
    return t

# ═══════════════════════════════════════════════════════════════════════
#  TITLE PAGE
# ═══════════════════════════════════════════════════════════════════════
story.append(Spacer(1, 4*cm))
story.append(Paragraph("Regime-Aware Machine Learning<br/>Trading System", styles["TitlePage"]))
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("Project Strategy, Research Review &amp; Technical Roadmap", styles["Subtitle"]))
story.append(Spacer(1, 1.5*cm))
story.append(Paragraph(f"Dataset: SPY &mdash; {date_start} to {date_end} ({n_bars:,} trading days)", styles["Subtitle"]))
story.append(Paragraph("March 2026", styles["Subtitle"]))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════
#  TABLE OF CONTENTS
# ═══════════════════════════════════════════════════════════════════════
story.append(SH("Table of Contents"))
toc = [
    "1. Executive Summary &amp; Project Purpose",
    "2. Research Question &amp; Hypothesis",
    "3. Literature Review",
    "    3.1 Market Regime Detection",
    "    3.2 Technical Pattern Detection",
    "    3.3 Machine Learning for Trading",
    "    3.4 Backtesting &amp; Validation",
    "    3.5 Risk Management",
    "    3.6 Feature Engineering (de Prado)",
    "4. System Architecture",
    "5. Use Cases",
    "6. Current State &amp; Results",
    "7. Roadmap &amp; Implementation Plan",
    "8. Recommendations &amp; Suggestions",
    "9. References",
]
for item in toc:
    story.append(B(item))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════
#  1. EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════
story.append(SH("1. Executive Summary &amp; Project Purpose"))
story.append(B(
    "This project develops a <b>regime-aware machine learning trading system</b> that "
    "combines three pillars: (1) algorithmic technical pattern detection, (2) Hidden Markov "
    "Model-based market regime classification, and (3) supervised ML models that learn "
    "regime-conditional trading signals. The core insight is that financial markets are "
    "<b>non-stationary</b> &mdash; a strategy calibrated to bull-market dynamics will "
    "systematically fail in bear markets. By explicitly modelling regime switches, the "
    "system adapts its signal generation, position sizing, and risk limits to the current "
    "market environment."
))
story.append(B(
    "The system is being developed in phases. <b>Phase 0</b> (data pipeline) and "
    "<b>Phase 1</b> (pattern detection with ATR-based adaptive thresholds) are complete. "
    "The pattern detectors have been tuned to a combined event rate of <b>27.3%</b>, within "
    "the 10&ndash;30% target band. The next phases involve feature engineering, HMM regime "
    "detection, ML signal modelling, and walk-forward backtesting."
))
story.append(SS("Purpose"))
purpose = [
    "Build a systematic, evidence-based trading framework that outperforms regime-agnostic baselines.",
    "Investigate whether HMM-based regime detection provides statistically significant alpha when "
    "combined with technical pattern signals.",
    "Create a reusable, modular Python codebase suitable for academic research and practical deployment.",
    "Document all methodology with thesis-grade rigour for reproducibility.",
]
for p in purpose:
    story.append(BL(p))

# ═══════════════════════════════════════════════════════════════════════
#  2. RESEARCH QUESTION
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(SH("2. Research Question &amp; Hypothesis"))
story.append(B(
    "<b>Research Question:</b> Does combining HMM-based market regime detection with "
    "event-driven technical signal selection improve out-of-sample risk-adjusted trading "
    "performance on U.S. equity markets?"
))
story.append(B("<b>Hypothesis H1:</b> A regime-aware signal model (XGBoost conditioned on "
    "HMM regime labels + pattern features) achieves a higher out-of-sample Sharpe ratio "
    "than the same model without regime labels."))
story.append(B("<b>Hypothesis H2:</b> Regime-conditional position sizing (reducing exposure "
    "in bear regimes) reduces maximum drawdown by at least 20% compared to fixed sizing, "
    "without proportionally reducing returns."))
story.append(B("<b>Null Hypothesis H0:</b> Regime labels carry no incremental predictive "
    "information beyond what is already captured by volatility and momentum features."))
story.append(SS("Evaluation Framework"))
eval_data = [
    ["Metric", "Target", "Benchmark"],
    ["Sharpe Ratio (OOS)", "> 1.0", "Buy & Hold SPY (~0.7)"],
    ["Max Drawdown", "< 20%", "SPY MDD (~34%)"],
    ["Hit Rate", "> 52%", "Random (50%)"],
    ["Calmar Ratio", "> 0.5", "SPY (~0.3)"],
    ["H1 Ablation", "Regime model > No-regime", "Same features, no HMM"],
]
story.append(make_table(eval_data, [4, 4, 5.5]))

# ═══════════════════════════════════════════════════════════════════════
#  3. LITERATURE REVIEW
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(SH("3. Literature Review"))

# 3.1 Regime Detection
story.append(SS("3.1 Market Regime Detection"))
story.append(B(
    "Market regimes are persistent, qualitatively distinct states of market behaviour "
    "characterised by differing return distributions, volatility levels, and correlation "
    "structures. The foundational work is <b>Hamilton (1989)</b>, which introduced "
    "Markov-switching autoregressive models for business cycle analysis. In finance, the "
    "canonical setup uses a Gaussian Hidden Markov Model (HMM) with 2&ndash;3 hidden states "
    "trained on daily log returns and realised volatility."
))
story.append(Img(hmm_path, w=11, h=7))
story.append(Spacer(1, 0.3*cm))
story.append(B(
    "The diagram above illustrates a 3-state HMM with typical transition probabilities. "
    "Bull regimes are highly persistent (P=0.92), bear regimes less so (P=0.82), and "
    "sideways regimes serve as transitional states. The Viterbi algorithm decodes the most "
    "likely state sequence given observed returns."
))
story.append(SSS("Alternative Approaches"))
alt_approaches = [
    "<b>Statistical Jump Models</b> (Nystrup, Kolm &amp; Lindstrom, 2020): temporal clustering "
    "with a jump penalty discouraging frequent regime switches. More robust than HMMs against "
    "poor EM initialisation and Gaussian emission misspecification.",
    "<b>VIX Threshold Models:</b> Simple rule-based classification (VIX&lt;15 = bull, 15&ndash;25 = "
    "normal, &gt;25 = bear). Transparent and lag-free but cannot capture latent structure.",
    "<b>Hybrid Ensemble-HMM</b> (Gupta et al., 2025): combines HMM with Random Forest and "
    "Gradient Boosting via a voting classifier. Achieves state-of-the-art regime detection "
    "accuracy by leveraging both generative and discriminative models.",
]
for a in alt_approaches:
    story.append(BL(a))

story.append(Img(regime_path, w=16, h=7.5))
story.append(B(
    "The chart above shows SPY price with a volatility-based regime proxy. In the final system, "
    "an HMM will replace this heuristic with probabilistic regime labels."
))
story.append(Img(rr_path, w=14, h=5.5))
story.append(B(
    "Return distributions differ substantially across regimes. Bear regimes exhibit wider tails "
    "and negative skew, validating the need for regime-conditional risk management."
))

# 3.2 Technical Pattern Detection
story.append(PageBreak())
story.append(SS("3.2 Technical Pattern Detection in Quantitative Finance"))
story.append(B(
    "The algorithmic detection of chart patterns was formalised by <b>Lo, Mamaysky &amp; Wang "
    "(2000)</b>, who used nonparametric kernel regression to identify head-and-shoulders, "
    "triangles, and channels across all NYSE/AMEX/Nasdaq stocks (1962&ndash;1996). They found "
    "that several patterns provide incremental information beyond random-walk null models. "
    "Earlier, <b>Brock, Lakonishok &amp; LeBaron (1992)</b> showed that simple moving-average "
    "and trading-range-breakout rules produced returns inconsistent with standard null models "
    "on the DJIA (1897&ndash;1986)."
))
story.append(SSS("Our Implementation"))
our_impl = [
    "<b>Support/Resistance:</b> Rolling max(High)/min(Low) with ATR-based proximity bands (0.3&times;ATR). "
    "Adapts to volatility; 50-bar window captures meaningful levels.",
    "<b>Triangles:</b> Linear regression on highs/lows to detect converging trendlines. "
    "Requires 3% compression and fires only on the breakout bar (0.3&times;ATR beyond recent range).",
    "<b>Channels:</b> Parallel trendlines validated with 5 filters: slope magnitude, parallelism "
    "(&lt;15%), same direction, width (1&ndash;6&times;ATR), and minimum 3 touches per band.",
    "<b>Multiple Tops/Bottoms:</b> Rolling extremes with 3-bar close-trend confirmation via polyfit.",
]
for item in our_impl:
    story.append(BL(item))

story.append(SSS("Criticisms &amp; Mitigations"))
story.append(B(
    "<b>Data-snooping bias:</b> With many patterns and parameters, some will appear profitable "
    "by chance. Mitigation: walk-forward validation and the Deflated Sharpe Ratio "
    "(Bailey et al., 2017). <b>Subjectivity:</b> Even algorithmic implementations have parameter "
    "degrees of freedom. Mitigation: use pattern signals as ML features rather than standalone "
    "rules, letting the model learn their value. <b>Diminishing edge:</b> Widely known patterns "
    "may lose profitability over time (Adaptive Markets Hypothesis, Lo 2004). Mitigation: "
    "regime conditioning adds a layer of differentiation."
))

# 3.3 ML Models
story.append(PageBreak())
story.append(SS("3.3 Machine Learning Models for Trading"))
story.append(B(
    "The model taxonomy for financial prediction spans traditional ML and deep learning. "
    "For tabular financial features (our case), gradient-boosted trees consistently outperform "
    "neural networks. The recommended architecture:"
))
model_data = [
    ["Model", "Strengths", "Best For"],
    ["XGBoost / LightGBM", "State-of-art tabular perf.\nHandles missing data, fast", "Signal classification\nReturn prediction"],
    ["Random Forest", "Robust to overfitting\nBuilt-in feature importance", "Baseline model\nFeature selection"],
    ["LSTM", "Captures sequential dependencies\nHandles variable sequences", "Volatility forecasting\nMedium-horizon returns"],
    ["Stacked Ensemble", "Combines best of each\nXGBoost as meta-learner", "Production system\nMaximum accuracy"],
]
story.append(make_table(model_data, [3, 5.5, 5]))
story.append(Spacer(1, 0.3*cm))
story.append(B(
    "For our system, we recommend XGBoost as the primary signal model with a Random Forest "
    "baseline for ablation studies. Pattern signals, regime labels, and momentum/volatility "
    "features serve as inputs. Labels are generated via the Triple Barrier Method."
))

# 3.4 Backtesting
story.append(SS("3.4 Backtesting &amp; Validation"))
story.append(B(
    "<b>Walk-Forward Validation (WFO)</b> is the gold standard for temporal data. The dataset "
    "is divided into sequential rolling windows: a training window (e.g., 3 years) followed by "
    "a test window (e.g., 6 months). The process &lsquo;walks forward&rsquo; through time, retraining on "
    "each new window. This prevents look-ahead bias."
))
story.append(B(
    "<b>Combinatorial Purged Cross-Validation (CPCV)</b>, introduced by de Prado (2018), "
    "produces a distribution of performance metrics rather than a single backtest. Purging "
    "removes training observations that overlap temporally with test labels. Embargo adds "
    "a buffer period. This enables rigorous overfitting assessment."
))
bias_data = [
    ["Bias", "Description", "Mitigation"],
    ["Look-ahead", "Using future data in features", "Strict temporal ordering"],
    ["Survivorship", "Testing only on surviving assets", "Point-in-time databases"],
    ["Data-snooping", "Reporting best of many tests", "Deflated Sharpe Ratio"],
    ["Overfitting", "Over-optimising to historical data", "WFO + CPCV + regularisation"],
    ["Unrealistic fills", "Ignoring slippage/impact", "Commission + slippage models"],
]
story.append(make_table(bias_data, [3, 5, 5.5]))

# 3.5 Risk Management
story.append(PageBreak())
story.append(SS("3.5 Risk Management in Regime-Aware Systems"))
story.append(B(
    "The key innovation is <b>dynamic sizing conditional on the detected regime</b>. "
    "The Kelly Criterion defines optimal bet size as f* = edge/odds, but in practice "
    "fractional Kelly (half or less) is standard because edge estimates are uncertain. "
    "Thorp (2006) showed that betting double Kelly eliminates 100% of the gain."
))
risk_table = [
    ["Regime", "Strategy Emphasis", "Position Sizing", "Stops"],
    ["Bull / Low Vol", "Trend-following, momentum", "Full target vol", "Wider (1.5x ATR)"],
    ["Bear / High Vol", "Defensive, hedged", "Half target vol", "Tight (0.75x ATR)"],
    ["Sideways", "Mean-reversion, range-bound", "Reduced size", "Medium (1x ATR)"],
]
story.append(make_table(risk_table, [3, 4, 3.5, 3]))
story.append(Spacer(1, 0.3*cm))
story.append(B(
    "A <b>drawdown brake</b> mechanism progressively reduces position sizes at predefined "
    "drawdown thresholds (-5%, -10%, -15%) and halts all new positions at -20%. "
    "Full sizing is restored only after a confirmed 50% equity recovery."
))

# 3.6 Feature Engineering
story.append(SS("3.6 Feature Engineering (de Prado)"))
story.append(B(
    "<b>Advances in Financial Machine Learning</b> (de Prado, 2018) introduces several "
    "paradigm-shifting concepts for financial feature engineering:"
))
fe_concepts = [
    "<b>Fractional Differentiation:</b> Apply a non-integer differencing operator d (typically "
    "~0.2) to achieve stationarity while preserving >90% memory of the original series. "
    "Standard returns (d=1) destroy all memory.",
    "<b>Triple Barrier Method:</b> Labels each trade entry with +1 (profit-take hit), -1 "
    "(stop-loss hit), or 0 (expiration). Encodes risk-reward dynamics better than simple "
    "return signs.",
    "<b>Meta-Labeling:</b> Separates side prediction (long/short) from sizing (confidence). "
    "A secondary model predicts bet size given the primary model&rsquo;s signal.",
    "<b>CUSUM Filter:</b> Event-driven sampling that triggers only when cumulative returns "
    "deviate from zero by a threshold, improving signal-to-noise ratio.",
    "<b>Feature Importance:</b> MDI, MDA, and SFI methods for assessing feature relevance "
    "with proper handling of temporal dependencies.",
]
for c in fe_concepts:
    story.append(BL(c))
story.append(Spacer(1, 0.3*cm))
story.append(Img(tb_path, w=12, h=6.5))
story.append(B(
    "The Triple Barrier illustration above shows an entry at bar 10 with three boundaries: "
    "profit-take (green), stop-loss (red), and expiration (orange). The first barrier touched "
    "determines the label."
))

feat_table = [
    ["Category", "Specific Features", "Source"],
    ["Returns", "Frac. diff. prices (d~0.2), log returns 1d/5d/21d/63d", "de Prado Ch. 5"],
    ["Volatility", "ATR, realised vol, Garman-Klass, Yang-Zhang", "Standard + utils.py"],
    ["Momentum", "RSI, MACD, MA distances, rate-of-change", "Technical analysis"],
    ["Volume", "Relative volume, OBV, VWAP deviation", "Microstructure"],
    ["Patterns", "near_support, near_resistance, triangle, channel, MTB", "scanner.py"],
    ["Regime", "HMM state label, transition probability, time-in-regime", "Regime model"],
]
story.append(make_table(feat_table, [2.5, 6.5, 4.5]))

# ═══════════════════════════════════════════════════════════════════════
#  4. SYSTEM ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(SH("4. System Architecture"))
story.append(B(
    "The system follows a layered pipeline architecture where each module transforms data "
    "and passes enriched outputs downstream. The diagram below shows all components, their "
    "dependencies, and current implementation status."
))
story.append(Img(arch_path, w=16, h=10.3))
story.append(Spacer(1, 0.3*cm))
story.append(SS("4.1 Pipeline Flow"))
story.append(Img(pipe_path, w=16, h=6))
story.append(B(
    "The pipeline proceeds in 6 phases. Phases 0&ndash;1 (Data + Patterns) are complete. "
    "Phase 2 (Feature Engineering + Regime Detection) is the immediate next step. "
    "The remaining phases build on each other sequentially."
))

story.append(SS("4.2 Module Responsibilities"))
module_data = [
    ["Module", "Directory", "Key Files", "Status"],
    ["Data Layer", "src/data/", "download_data.py, load_data.py, utils.py", "Complete"],
    ["Pattern Detection", "src/patterns/", "scanner.py, 4 detectors, evaluate_rates.py", "Complete"],
    ["Feature Engineering", "src/features/", "(placeholder)", "Next"],
    ["Regime Detection", "src/regimes/", "(placeholder)", "Next"],
    ["Signal Model", "src/models/", "(placeholder)", "Planned"],
    ["Labeling", "src/labeling/", "(placeholder)", "Planned"],
    ["Backtesting", "src/backtest/", "(placeholder)", "Planned"],
    ["Reporting", "reports/", "3 PDF generators", "Ongoing"],
]
story.append(make_table(module_data, [3, 2.5, 5.5, 2.5]))

# ═══════════════════════════════════════════════════════════════════════
#  5. USE CASES
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(SH("5. Use Cases"))
story.append(Img(uc_path, w=16, h=9.3))
story.append(Spacer(1, 0.3*cm))

uc_details = [
    ["UC-01", "Download & Load Data", "Quant Researcher", "Fetch SPY OHLCV from yfinance, clean, validate columns, store CSV"],
    ["UC-02", "Compute Features", "Quant Researcher", "Generate ATR, momentum, volatility, fractional diff. features"],
    ["UC-03", "Detect Patterns", "Quant Researcher", "Run S/R, triangles, channels, multi T/B detectors on OHLCV data"],
    ["UC-04", "Detect Regimes", "Quant Researcher", "Train HMM on returns+vol, decode regime labels via Viterbi"],
    ["UC-05", "Train Signal Model", "Quant Researcher", "Train XGBoost on features+regime+patterns with triple-barrier labels"],
    ["UC-06", "Backtest Strategy", "Portfolio Manager", "Walk-forward validate strategy with realistic slippage model"],
    ["UC-07", "Generate Reports", "Both", "PDF reports with detection rates, charts, spot-checks, regime analysis"],
    ["UC-08", "Manage Risk & Size", "Portfolio Manager", "Regime-conditional Kelly sizing, drawdown brakes, VaR limits"],
    ["UC-09", "Walk-Forward Validate", "Scheduler", "Automated rolling retrain + test cycle for production monitoring"],
]
story.append(make_table(
    [["ID", "Use Case", "Actor", "Description"]] + uc_details,
    [1.3, 3.5, 2.8, 6]
))

# ═══════════════════════════════════════════════════════════════════════
#  6. CURRENT STATE & RESULTS
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(SH("6. Current State &amp; Results"))

story.append(SS("6.1 Completed Work"))
completed = [
    "<b>Data pipeline:</b> yfinance download + CSV loader for SPY (2010&ndash;2025, ~4,000 bars).",
    "<b>Shared ATR utility:</b> Wilder&rsquo;s True Range with 14-period SMA, used across all detectors.",
    "<b>4 pattern detectors</b> with ATR-based adaptive thresholds, linear regression trendlines, "
    "breakout-only firing, and structural validation (touch counts, compression checks).",
    "<b>2 iterations of parameter tuning:</b> combined rate reduced from 45.7% to 27.3%.",
    "<b>3 PDF report generators</b> documenting methodology, tuning, and visual validation.",
]
for c in completed:
    story.append(BL(c))

story.append(SS("6.2 Detection Rates (Iteration 2)"))
rates_data = [
    ["Detector", "Count", "Rate", "Iter 1 Rate", "Change"],
    ["Support / Resistance", str(int(sr.sum())), f"{sr.mean()*100:.1f}%", "28.2%", f"{sr.mean()*100 - 28.2:+.1f}pp"],
    ["Triangles", str(int(tri.sum())), f"{tri.mean()*100:.1f}%", "2.0%", f"{tri.mean()*100 - 2.0:+.1f}pp"],
    ["Channels", str(int(ch.sum())), f"{ch.mean()*100:.1f}%", "24.1%", f"{ch.mean()*100 - 24.1:+.1f}pp"],
    ["Multi Top / Bottom", str(int(mtb.sum())), f"{mtb.mean()*100:.1f}%", "3.5%", f"{mtb.mean()*100 - 3.5:+.1f}pp"],
    ["COMBINED", str(int(combined.sum())), f"{combined.mean()*100:.1f}%", "45.7%", f"{combined.mean()*100 - 45.7:+.1f}pp"],
]
story.append(make_table(rates_data, [3.5, 2, 2, 2, 2]))
story.append(Spacer(1, 0.3*cm))
story.append(Img(event_path, w=16, h=5.5))

# ═══════════════════════════════════════════════════════════════════════
#  7. ROADMAP
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(SH("7. Roadmap &amp; Implementation Plan"))

roadmap_data = [
    ["Phase", "Name", "Key Deliverables", "Dependencies"],
    ["0", "Data Pipeline", "yfinance download, CSV loader, ATR utility", "None"],
    ["1", "Pattern Detection", "4 detectors, scanner, evaluate_rates.py", "Phase 0"],
    ["2a", "Feature Engineering", "Frac. diff., momentum, vol, volume features\nCUSUM filter", "Phase 0"],
    ["2b", "Regime Detection", "GaussianHMM (2-3 states)\nViterbi decoding, BIC model selection", "Phase 0"],
    ["3", "Trade Labeling", "Triple-barrier labels\nMeta-labeling framework", "Phases 1, 2a"],
    ["4", "Signal Model", "XGBoost classifier\nRF baseline, regime-ablation", "Phases 2b, 3"],
    ["5", "Risk & Backtest", "Walk-forward engine, Kelly sizing\nDrawdown brakes, CPCV", "Phase 4"],
    ["6", "Final Evaluation", "Per-regime Sharpe/MDD decomposition\nDeflated Sharpe, bootstrap CI", "Phase 5"],
]
story.append(make_table(roadmap_data, [1.2, 3, 5.5, 3.5]))
story.append(Spacer(1, 0.5*cm))

story.append(SS("7.1 Phase 2a — Feature Engineering (Detailed)"))
fe_steps = [
    "Implement fractional differentiation using the <font face='Courier' size='9'>fracdiff</font> package. "
    "Find minimum d that passes ADF test at 95% confidence.",
    "Compute momentum features: RSI(14), MACD(12,26,9), rate-of-change at 5/21/63 horizons.",
    "Compute volatility features: realised vol (21d), Garman-Klass vol, ATR (already done).",
    "Compute volume features: relative volume (vol / 20d avg), OBV, VWAP deviation.",
    "Implement CUSUM event filter as an alternative sampling method.",
    "Build feature matrix combining all features + pattern signals for model training.",
]
for i, s in enumerate(fe_steps, 1):
    story.append(BL(f"<b>Step {i}:</b> {s}"))

story.append(SS("7.2 Phase 2b — Regime Detection (Detailed)"))
regime_steps = [
    "Train GaussianHMM with 2, 3, and 4 states on daily log returns + 21d realised vol.",
    "Select optimal state count using BIC / AIC.",
    "Decode regime labels via Viterbi algorithm. Verify regime persistence (avg duration > 20 bars).",
    "Validate: compare regime-conditional return distributions (expect different means and variances).",
    "Export regime labels as a feature column for the signal model.",
]
for i, s in enumerate(regime_steps, 1):
    story.append(BL(f"<b>Step {i}:</b> {s}"))

story.append(SS("7.3 Phase 4 — Signal Model (Detailed)"))
model_steps = [
    "Generate triple-barrier labels (TP=2xATR, SL=1xATR, expiration=10 bars) for each event bar.",
    "Build training set: features + regime + patterns as X, triple-barrier label as y.",
    "Train XGBoost classifier with walk-forward splits (3yr train / 6mo test).",
    "Train Random Forest baseline for ablation comparison.",
    "Ablation study: train same XGBoost <b>without</b> regime features to test H1.",
    "Evaluate: OOS Sharpe, hit rate, precision/recall, feature importance (SHAP).",
]
for i, s in enumerate(model_steps, 1):
    story.append(BL(f"<b>Step {i}:</b> {s}"))

# ═══════════════════════════════════════════════════════════════════════
#  8. RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(SH("8. Recommendations &amp; Suggestions"))

story.append(SS("8.1 Immediate Actions (Next Sprint)"))
imm = [
    "<b>Implement feature engineering module</b> (src/features/). Start with fractional "
    "differentiation and momentum features &mdash; these have the strongest empirical support.",
    "<b>Build HMM regime detector</b> (src/regimes/). Use hmmlearn.GaussianHMM with 2&ndash;3 "
    "states. The dependencies (hmmlearn, scikit-learn) are already in requirements.txt.",
    "<b>Vectorise triangle &amp; channel loops.</b> The current O(n&times;w) bar-by-bar loops "
    "work for ~4,000 bars but will not scale. Use numpy stride tricks or rolling regression.",
]
for a in imm:
    story.append(BL(a))

story.append(SS("8.2 Data Suggestions"))
data_sugg = [
    "<b>Multi-asset expansion:</b> Add QQQ, IWM, TLT, GLD to test regime detection across "
    "asset classes. Cross-asset regime signals are often more robust.",
    "<b>VIX as an input feature:</b> Download ^VIX and include it as a feature &mdash; it is "
    "forward-looking (implied vol) vs. ATR which is backward-looking (realised vol).",
    "<b>Volume bars:</b> Consider dollar-bar sampling (de Prado) instead of time bars for "
    "more information-uniform sampling.",
]
for s in data_sugg:
    story.append(BL(s))

story.append(SS("8.3 Model Suggestions"))
model_sugg = [
    "<b>Start simple:</b> Logistic Regression &rarr; Random Forest &rarr; XGBoost. Each model "
    "serves as a baseline for the next. Avoid jumping to deep learning without beating tree baselines.",
    "<b>Meta-labeling:</b> Train a secondary model that predicts bet size (confidence) given "
    "the primary model&rsquo;s direction signal. This naturally handles regime-dependent signal quality.",
    "<b>SHAP analysis:</b> Use SHAP values to explain which features drive predictions in each "
    "regime. This provides interpretability and helps debug the model.",
    "<b>Ensemble with HMM:</b> Consider the Gupta et al. (2025) hybrid approach: HMM + "
    "ensemble ML voting classifier for regime detection.",
]
for s in model_sugg:
    story.append(BL(s))

story.append(SS("8.4 Risk &amp; Backtest Suggestions"))
risk_sugg = [
    "<b>Use VectorBT</b> for backtesting &mdash; it is the fastest Python framework and supports "
    "vectorised operations natively. Ideal for parameter sweeps.",
    "<b>Implement the Deflated Sharpe Ratio</b> (Bailey et al., 2017) to control for "
    "multiple-testing bias when comparing strategies.",
    "<b>Regime-conditional position sizing:</b> Reduce to half-Kelly in bear regimes. "
    "Your ATR utility already provides the volatility measure for stop placement.",
    "<b>Walk-forward with embargo:</b> Use 3-year train / 6-month test windows with a "
    "5-day embargo between train and test to prevent label leakage.",
]
for s in risk_sugg:
    story.append(BL(s))

story.append(SS("8.5 Thesis-Level Contributions"))
story.append(B("To elevate this project to thesis quality, focus on these original contributions:"))
thesis_sugg = [
    "<b>Regime-ablation study:</b> Rigorously compare &lsquo;same model with regime&rsquo; vs "
    "&lsquo;same model without regime&rsquo; on identical walk-forward splits. This directly "
    "tests H1 and is the core contribution.",
    "<b>Per-regime performance decomposition:</b> Report Sharpe, MDD, hit rate separately "
    "for bull, bear, and sideways regimes. Show that the system adapts its behaviour.",
    "<b>Bootstrap confidence intervals:</b> Use block bootstrap to compute confidence "
    "intervals for all metrics, establishing statistical significance.",
    "<b>Transaction cost sensitivity:</b> Show how results degrade across different "
    "slippage assumptions (5bps, 10bps, 20bps).",
]
for s in thesis_sugg:
    story.append(BL(s))

# ═══════════════════════════════════════════════════════════════════════
#  9. REFERENCES
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(SH("9. References"))

refs = [
    "[1] Hamilton, J.D. (1989). &ldquo;A New Approach to the Economic Analysis of Nonstationary "
    "Time Series and the Business Cycle.&rdquo; <i>Econometrica</i>, 57(2), 357&ndash;384.",

    "[2] Nystrup, P., Kolm, P.N. &amp; Lindstrom, E. (2020). &ldquo;Greedy Online Classification "
    "of Persistent Market States Using Realized Intraday Volatility Features.&rdquo; "
    "<i>Journal of Financial Data Science</i>, 2(3).",

    "[3] Gupta, R., Kapoor, S., Gupta, H. &amp; Natesan, S. (2025). &ldquo;A Forest of Opinions: "
    "A Multi-Model Ensemble-HMM Voting Framework for Market Regime Shift Detection.&rdquo; "
    "<i>Data Science in Finance and Economics</i>, 5(4): 466&ndash;501.",

    "[4] Lo, A.W., Mamaysky, H. &amp; Wang, J. (2000). &ldquo;Foundations of Technical Analysis: "
    "Computational Algorithms, Statistical Inference, and Empirical Implementation.&rdquo; "
    "<i>Journal of Finance</i>, 55(4), 1705&ndash;1765.",

    "[5] Brock, W., Lakonishok, J. &amp; LeBaron, B. (1992). &ldquo;Simple Technical Trading Rules "
    "and the Stochastic Properties of Stock Returns.&rdquo; <i>Journal of Finance</i>, 47(5), 1731&ndash;1764.",

    "[6] de Prado, M.L. (2018). <i>Advances in Financial Machine Learning</i>. Wiley.",

    "[7] Bailey, D.H., Borwein, J.M., de Prado, M.L. &amp; Zhu, Q.J. (2017). &ldquo;The Probability "
    "of Backtest Overfitting.&rdquo; <i>Journal of Computational Finance</i>.",

    "[8] Ang, A. &amp; Bekaert, G. (2002). &ldquo;International Asset Allocation with Regime Shifts.&rdquo; "
    "<i>Review of Financial Studies</i>, 15(4), 1137&ndash;1187.",

    "[9] Thorp, E.O. (2006). &ldquo;The Kelly Criterion in Blackjack, Sports Betting, and the "
    "Stock Market.&rdquo; In <i>Handbook of Asset and Liability Management</i>.",

    "[10] Aydinhan, A., Kolm, P.N., Mulvey, J.M. &amp; Shu, Y.O. (2024). &ldquo;Identifying Patterns "
    "in Financial Markets: Extending the Statistical Jump Model.&rdquo; "
    "<i>Annals of Operations Research</i>.",

    "[11] Ibanez, F.A. (2024). &ldquo;Incorporating Market Regimes into Large-Scale Stock Portfolios.&rdquo; "
    "MPRA Paper No. 121552.",

    "[12] Lo, A.W. (2004). &ldquo;The Adaptive Markets Hypothesis.&rdquo; "
    "<i>Journal of Portfolio Management</i>, 30(5), 15&ndash;29.",

    "[13] RegimeFolio (2025). &ldquo;Regime-Based Sector-Specific Portfolio Optimization.&rdquo; "
    "Arxiv 2510.14986 / IEEE Access.",

    "[14] Sullivan, R., Timmermann, A. &amp; White, H. (1999). &ldquo;Data-Snooping, Technical "
    "Trading Rule Performance, and the Bootstrap.&rdquo; <i>Journal of Finance</i>, 54(5).",
]
for r in refs:
    story.append(Ref(r))
    story.append(Spacer(1, 2*mm))

# ── BUILD ────────────────────────────────────────────────────────────────
print("Building PDF...")
doc.build(story)
print(f"Report generated: {OUTPUT_PATH}")
