"""Generate a PDF progress report for the Regime-Aware ML Trading project."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fpdf import FPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data.load_data import load_spy
from src.patterns.scanner import scan_all_patterns

REPORT_DIR = os.path.dirname(__file__)


def create_price_chart(df, path):
    """Create a simple SPY price chart and save as PNG."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Close"], linewidth=0.7, color="#2563eb")
    ax.set_title("SPY Daily Close Price (2010-2025)", fontsize=12)
    ax.set_ylabel("Price ($)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_event_chart(df, path):
    """Create a chart showing event density over time."""
    # Monthly event rate
    monthly = df["has_event"].resample("ME").mean() * 100
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(monthly.index, monthly.values, width=25, color="#10b981", alpha=0.7)
    ax.set_title("Monthly Event Rate (%)", fontsize=12)
    ax.set_ylabel("% of days with events")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_pattern_breakdown_chart(df, path):
    """Bar chart of pattern type counts."""
    counts = {
        "Near Support": int(df["near_support"].sum()),
        "Near Resistance": int(df["near_resistance"].sum()),
        "Triangles": int(df["triangle_pattern"].notna().sum()),
        "Multi Top/Bottom": int(df["multiple_top_bottom_pattern"].notna().sum()),
        "Channels": int(df["channel_pattern"].notna().sum()),
    }
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(list(counts.keys()), list(counts.values()), color="#6366f1")
    ax.set_title("Pattern Detection Counts", fontsize=12)
    ax.set_xlabel("Number of days detected")
    for bar, val in zip(bars, counts.values()):
        ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_pdf():
    # Load data and run patterns
    df = load_spy()
    df = scan_all_patterns(df, window=20)
    event_count = int(df["has_event"].sum())
    total = len(df)

    # Generate chart images
    charts_dir = os.path.join(REPORT_DIR, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    price_chart = os.path.join(charts_dir, "spy_price.png")
    event_chart = os.path.join(charts_dir, "event_rate.png")
    pattern_chart = os.path.join(charts_dir, "pattern_breakdown.png")

    create_price_chart(df, price_chart)
    create_event_chart(df, event_chart)
    create_pattern_breakdown_chart(df, pattern_chart)

    # Build PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # --- Page 1: Title + What Was Done ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 15, "Regime-Aware ML Trading", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, "Progress Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)
    pdf.set_draw_color(100, 100, 100)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(10)

    # Section: What Was Done
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "1. Completed Work", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1.1 Data Pipeline", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    items = [
        "Downloaded SPY daily OHLCV data via yfinance (2010-2025)",
        f"Dataset: {total:,} trading days, ~200KB CSV file",
        "Built data loader with cleaning and validation",
        "Data stored in data/raw/spy.csv",
    ]
    for item in items:
        pdf.cell(10)
        pdf.cell(0, 6, f"- {item}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Price chart
    pdf.image(price_chart, x=15, w=180)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1.2 Pattern Detection Module", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    items = [
        "Adapted 4 pattern detectors from TradingPatternScanner reference repo:",
        "  (a) Support & Resistance: 2-sigma bands on rolling High/Low + proximity flags",
        "  (b) Triangle Patterns: ascending/descending based on converging rolling extremes",
        "  (c) Multiple Tops/Bottoms: highs hitting ceiling while closes turn down (and mirror)",
        "  (d) Channel Detection: parallel trending bands with range constraint",
        "Built unified scanner (scanner.py) that runs all detectors",
        f"Results: {event_count:,} event bars out of {total:,} ({100*event_count/total:.1f}%)",
    ]
    for item in items:
        pdf.cell(10)
        pdf.cell(0, 6, f"- {item}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # --- Page 2: Charts + Known Issues ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1.3 Pattern Detection Results", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.image(pattern_chart, x=15, w=170)
    pdf.ln(5)
    pdf.image(event_chart, x=15, w=170)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1.4 Known Issues", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    items = [
        "Event rate is very high (96.6%) - triangle and channel detectors are too loose",
        "These are heuristic approximations of classical TA patterns, not rigorous definitions",
        "Window parameter (20 days) needs tuning; larger windows may reduce false positives",
        "Pattern detectors need stricter convergence/divergence thresholds",
    ]
    for item in items:
        pdf.cell(10)
        pdf.cell(0, 6, f"- {item}", new_x="LMARGIN", new_y="NEXT")

    # Section: Project Structure
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1.5 Project Structure", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", "", 8)
    structure = [
        "regime-aware-ml-trading/",
        "  data/raw/spy.csv              <- Downloaded SPY data",
        "  src/",
        "    data/download_data.py        <- yfinance downloader",
        "    data/load_data.py            <- Data loader + cleaner",
        "    patterns/",
        "      support_resistance.py      <- S/R level detection",
        "      triangles.py               <- Triangle pattern detection",
        "      multiple_tops_bottoms.py   <- Multiple top/bottom detection",
        "      channels.py                <- Channel pattern detection",
        "      scanner.py                 <- Unified pattern scanner",
        "    regimes/                     <- (next: HMM regime detection)",
        "    features/                    <- (next: feature engineering)",
        "    labeling/                    <- (next: trade outcome labeling)",
        "    models/                      <- (next: RF + Logistic Regression)",
        "    backtest/                    <- (next: walk-forward backtesting)",
        "  reports/                       <- This report",
        "  tests/                         <- (next: unit tests)",
    ]
    for line in structure:
        pdf.cell(10)
        pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")

    # --- Page 3: Plan ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "2. Plan: What Comes Next", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    phases = [
        ("Phase 1: Tune Pattern Detectors (Current Priority)", [
            "Increase window sizes and add stricter thresholds to reduce false signals",
            "Target event rate: 10-30% of trading days",
            "Validate patterns visually on price charts",
        ]),
        ("Phase 2: Feature Engineering", [
            "Technical indicators: RSI, moving average crossovers, ATR, Bollinger Bands",
            "Return features: log returns, lagged returns (1d, 5d, 21d), rolling Sharpe",
            "Pattern-specific features: distance to S/R, pattern duration, breakout direction",
        ]),
        ("Phase 3: HMM Regime Detection", [
            "Fit GaussianHMM on rolling returns + volatility (2-3 states)",
            "Label regimes via Viterbi decoding (bull / bear / sideways)",
            "Use BIC/AIC for model selection on number of states",
            "Create regime features: regime ID, duration, transition probabilities",
        ]),
        ("Phase 4: Trade Labeling", [
            "For each event, look forward N bars and assign: Long / Short / No-trade",
            "Use fixed-horizon return thresholds or triple-barrier method",
            "Check class balance and plan resampling if needed",
        ]),
        ("Phase 5: ML Models", [
            "Logistic Regression baseline (interpretable coefficients)",
            "Random Forest classifier (handles nonlinearity)",
            "Regime-agnostic ablation: same models without regime features",
            "Non-event ablation: train on all timesteps (not just events)",
        ]),
        ("Phase 6: Walk-Forward Backtesting", [
            "Rolling train/test splits (e.g., 3yr train, 6mo test, slide forward)",
            "Re-fit HMM + ML models each window to avoid lookahead bias",
            "Transaction costs: ~10bps round-trip slippage + commission",
            "Metrics: Sharpe, Sortino, max drawdown, Calmar, turnover, win rate",
        ]),
        ("Phase 7: Comparison & Reporting", [
            "Full strategy vs regime-agnostic vs non-event vs buy-and-hold",
            "Bootstrap confidence intervals on Sharpe ratio",
            "Regime-conditional performance analysis",
            "Final report answering the research question",
        ]),
    ]

    for title, bullets in phases:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for b in bullets:
            pdf.cell(10)
            pdf.cell(0, 6, f"- {b}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # Research question reminder
    pdf.ln(5)
    pdf.set_draw_color(100, 100, 100)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Helvetica", "BI", 10)
    pdf.multi_cell(0, 6,
        "Research Question: Does combining HMM-based market regime detection with "
        "event-driven technical signal selection improve out-of-sample risk-adjusted "
        "trading performance compared to regime-agnostic and non-event-based models?"
    )

    # Save
    output_path = os.path.join(REPORT_DIR, "progress_report.pdf")
    pdf.output(output_path)
    print(f"PDF saved to {output_path}")


if __name__ == "__main__":
    generate_pdf()
