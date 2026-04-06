"""Export triangle & channel detections with boundary-line charts and CSV.

Provides a pipeline to:
1. Run detectors with ``return_details=True`` to capture trendline metadata.
2. Generate per-detection candlestick charts with upper/lower boundary lines.
3. Export a summary CSV with all trendline coefficients.

Can be run as a script: ``python -m src.patterns.export_patterns``
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for chart export
import matplotlib.pyplot as plt
import pandas as pd

from src.data.load_data import load_spy
from src.patterns.triangles import detect_triangle_pattern
from src.patterns.channels import detect_channel
from src.utils.plotting import plot_candlestick, add_trendline, add_event_marker


# ---------------------------------------------------------------------------
# 1. Collect details from both detectors
# ---------------------------------------------------------------------------

def collect_pattern_details(df, tri_kwargs=None, ch_kwargs=None):
    """Run triangle and channel detectors and return combined detail dicts.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex.
    tri_kwargs, ch_kwargs : dict, optional
        Extra keyword arguments forwarded to each detector.

    Returns
    -------
    list[dict]
        Combined list of detection metadata dicts.
    """
    tri_kwargs = tri_kwargs or {}
    ch_kwargs = ch_kwargs or {}

    _, tri_details = detect_triangle_pattern(df, return_details=True, **tri_kwargs)
    _, ch_details = detect_channel(df, return_details=True, **ch_kwargs)

    return tri_details + ch_details


# ---------------------------------------------------------------------------
# 2. Export individual charts
# ---------------------------------------------------------------------------

FORWARD_CONTEXT = 10  # extra bars shown after the event bar


def export_pattern_charts(details, df, base_dir="reports/charts"):
    """Save a candlestick + trendline PNG for each detection.

    Charts are saved under ``base_dir/triangles/`` or ``base_dir/channels/``
    depending on the pattern type.
    """
    tri_dir = os.path.join(base_dir, "triangles")
    ch_dir = os.path.join(base_dir, "channels")
    os.makedirs(tri_dir, exist_ok=True)
    os.makedirs(ch_dir, exist_ok=True)

    for det in details:
        start = det["start_idx"]
        end = min(det["end_idx"] + FORWARD_CONTEXT, len(df) - 1)
        chart_slice = df.iloc[start:end + 1]
        window_slice = df.iloc[det["start_idx"]:det["end_idx"]]

        fig, ax = plot_candlestick(
            chart_slice,
            title=f"{det['pattern_type']}  —  {det['event_date'].strftime('%Y-%m-%d')}",
        )

        upper_coeffs = [det["upper_slope"], det["upper_intercept"]]
        lower_coeffs = [det["lower_slope"], det["lower_intercept"]]

        add_trendline(ax, window_slice, upper_coeffs, det["window"],
                      color="red", label="Upper")
        add_trendline(ax, window_slice, lower_coeffs, det["window"],
                      color="blue", label="Lower")

        event_price = df.loc[det["event_date"], "Close"]
        add_event_marker(ax, det["event_date"], event_price,
                         marker="v", color="orange", size=100,
                         label="Detection")

        ax.legend(fontsize=8)

        is_triangle = "triangle" in det["pattern_type"]
        out_dir = tri_dir if is_triangle else ch_dir
        fname = f"{det['event_date'].strftime('%Y-%m-%d')}_{det['pattern_type']}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=100, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Export CSV summary
# ---------------------------------------------------------------------------

def export_pattern_csv(details, output_path="outputs/patterns_summary.csv"):
    """Write detection details to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cols = [
        "event_date", "pattern_type", "start_date", "end_date",
        "upper_slope", "upper_intercept", "lower_slope", "lower_intercept",
    ]
    summary = pd.DataFrame(details)[cols]
    summary.to_csv(output_path, index=False)
    return summary


# ---------------------------------------------------------------------------
# 4. Convenience wrapper
# ---------------------------------------------------------------------------

def export_all(df, charts_dir="reports/charts",
               csv_path="outputs/patterns_summary.csv"):
    """Collect detections, export charts and CSV, print summary."""
    details = collect_pattern_details(df)

    export_pattern_charts(details, df, base_dir=charts_dir)
    summary = export_pattern_csv(details, output_path=csv_path)

    tri_count = sum(1 for d in details if "triangle" in d["pattern_type"])
    ch_count = len(details) - tri_count
    print(f"Exported {len(details)} patterns: "
          f"{tri_count} triangles, {ch_count} channels")
    print(f"  Charts → {charts_dir}/triangles/ & {charts_dir}/channels/")
    print(f"  CSV    → {csv_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    spy = load_spy()
    export_all(spy)
