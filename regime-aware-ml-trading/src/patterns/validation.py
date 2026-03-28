"""Pattern validation module for visual quality assessment.

Provides tools to:
- Sample detected events by type
- Plot each event with its candlestick context and annotations
- Compute forward-return quality metrics per detector
- Support manual accept/reject tagging for ground-truth building

Usage
-----
    from src.patterns.validation import EventValidator
    validator = EventValidator(df_with_patterns)
    validator.plot_sample(detector="triangles", n=10)
    validator.quality_report()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.utils import compute_atr
from src.utils.plotting import plot_candlestick, add_horizontal_line, add_event_marker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_event_type(row):
    """Return the primary event type for a row (first match wins)."""
    if pd.notna(row.get("triangle_pattern")):
        return row["triangle_pattern"]
    if pd.notna(row.get("channel_pattern")):
        return row["channel_pattern"]
    if pd.notna(row.get("multiple_top_bottom_pattern")):
        return row["multiple_top_bottom_pattern"]
    if row.get("near_support", False):
        return "near_support"
    if row.get("near_resistance", False):
        return "near_resistance"
    return None


def _forward_return(df, idx, horizon=10):
    """Compute the forward return from idx over *horizon* bars.

    Returns NaN if not enough future bars exist.
    """
    loc = df.index.get_loc(idx)
    if loc + horizon >= len(df):
        return np.nan
    entry_price = df["Close"].iloc[loc]
    exit_price = df["Close"].iloc[loc + horizon]
    return (exit_price - entry_price) / entry_price


# ---------------------------------------------------------------------------
# EventValidator
# ---------------------------------------------------------------------------

class EventValidator:
    """Visual and quantitative validation of detected pattern events.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV DataFrame *after* running scan_all_patterns().
        Must contain pattern columns (near_support, triangle_pattern, etc.)
        and standard OHLCV columns.
    context_bars : int
        Number of bars to show before and after each event (default 30).
    forward_horizon : int
        Number of bars for forward-return calculation (default 10).
    """

    def __init__(self, df, context_bars=30, forward_horizon=10):
        self.df = df.copy()
        self.context_bars = context_bars
        self.forward_horizon = forward_horizon
        self.atr = compute_atr(df)

        # Classify each event row
        self.df["event_type"] = self.df.apply(_classify_event_type, axis=1)
        self.events = self.df[self.df["event_type"].notna()].copy()

        # Pre-compute forward returns for all events
        self.events["fwd_return"] = [
            _forward_return(self.df, idx, self.forward_horizon)
            for idx in self.events.index
        ]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_events(self, detector=None, n=10, seed=42):
        """Sample *n* events, optionally filtered by detector type.

        Parameters
        ----------
        detector : str or None
            Filter to a specific event type (e.g. 'ascending_triangle',
            'near_support', 'channel_up'). If None, samples from all events.
        n : int
            Number of events to sample.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Sampled event rows.
        """
        pool = self.events
        if detector is not None:
            pool = pool[pool["event_type"] == detector]
        n = min(n, len(pool))
        if n == 0:
            print(f"No events found for detector='{detector}'")
            return pd.DataFrame()
        return pool.sample(n=n, random_state=seed).sort_index()

    def get_detector_types(self):
        """Return a list of unique event types detected."""
        return sorted(self.events["event_type"].dropna().unique().tolist())

    # ------------------------------------------------------------------
    # Single-event plot
    # ------------------------------------------------------------------

    def plot_event(self, event_date, ax=None, show_levels=True):
        """Plot a single event with candlestick context and annotations.

        Parameters
        ----------
        event_date : pd.Timestamp
            The date of the event to plot.
        ax : matplotlib.axes.Axes, optional
            If None, creates a new figure.
        show_levels : bool
            If True, draw support/resistance levels when available.

        Returns
        -------
        fig, ax
        """
        loc = self.df.index.get_loc(event_date)
        start = max(0, loc - self.context_bars)
        end = min(len(self.df), loc + self.context_bars + 1)
        window = self.df.iloc[start:end]

        event_row = self.df.loc[event_date]
        event_type = event_row.get("event_type", "unknown")

        # Title with event info
        title = f"{event_type}  |  {event_date.strftime('%Y-%m-%d')}  |  Close={event_row['Close']:.2f}"

        fig, ax = plot_candlestick(window, ax=ax, title=title)

        # Mark the event bar
        marker = "^" if "bottom" in str(event_type) or "support" in str(event_type) else "v"
        color = "#2ca02c" if marker == "^" else "#d62728"
        add_event_marker(ax, event_date, event_row["Close"], marker=marker, color=color, size=120)

        # Draw support/resistance levels if available
        if show_levels and "support" in self.df.columns:
            support_val = event_row.get("support")
            resistance_val = event_row.get("resistance")
            if pd.notna(support_val):
                add_horizontal_line(ax, support_val, color="#2ca02c", label=f"Support {support_val:.2f}")
            if pd.notna(resistance_val):
                add_horizontal_line(ax, resistance_val, color="#d62728", label=f"Resistance {resistance_val:.2f}")

        # Show forward return if available
        fwd_ret = _forward_return(self.df, event_date, self.forward_horizon)
        if not np.isnan(fwd_ret):
            fwd_text = f"{self.forward_horizon}-bar fwd: {fwd_ret*100:+.2f}%"
            ax.text(0.02, 0.95, fwd_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.legend(loc="lower right", fontsize=8)
        return fig, ax

    # ------------------------------------------------------------------
    # Batch plots
    # ------------------------------------------------------------------

    def plot_sample(self, detector=None, n=10, cols=2, seed=42):
        """Plot a grid of sampled events for visual inspection.

        Parameters
        ----------
        detector : str or None
            Filter by event type. None = all types.
        n : int
            Number of events to show.
        cols : int
            Number of columns in the grid.
        seed : int
            Random seed.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        sample = self.sample_events(detector=detector, n=n, seed=seed)
        if sample.empty:
            return None

        rows = int(np.ceil(len(sample) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows))
        axes = np.atleast_2d(axes)

        for idx, (event_date, _) in enumerate(sample.iterrows()):
            r, c = divmod(idx, cols)
            self.plot_event(event_date, ax=axes[r, c], show_levels=True)

        # Hide unused axes
        for idx in range(len(sample), rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].set_visible(False)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    def quality_report(self):
        """Compute per-detector quality metrics.

        Returns
        -------
        pd.DataFrame
            Columns: detector, count, fwd_mean, fwd_median, pct_positive,
            pct_negative, avg_magnitude
        """
        records = []
        for etype in self.get_detector_types():
            subset = self.events[self.events["event_type"] == etype]
            fwd = subset["fwd_return"].dropna()
            records.append({
                "detector": etype,
                "count": len(subset),
                "fwd_mean_%": fwd.mean() * 100 if len(fwd) > 0 else np.nan,
                "fwd_median_%": fwd.median() * 100 if len(fwd) > 0 else np.nan,
                "pct_positive": (fwd > 0).mean() * 100 if len(fwd) > 0 else np.nan,
                "pct_negative": (fwd < 0).mean() * 100 if len(fwd) > 0 else np.nan,
                "avg_magnitude_%": fwd.abs().mean() * 100 if len(fwd) > 0 else np.nan,
            })
        return pd.DataFrame.from_records(records)

    def event_density_by_month(self):
        """Return monthly event counts and rates.

        Returns
        -------
        pd.DataFrame
            Indexed by month with columns: total_bars, event_count, event_rate.
        """
        monthly_total = self.df.resample("ME").size()
        monthly_events = self.events.resample("ME").size()
        density = pd.DataFrame({
            "total_bars": monthly_total,
            "event_count": monthly_events,
        }).fillna(0)
        density["event_rate"] = density["event_count"] / density["total_bars"]
        return density

    def event_density_by_detector(self):
        """Return event count breakdown by detector type.

        Returns
        -------
        pd.Series
            Counts indexed by event_type.
        """
        return self.events["event_type"].value_counts()
