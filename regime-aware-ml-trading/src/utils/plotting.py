"""Shared plotting utilities for the regime-aware trading project.

Provides candlestick charts and annotation helpers used across notebooks
and the pattern validation module.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


def plot_candlestick(df, ax=None, title=None, figsize=(14, 5)):
    """Plot an OHLC candlestick chart using matplotlib rectangles.

    Parameters
    ----------
    df : pd.DataFrame
        Must have DatetimeIndex and columns: Open, High, Low, Close.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates a new figure.
    title : str, optional
        Chart title.
    figsize : tuple
        Figure size if creating a new figure.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    dates = mdates.date2num(df.index.to_pydatetime())
    # Bar width: ~60% of average spacing between bars
    if len(dates) > 1:
        bar_width = 0.6 * np.median(np.diff(dates))
    else:
        bar_width = 0.5

    for i in range(len(df)):
        o, h, l, c = df["Open"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i]
        color = "#2ca02c" if c >= o else "#d62728"  # green up, red down

        # Wick (high-low line)
        ax.plot([dates[i], dates[i]], [l, h], color=color, linewidth=0.8)

        # Body (open-close rectangle)
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < 1e-9:
            body_height = (h - l) * 0.05  # doji: small visible body
        rect = Rectangle(
            (dates[i] - bar_width / 2, body_bottom),
            bar_width, body_height,
            facecolor=color, edgecolor=color, linewidth=0.5
        )
        ax.add_patch(rect)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=45)
    ax.set_ylabel("Price")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Let matplotlib auto-scale based on the data
    ax.autoscale_view()

    return fig, ax


def add_horizontal_line(ax, y, color="blue", linestyle="--", alpha=0.7, label=None):
    """Add a horizontal price level to an axes."""
    ax.axhline(y=y, color=color, linestyle=linestyle, alpha=alpha, label=label)


def add_event_marker(ax, date, price, marker="^", color="blue", size=80, label=None):
    """Mark a specific event on the chart."""
    x = mdates.date2num(date)
    ax.scatter(x, price, marker=marker, color=color, s=size, zorder=5, label=label)


def add_trendline(ax, df, coeffs, window, color="blue", linestyle="-", alpha=0.6, label=None):
    """Draw a trendline on the axes using polyfit coefficients.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df : pd.DataFrame
        The window slice used for fitting.
    coeffs : array-like
        [slope, intercept] from np.polyfit.
    window : int
        Number of bars in the window.
    color, linestyle, alpha, label : styling parameters.
    """
    x = np.arange(window)
    y = np.polyval(coeffs, x)
    dates = mdates.date2num(df.index.to_pydatetime())
    ax.plot(dates, y, color=color, linestyle=linestyle, alpha=alpha, label=label)
