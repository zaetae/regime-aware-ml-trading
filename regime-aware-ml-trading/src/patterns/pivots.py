"""Pivot-point (swing high / swing low) detection and containment utilities.

Shared by the triangle and channel detectors. A swing high at bar *i*
has the highest High in a symmetric neighbourhood of ±*order* bars.
Swing lows are defined analogously.

References
----------
Lo, Mamaysky & Wang (2000).  "Foundations of Technical Analysis."
"""

import numpy as np


def find_swing_highs(highs, order=5):
    """Return indices where High[i] is the max within ±order bars.

    Parameters
    ----------
    highs : np.ndarray
        High prices for the window.
    order : int
        Half-width of the neighbourhood (default 5 → 11-bar window).

    Returns
    -------
    list[int]
        Indices (relative to the input array) of confirmed swing highs.
    """
    pivots = []
    for i in range(order, len(highs) - order):
        window = highs[i - order: i + order + 1]
        if highs[i] == window.max() and np.sum(window == highs[i]) == 1:
            pivots.append(i)
    return pivots


def find_swing_lows(lows, order=5):
    """Return indices where Low[i] is the min within ±order bars."""
    pivots = []
    for i in range(order, len(lows) - order):
        window = lows[i - order: i + order + 1]
        if lows[i] == window.min() and np.sum(window == lows[i]) == 1:
            pivots.append(i)
    return pivots


def containment_ratio(highs, lows, upper_line, lower_line, tolerance=0.0):
    """Fraction of bars fully contained between upper and lower boundaries.

    A bar is contained when  High <= upper + tol  AND  Low >= lower - tol.

    Parameters
    ----------
    highs, lows : np.ndarray
        High and Low prices (same length as *upper_line* / *lower_line*).
    upper_line, lower_line : np.ndarray
        Trendline values evaluated at each bar position.
    tolerance : float
        Absolute tolerance (typically a fraction of ATR).

    Returns
    -------
    float
        Ratio in [0, 1].
    """
    contained = (highs <= upper_line + tolerance) & (lows >= lower_line - tolerance)
    return float(contained.mean())
