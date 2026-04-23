"""Pivot-point detection, chunk-based extremes, containment, and touch utilities.

Two approaches for finding representative highs/lows:

1. **Swing pivots** (for triangles): bar *i* is a swing high if High[i]
   is the max within ±*order* bars.
2. **Chunk extremes** (for channels, from TrendLineChannelDetection
   reference notebook): divide the window into non-overlapping chunks
   of *wind* bars; each chunk contributes its max-High bar and
   min-Low bar.

References
----------
Lo, Mamaysky & Wang (2000).  "Foundations of Technical Analysis."
"""

import numpy as np


# -------------------------------------------------------------------
# 1. Swing pivot detection (for triangles)
# -------------------------------------------------------------------

def find_swing_highs(highs, order=3):
    """Return indices where High[i] is the max within ±order bars."""
    pivots = []
    for i in range(order, len(highs) - order):
        window = highs[i - order: i + order + 1]
        if highs[i] >= window.max():
            pivots.append(i)
    return pivots


def find_swing_lows(lows, order=3):
    """Return indices where Low[i] is the min within ±order bars."""
    pivots = []
    for i in range(order, len(lows) - order):
        window = lows[i - order: i + order + 1]
        if lows[i] <= window.min():
            pivots.append(i)
    return pivots


# -------------------------------------------------------------------
# 2. Chunk-based extremes (for channels — reference notebook method)
# -------------------------------------------------------------------

def chunk_extremes(highs, lows, wind=5):
    """Divide the array into chunks of *wind* bars, return max-High
    and min-Low indices from each chunk.

    This is the approach from the *TrendLineChannelDetection* reference
    notebook — it guarantees one representative point per chunk with
    regular spacing.

    Parameters
    ----------
    highs, lows : np.ndarray
        High and Low price arrays (same length).
    wind : int
        Chunk size (default 5).

    Returns
    -------
    (xxmax, maxim, xxmin, minim) : tuple of np.ndarray
        xxmax/xxmin = indices of the extreme bars.
        maxim/minim = prices at those bars.
    """
    n = len(highs)
    xxmax, maxim = [], []
    xxmin, minim = [], []
    for start in range(0, n, wind):
        end = min(start + wind, n)
        chunk_h = highs[start:end]
        chunk_l = lows[start:end]
        if len(chunk_h) == 0:
            continue
        rel_idx_h = int(np.argmax(chunk_h))
        xxmax.append(start + rel_idx_h)
        maxim.append(float(chunk_h[rel_idx_h]))
        rel_idx_l = int(np.argmin(chunk_l))
        xxmin.append(start + rel_idx_l)
        minim.append(float(chunk_l[rel_idx_l]))
    return (np.array(xxmax, dtype=float), np.array(maxim),
            np.array(xxmin, dtype=float), np.array(minim))


# -------------------------------------------------------------------
# 3. Containment ratio
# -------------------------------------------------------------------

def containment_ratio(highs, lows, upper_line, lower_line, tolerance=0.0):
    """Fraction of bars fully contained between upper and lower boundaries.

    A bar is contained when  High <= upper + tol  AND  Low >= lower - tol.
    """
    contained = (highs <= upper_line + tolerance) & (lows >= lower_line - tolerance)
    return float(contained.mean())


# -------------------------------------------------------------------
# 4. Trendline touch counting
# -------------------------------------------------------------------

def count_touches(prices, line_values, tolerance, side="upper"):
    """Count how many bars touch a trendline within tolerance.

    A touch is defined as a bar whose relevant price extreme is within
    *tolerance* of the line value. For upper lines we check Highs;
    for lower lines we check Lows.

    Parameters
    ----------
    prices : np.ndarray
        High array (for upper) or Low array (for lower).
    line_values : np.ndarray
        Trendline values at each bar position (same length as prices).
    tolerance : float
        Maximum distance to count as a touch (typically 0.2–0.5 × ATR).
    side : str
        "upper" — prices should approach from below (High near upper line).
        "lower" — prices should approach from above (Low near lower line).

    Returns
    -------
    dict with keys:
        touch_count : int — number of bars touching within tolerance.
        touch_indices : list[int] — bar indices that touch.
        mean_error : float — mean absolute distance from line at touch points.
        max_error : float — max absolute distance at touch points.
        violations : int — bars that breach the line beyond tolerance.
    """
    n = min(len(prices), len(line_values))
    prices = prices[:n]
    line_values = line_values[:n]

    if side == "upper":
        distance = line_values - prices   # positive = inside channel
    else:
        distance = prices - line_values   # positive = inside channel

    abs_dist = np.abs(distance)
    touch_mask = abs_dist <= tolerance
    violation_mask = distance < -tolerance  # breached beyond tolerance

    touch_idx = list(np.where(touch_mask)[0])
    touch_errors = abs_dist[touch_mask]

    return {
        "touch_count": int(touch_mask.sum()),
        "touch_indices": touch_idx,
        "mean_error": float(touch_errors.mean()) if len(touch_errors) > 0 else 0.0,
        "max_error": float(touch_errors.max()) if len(touch_errors) > 0 else 0.0,
        "violations": int(violation_mask.sum()),
    }
