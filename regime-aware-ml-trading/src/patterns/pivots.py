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
    """Count how many bars genuinely touch a trendline.

    A touch requires three conditions:
    1. The bar's price extreme is within tolerance of the line.
    2. The bar is *approaching* the line (not deep inside the channel) —
       for upper lines the High must be at or above (line - tolerance);
       slight breaches up to tolerance/2 past the line are allowed.
    3. The bar is a *local extreme* in a ±1-bar neighbourhood, so
       consecutive bars in a flat cluster near the line are not all
       counted as separate touches.

    Parameters
    ----------
    prices : np.ndarray
        High array (for upper) or Low array (for lower).
    line_values : np.ndarray
        Trendline values at each bar position (same length as prices).
    tolerance : float
        Maximum distance to count as a touch (use 0.15 × ATR).
    side : str
        "upper" — High approaching upper line from below.
        "lower" — Low approaching lower line from above.

    Returns
    -------
    dict with keys:
        touch_count, touch_indices, mean_error, max_error, violations.
    """
    n = min(len(prices), len(line_values))
    prices = prices[:n]
    line_values = line_values[:n]

    if side == "upper":
        # signed_dist > 0 means price is below the line (inside)
        signed_dist = line_values - prices
    else:
        # signed_dist > 0 means price is above the line (inside)
        signed_dist = prices - line_values

    abs_dist = np.abs(signed_dist)

    # Condition 1+2: price extreme is close to the line AND approaching it.
    # Allow the wick to be anywhere from (tolerance/2 past the line)
    # to (tolerance below the line).
    near_mask = (signed_dist >= -tolerance * 0.5) & (signed_dist <= tolerance)

    # Condition 3: local extreme — bar's price is the most extreme
    # (highest High for upper, lowest Low for lower) in a ±1-bar window.
    local_ext = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n, i + 2)
        if side == "upper":
            local_ext[i] = prices[i] >= prices[lo:hi].max() - 1e-9
        else:
            local_ext[i] = prices[i] <= prices[lo:hi].min() + 1e-9

    touch_mask = near_mask & local_ext
    violation_mask = signed_dist < -tolerance

    touch_idx = list(np.nonzero(touch_mask)[0])
    touch_errors = abs_dist[touch_mask]

    return {
        "touch_count": int(touch_mask.sum()),
        "touch_indices": touch_idx,
        "mean_error": float(touch_errors.mean()) if len(touch_errors) > 0 else 0.0,
        "max_error": float(touch_errors.max()) if len(touch_errors) > 0 else 0.0,
        "violations": int(violation_mask.sum()),
    }
