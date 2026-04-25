"""Channel detection using chunk-based extremes + dynamic window optimisation.

Adapted from the *TrendLineChannelDetection* reference notebook.

The approach:
1. Divide the lookback window into chunks of ``wind`` bars.
2. Take the max-High and min-Low from each chunk.
3. ``polyfit`` (degree 1) on those chunk extremes → slope.
4. Try multiple lookback lengths; pick the one with the **tightest**
   channel at the current bar (smallest distance between upper and
   lower line, i.e. most parallel and closest).
5. Adjust intercepts to **wrap** all candles (upper caps all chunk highs,
   lower floors all chunk lows).
6. Validate parallelism, width (1–6× ATR), containment (≥ 60%).
"""

import numpy as np
import pandas as pd

from src.data.utils import compute_atr
from src.patterns.pivots import chunk_extremes, containment_ratio, count_touches


def detect_channel(df, backcandles=40, brange=15, wind=5,
                   slope_tolerance=0.25, cooldown=10,
                   return_details=False, min_containment=0.60):
    """Detect price channels using the reference notebook method.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: High, Low, Close
    backcandles : int
        Base lookback window (default 40).
    brange : int
        Half-range for dynamic window search (default 15).
        The detector tries lookbacks from backcandles-brange to
        backcandles+brange and picks the tightest channel.
    wind : int
        Chunk size for grouping bars (default 5).
    slope_tolerance : float
        Max relative slope difference for parallelism (default 0.25).
    cooldown : int
        Minimum bars between signals (default 10).
    return_details : bool
        If True, return (df, details_list).
    min_containment : float
        Minimum fraction of bars inside the channel (default 0.60).

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, list[dict])
    """
    df = df.copy()
    atr = compute_atr(df, window=14)

    signals = pd.Series(None, index=df.index, dtype=object)
    details = [] if return_details else None
    bars_since_last = cooldown + 1

    max_lookback = backcandles + brange

    for i in range(max_lookback, len(df)):
        bars_since_last += 1
        atr_i = atr.iloc[i]
        if pd.isna(atr_i) or atr_i <= 0:
            continue
        if bars_since_last <= cooldown:
            continue

        # --- Dynamic window optimisation (reference notebook cell-10) ---
        # Try different lookback lengths, pick the tightest channel.
        best_dist = float('inf')
        best = None

        for bc in range(backcandles - brange, backcandles + brange + 1):
            if bc < wind * 3:
                continue
            start = i - bc
            if start < 0:
                continue

            highs = df["High"].values[start:i]
            lows = df["Low"].values[start:i]

            xxmax, maxim, xxmin, minim = chunk_extremes(highs, lows, wind=wind)

            if len(xxmax) < 2 or len(xxmin) < 2:
                continue

            # polyfit for slope (same as reference notebook)
            slmax, _ = np.polyfit(xxmax, maxim, 1)
            slmin, _ = np.polyfit(xxmin, minim, 1)

            # Adjusted intercepts to wrap candles
            adj_intercmax = float((maxim - slmax * xxmax).max())
            adj_intercmin = float((minim - slmin * xxmin).min())

            # Channel distance at current bar
            n = len(highs)
            upper_at_end = slmax * n + adj_intercmax
            lower_at_end = slmin * n + adj_intercmin
            dist = upper_at_end - lower_at_end

            if dist <= 0:
                continue
            if dist < best_dist:
                best_dist = dist
                best = {
                    "bc": bc, "start": start,
                    "slmax": slmax, "slmin": slmin,
                    "adj_intercmax": adj_intercmax,
                    "adj_intercmin": adj_intercmin,
                    "xxmax": xxmax, "maxim": maxim,
                    "xxmin": xxmin, "minim": minim,
                    "highs": highs, "lows": lows,
                }

        if best is None:
            continue

        sl_u = best["slmax"]
        sl_l = best["slmin"]
        ic_u = best["adj_intercmax"]
        ic_l = best["adj_intercmin"]
        highs = best["highs"]
        lows = best["lows"]
        start = best["start"]
        bc = best["bc"]

        # --- Parallelism check ---
        if abs(sl_u) < 1e-9:
            continue
        if sl_u * sl_l < 0:
            continue
        if abs(sl_u - sl_l) / abs(sl_u) > slope_tolerance:
            continue

        # --- Evaluate trendlines ---
        x = np.arange(len(highs))
        upper_line = sl_u * x + ic_u
        lower_line = sl_l * x + ic_l

        # --- Width check: 1–6× ATR ---
        channel_width = (upper_line - lower_line).mean()
        if channel_width < atr_i or channel_width > atr_i * 6.0:
            continue

        # --- Containment ---
        tol = 0.1 * atr_i
        cr = containment_ratio(highs, lows, upper_line, lower_line,
                               tolerance=tol)
        if cr < min_containment:
            continue

        # --- Signal: current price near a boundary ---
        n = len(highs)
        current_upper = sl_u * n + ic_u
        current_lower = sl_l * n + ic_l
        current_close = df["Close"].iloc[i]

        near_upper = abs(current_upper - current_close) < atr_i * 0.3
        near_lower = abs(current_close - current_lower) < atr_i * 0.3

        if near_upper or near_lower:
            signals.iloc[i] = "channel_up" if sl_u > 0 else "channel_down"
            if return_details:
                # Touch counting with 0.15 × ATR tolerance (tight to avoid false marks)
                touch_tol = 0.15 * atr_i
                upper_touches = count_touches(highs, upper_line, touch_tol, side="upper")
                lower_touches = count_touches(lows, lower_line, touch_tol, side="lower")

                details.append({
                    "event_date": df.index[i],
                    "pattern_type": signals.iloc[i],
                    "start_idx": start,
                    "end_idx": i,
                    "start_date": df.index[start],
                    "end_date": df.index[i],
                    "upper_slope": sl_u,
                    "upper_intercept": ic_u,
                    "lower_slope": sl_l,
                    "lower_intercept": ic_l,
                    "window": bc,
                    "containment_ratio": round(cr, 3),
                    "channel_width_atr": round(channel_width / atr_i, 2),
                    # Touch statistics
                    "upper_touches": upper_touches["touch_count"],
                    "lower_touches": lower_touches["touch_count"],
                    "upper_touch_indices": [start + j for j in upper_touches["touch_indices"]],
                    "lower_touch_indices": [start + j for j in lower_touches["touch_indices"]],
                    "upper_mean_error": round(upper_touches["mean_error"], 4),
                    "lower_mean_error": round(lower_touches["mean_error"], 4),
                    "upper_violations": upper_touches["violations"],
                    "lower_violations": lower_touches["violations"],
                    # Chunk extreme positions (absolute indices for plotting)
                    "swing_high_idx": [start + int(j) for j in best["xxmax"]],
                    "swing_high_prices": list(best["maxim"]),
                    "swing_low_idx": [start + int(j) for j in best["xxmin"]],
                    "swing_low_prices": list(best["minim"]),
                    "pivot_highs": len(best["xxmax"]),
                    "pivot_lows": len(best["xxmin"]),
                })
            bars_since_last = 0

    df["channel_pattern"] = signals
    if return_details:
        return df, details
    return df
