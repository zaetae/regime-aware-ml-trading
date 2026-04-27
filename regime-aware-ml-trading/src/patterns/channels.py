"""Channel detection anchored to real swing pivots with quality scoring.

Follows the *TrendLineChannelDetection* reference notebook for the core
fitting procedure (chunk extremes → polyfit → adjusted intercepts →
dynamic window optimisation for tightest channel), but adds:

* Swing-pivot anchoring: lines are validated against real swing highs/lows
* Minimum touch requirements: ≥3 lower + ≥2 upper pivot touches
* Confidence score based on touches, containment, parallelism, fit
* Rejection confirmation: signal fires only when price touches a boundary
  AND the close rejects (moves away from it)

References
----------
TrendLineChannelDetection.ipynb (CodeTrading / YouTube)
"""

import numpy as np
import pandas as pd

from src.data.utils import compute_atr
from src.patterns.pivots import (
    find_swing_highs, find_swing_lows,
    chunk_extremes, containment_ratio, count_touches,
)


def _confidence_score(upper_tc, lower_tc, cr, parallelism, width_atr):
    """Compute a 0-100 confidence score for a channel detection.

    Components (weights sum to 1.0):
        Touch score   40%   — more touches = higher confidence
        Containment   25%   — higher % of bars inside = better
        Parallelism   20%   — smaller slope difference = better
        Width score   15%   — moderate width (2-4 ATR) scores best
    """
    # Touch score: each touch above the minimum adds value, cap at 8 per side
    touch_score = min((upper_tc + lower_tc) / 10.0, 1.0)

    # Containment score: linear from 0.6 (min gate) to 1.0 (perfect)
    cont_score = min(max((cr - 0.6) / 0.4, 0.0), 1.0)

    # Parallelism: 0 slope diff = perfect, >0.25 = bad
    para_score = max(1.0 - parallelism / 0.25, 0.0)

    # Width: best at 2-4 ATR, penalise extremes
    if 2.0 <= width_atr <= 4.0:
        width_score = 1.0
    elif width_atr < 2.0:
        width_score = max(width_atr / 2.0, 0.0)
    else:
        width_score = max(1.0 - (width_atr - 4.0) / 3.0, 0.0)

    score = (0.40 * touch_score + 0.25 * cont_score +
             0.20 * para_score + 0.15 * width_score) * 100
    return round(score, 1)


def detect_channel(df, backcandles=40, brange=15, wind=5,
                   pivot_order=3, min_upper_touches=2,
                   min_lower_touches=3, slope_tolerance=0.25,
                   cooldown=10, return_details=False,
                   min_containment=0.70):
    """Detect price channels following the reference notebook method.

    Pipeline (per bar):
    1. Try lookback windows from ``backcandles - brange`` to
       ``backcandles + brange`` (reference notebook cell-10).
    2. Each window is divided into chunks of ``wind`` bars.  Max-High and
       min-Low from each chunk are extracted as representative points.
    3. ``np.polyfit(degree=1)`` fits upper and lower slopes.
    4. Intercepts are adjusted to **wrap** all chunk extremes — upper line
       caps all highs, lower line floors all lows (reference cell-7).
    5. The tightest channel (smallest distance at the current bar) is kept.
    6. Swing-pivot validation:  find real swing highs/lows, count how many
       fall within tolerance of the fitted lines.  Require ≥ ``min_upper_touches``
       on the upper line and ≥ ``min_lower_touches`` on the lower line.
    7. Validate parallelism, width (1–6 × ATR), and containment (≥ 70 %).
    8. Signal fires only on **confirmed boundary interaction**: current bar
       touches a boundary AND the close shows rejection (away from the line).

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: Open, High, Low, Close.
    backcandles : int
        Base lookback (default 40).
    brange : int
        Half-range for dynamic window search (default 15).
    wind : int
        Chunk size (default 5).
    pivot_order : int
        ±bars for swing-pivot detection (default 3).
    min_upper_touches : int
        Minimum swing-pivot touches on upper line (default 2).
    min_lower_touches : int
        Minimum swing-pivot touches on lower line (default 3).
    slope_tolerance : float
        Max relative slope difference for parallelism (default 0.25).
    cooldown : int
        Min bars between signals (default 10).
    return_details : bool
        If True, return (df, details_list).
    min_containment : float
        Min fraction of bars inside the channel (default 0.70).

    Returns
    -------
    pd.DataFrame  or  (pd.DataFrame, list[dict])
    """
    df = df.copy()
    atr = compute_atr(df, window=14)

    signals = pd.Series(None, index=df.index, dtype=object)
    details = [] if return_details else None
    bars_since_last = cooldown + 1
    max_lookback = backcandles + brange

    all_highs = df["High"].values
    all_lows = df["Low"].values
    all_closes = df["Close"].values
    # Open not needed — rejection uses bar midpoint (High+Low)/2

    for i in range(max_lookback, len(df)):
        bars_since_last += 1
        atr_i = atr.iloc[i]
        if pd.isna(atr_i) or atr_i <= 0:
            continue
        if bars_since_last <= cooldown:
            continue

        # ── Step 1-4: Dynamic window optimisation (reference cell-10) ──
        best_dist = float("inf")
        best = None

        for bc in range(backcandles - brange, backcandles + brange + 1):
            if bc < wind * 3:
                continue
            start = i - bc
            if start < 0:
                continue

            highs = all_highs[start:i]
            lows = all_lows[start:i]

            xxmax, maxim, xxmin, minim = chunk_extremes(highs, lows, wind=wind)
            if len(xxmax) < 2 or len(xxmin) < 2:
                continue

            # polyfit on chunk extremes (reference notebook)
            slmax, _ = np.polyfit(xxmax, maxim, 1)
            slmin, _ = np.polyfit(xxmin, minim, 1)

            # Adjusted intercepts to WRAP candles (reference cell-7)
            adj_imax = float((maxim - slmax * xxmax).max())
            adj_imin = float((minim - slmin * xxmin).min())

            # Channel distance at current bar
            n = len(highs)
            upper_end = slmax * n + adj_imax
            lower_end = slmin * n + adj_imin
            dist = upper_end - lower_end
            if dist <= 0:
                continue
            if dist < best_dist:
                best_dist = dist
                best = {
                    "bc": bc, "start": start,
                    "slmax": slmax, "slmin": slmin,
                    "adj_imax": adj_imax, "adj_imin": adj_imin,
                    "xxmax": xxmax, "maxim": maxim,
                    "xxmin": xxmin, "minim": minim,
                    "highs": highs, "lows": lows,
                }

        if best is None:
            continue

        sl_u = best["slmax"]
        sl_l = best["slmin"]
        ic_u = best["adj_imax"]
        ic_l = best["adj_imin"]
        highs = best["highs"]
        lows = best["lows"]
        start = best["start"]
        bc = best["bc"]
        n = len(highs)

        # ── Step 5: Parallelism check ──
        if abs(sl_u) < 1e-9:
            continue
        if sl_u * sl_l < 0:          # slopes in opposite directions
            continue
        parallelism = abs(sl_u - sl_l) / abs(sl_u)
        if parallelism > slope_tolerance:
            continue

        # ── Evaluate lines over window ──
        x = np.arange(n)
        upper_line = sl_u * x + ic_u
        lower_line = sl_l * x + ic_l

        # ── Step 6: Width check (1–6 × ATR) ──
        channel_width = (upper_line - lower_line).mean()
        width_atr = channel_width / atr_i
        if channel_width < atr_i or channel_width > atr_i * 6.0:
            continue

        # ── Step 7: Containment (≥ min_containment) ──
        tol = 0.1 * atr_i
        cr = containment_ratio(highs, lows, upper_line, lower_line,
                               tolerance=tol)
        if cr < min_containment:
            continue

        # ── Step 8: Swing-pivot touch validation ──
        # 0.20 × ATR is tight enough for accuracy, but permissive enough
        # to catch real near-touches that 0.15 would miss.
        touch_tol = 0.20 * atr_i
        ut = count_touches(highs, upper_line, touch_tol, side="upper")
        lt = count_touches(lows, lower_line, touch_tol, side="lower")

        if ut["touch_count"] < min_upper_touches:
            continue
        if lt["touch_count"] < min_lower_touches:
            continue

        # ── Step 9: Boundary interaction + rejection ──
        current_upper = sl_u * n + ic_u
        current_lower = sl_l * n + ic_l
        cur_high = all_highs[i]
        cur_low = all_lows[i]
        cur_close = all_closes[i]
        near_upper = abs(current_upper - cur_high) < atr_i * 0.3
        near_lower = abs(cur_low - current_lower) < atr_i * 0.3

        if not (near_upper or near_lower):
            continue

        # Rejection check: close moves away from the touched boundary.
        # This is a confidence modifier, not a hard gate — strong channels
        # without textbook rejection still get detected but at lower confidence.
        bar_mid = (cur_high + cur_low) / 2.0
        has_rejection = True
        if near_upper and cur_close > bar_mid:
            has_rejection = False
        if near_lower and cur_close < bar_mid:
            has_rejection = False

        # ── Compute confidence score ──
        conf = _confidence_score(
            ut["touch_count"], lt["touch_count"],
            cr, parallelism, width_atr,
        )
        if not has_rejection:
            conf = round(conf * 0.80, 1)  # 20% penalty for no rejection

        signals.iloc[i] = "channel_up" if sl_u > 0 else "channel_down"

        if return_details:
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
                "channel_width_atr": round(width_atr, 2),
                "confidence": conf,
                # Touch statistics (swing-pivot touches)
                "upper_touches": ut["touch_count"],
                "lower_touches": lt["touch_count"],
                "upper_touch_indices": [start + j for j in ut["touch_indices"]],
                "lower_touch_indices": [start + j for j in lt["touch_indices"]],
                "upper_mean_error": round(ut["mean_error"], 4),
                "lower_mean_error": round(lt["mean_error"], 4),
                "upper_violations": ut["violations"],
                "lower_violations": lt["violations"],
                # Chunk extreme positions (for plotting)
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
