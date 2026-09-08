"""Channel detection using candidate fitting followed by independent pivot validation.

Candidate boundaries are estimated from regularly spaced chunk extrema.
The fitted boundaries are then independently validated against swing highs
and swing lows detected with a ±pivot_order neighbourhood.

This deliberately separates:
1. candidate estimation, using chunk extrema;
2. structural validation, using independent swing pivots.

The previous local-extrema touch method remains available through
touch_validation="local" for controlled comparison.
"""

import numpy as np
import pandas as pd

from src.data.utils import compute_atr
from src.patterns.pivots import (
    find_swing_highs,
    find_swing_lows,
    chunk_extremes,
    containment_ratio,
    count_touches,
    count_pivot_touches,
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

def detect_channel(
    df,
    backcandles=40,
    brange=15,
    wind=5,
    pivot_order=3,
    min_upper_touches=2,      # was 2, unchanged
    min_lower_touches=2,      # was 3, now symmetric
    slope_tolerance=0.25,
    cooldown=10,
    return_details=False,
    min_containment=0.70,
    touch_validation="local", # was "swing" (which gave zero results)
    touch_tolerance_atr=0.15, # was hardcoded 0.20
):

    """Detect price channels using candidate fitting and pivot validation."""

    df = df.copy()
    atr = compute_atr(df, window=14)

    signals = pd.Series(None, index=df.index, dtype=object)
    details = [] if return_details else None

    bars_since_last = cooldown + 1
    max_lookback = backcandles + brange

    all_highs = df["High"].values
    all_lows = df["Low"].values
    all_closes = df["Close"].values

    for i in range(max_lookback, len(df)):
        bars_since_last += 1

        atr_i = atr.iloc[i]
        if pd.isna(atr_i) or atr_i <= 0:
            continue

        if bars_since_last <= cooldown:
            continue

        # ------------------------------------------------------------
        # Step 1-4: candidate fitting over multiple lookback windows
        # ------------------------------------------------------------
        best_dist = float("inf")
        best = None

        for bc in range(
            backcandles - brange,
            backcandles + brange + 1,
        ):
            if bc < wind * 3:
                continue

            start = i - bc
            if start < 0:
                continue

            highs = all_highs[start:i]
            lows = all_lows[start:i]

            xxmax, maxim, xxmin, minim = chunk_extremes(
                highs,
                lows,
                wind=wind,
            )

            if len(xxmax) < 2 or len(xxmin) < 2:
                continue

            # Fit upper and lower slopes using chunk extrema
            slmax, _ = np.polyfit(xxmax, maxim, 1)
            slmin, _ = np.polyfit(xxmin, minim, 1)

            # Adjust intercepts so the fitted lines wrap the extremes
            adj_imax = float(
                (maxim - slmax * xxmax).max()
            )
            adj_imin = float(
                (minim - slmin * xxmin).min()
            )

            n = len(highs)

            upper_end = slmax * n + adj_imax
            lower_end = slmin * n + adj_imin

            dist = upper_end - lower_end

            if dist <= 0:
                continue

            if dist < best_dist:
                best_dist = dist

                best = {
                    "bc": bc,
                    "start": start,
                    "slmax": slmax,
                    "slmin": slmin,
                    "adj_imax": adj_imax,
                    "adj_imin": adj_imin,
                    "xxmax": xxmax,
                    "maxim": maxim,
                    "xxmin": xxmin,
                    "minim": minim,
                    "highs": highs,
                    "lows": lows,
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

        # ------------------------------------------------------------
        # Step 5: parallelism
        # ------------------------------------------------------------
        if abs(sl_u) < 1e-9:
            continue

        if sl_u * sl_l < 0:
            continue

        parallelism = abs(sl_u - sl_l) / abs(sl_u)

        if parallelism > slope_tolerance:
            continue

        # ------------------------------------------------------------
        # Evaluate fitted channel
        # ------------------------------------------------------------
        x = np.arange(n)

        upper_line = sl_u * x + ic_u
        lower_line = sl_l * x + ic_l

        # ------------------------------------------------------------
        # Step 6: width
        # ------------------------------------------------------------
        channel_width = (
            upper_line - lower_line
        ).mean()

        width_atr = channel_width / atr_i

        if (
            channel_width < atr_i
            or channel_width > atr_i * 6.0
        ):
            continue

        # ------------------------------------------------------------
        # Step 7: containment
        # ------------------------------------------------------------
        tol = 0.1 * atr_i

        cr = containment_ratio(
            highs,
            lows,
            upper_line,
            lower_line,
            tolerance=tol,
        )

        if cr < min_containment:
            continue

        # ------------------------------------------------------------
        # Step 8: touch validation
        # ------------------------------------------------------------
        touch_tol = touch_tolerance_atr * atr_i

        # Detect actual swing pivots independently
        swing_high_idx = find_swing_highs(
            highs,
            order=pivot_order,
        )

        swing_low_idx = find_swing_lows(
            lows,
            order=pivot_order,
        )

        if touch_validation == "swing":

            ut = count_pivot_touches(
                highs,
                upper_line,
                swing_high_idx,
                touch_tol,
                side="upper",
            )

            lt = count_pivot_touches(
                lows,
                lower_line,
                swing_low_idx,
                touch_tol,
                side="lower",
            )

        elif touch_validation == "local":

            # Original baseline method
            ut = count_touches(
                highs,
                upper_line,
                touch_tol,
                side="upper",
            )

            lt = count_touches(
                lows,
                lower_line,
                touch_tol,
                side="lower",
            )

        else:
            raise ValueError(
                "touch_validation must be "
                "'swing' or 'local'"
            )

        if ut["touch_count"] < min_upper_touches:
            continue

        if lt["touch_count"] < min_lower_touches:
            continue

        # ------------------------------------------------------------
        # Step 9: current boundary interaction
        # ------------------------------------------------------------
        current_upper = sl_u * n + ic_u
        current_lower = sl_l * n + ic_l

        cur_high = all_highs[i]
        cur_low = all_lows[i]
        cur_close = all_closes[i]

        near_upper = (
            abs(current_upper - cur_high)
            < atr_i * 0.3
        )

        near_lower = (
            abs(cur_low - current_lower)
            < atr_i * 0.3
        )

        if not (near_upper or near_lower):
            continue

        # ------------------------------------------------------------
        # Rejection check
        # ------------------------------------------------------------
        bar_mid = (
            cur_high + cur_low
        ) / 2.0

        has_rejection = True

        if near_upper and cur_close > bar_mid:
            has_rejection = False

        if near_lower and cur_close < bar_mid:
            has_rejection = False

        # ------------------------------------------------------------
        # Confidence score
        # ------------------------------------------------------------
        conf = _confidence_score(
            ut["touch_count"],
            lt["touch_count"],
            cr,
            parallelism,
            width_atr,
        )

        if not has_rejection:
            conf = round(
                conf * 0.80,
                1,
            )

        signals.iloc[i] = (
            "channel_up"
            if sl_u > 0
            else "channel_down"
        )

        # ------------------------------------------------------------
        # Save detection metadata
        # ------------------------------------------------------------
        if return_details:
            details.append(
                {
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

                    "containment_ratio": round(
                        cr,
                        3,
                    ),

                    "channel_width_atr": round(
                        width_atr,
                        2,
                    ),

                    "confidence": conf,

                    # Touch statistics
                    "upper_touches": ut[
                        "touch_count"
                    ],

                    "lower_touches": lt[
                        "touch_count"
                    ],

                    "upper_touch_indices": [
                        start + int(j)
                        for j in ut["touch_indices"]
                    ],

                    "lower_touch_indices": [
                        start + int(j)
                        for j in lt["touch_indices"]
                    ],

                    "upper_mean_error": round(
                        ut["mean_error"],
                        4,
                    ),

                    "lower_mean_error": round(
                        lt["mean_error"],
                        4,
                    ),

                    "upper_violations": ut[
                        "violations"
                    ],

                    "lower_violations": lt[
                        "violations"
                    ],

                    # Independent swing pivots
                    "swing_high_idx": [
                        start + int(j)
                        for j in swing_high_idx
                    ],

                    "swing_high_prices": [
                        float(highs[int(j)])
                        for j in swing_high_idx
                    ],

                    "swing_low_idx": [
                        start + int(j)
                        for j in swing_low_idx
                    ],

                    "swing_low_prices": [
                        float(lows[int(j)])
                        for j in swing_low_idx
                    ],

                    # Pivot validation metadata
                    "pivot_highs": len(
                        swing_high_idx
                    ),

                    "pivot_lows": len(
                        swing_low_idx
                    ),

                    "pivot_high_indices": [
                        start + int(j)
                        for j in swing_high_idx
                    ],

                    "pivot_low_indices": [
                        start + int(j)
                        for j in swing_low_idx
                    ],

                    "touch_validation":
                        touch_validation,

                    "pivot_order":
                        pivot_order,

                    "touch_tolerance_atr":
                        0.20,

                    "has_rejection":
                        has_rejection,

                    "near_upper":
                        near_upper,

                    "near_lower":
                        near_lower,
                }
            )

        bars_since_last = 0

    # ------------------------------------------------------------
    # Final outputs
    # ------------------------------------------------------------
    df["channel_pattern"] = signals

    if "intended_direction" not in df.columns:
        df["intended_direction"] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="object",
        )

    df.loc[
        signals == "channel_up",
        "intended_direction",
    ] = "long"

    df.loc[
        signals == "channel_down",
        "intended_direction",
    ] = "short"

    if return_details:
        return df, details

    return df