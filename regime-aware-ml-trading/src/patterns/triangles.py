import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.data.utils import compute_atr
from src.patterns.pivots import find_swing_highs, find_swing_lows, containment_ratio, count_touches


def detect_triangle_pattern(df, window=25, min_convergence_pct=0.05,
                            cooldown=10, return_details=False,
                            pivot_order=3, min_pivots=2, min_r=0.85):
    """Detect triangle patterns using the pivot + linregress approach.

    Closely follows the *TrianglePricePatterns* reference notebook:

    1. Identify swing highs / lows (±3-bar neighbourhood).
    2. ``linregress`` on pivot points → slope, intercept, *r*.
    3. Require ``|r| >= 0.9`` on each trendline (tight pivot alignment).
    4. Classify by slope signs: ascending / descending / symmetric.
    5. Fire signal when current bar breaks out of the recent range.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: High, Low, Close
    window : int
        Lookback window (default 20, matching the reference notebook).
    min_convergence_pct : float
        Minimum range compression (default 0.05 = 5%).
    cooldown : int
        Minimum bars between signals (default 10).
    return_details : bool
        If True, return (df, details_list) with trendline metadata.
    pivot_order : int
        Half-width for swing detection (default 3).
    min_pivots : int
        Minimum swing points per trendline (default 2).
    min_r : float
        Minimum |r| on each trendline (default 0.9).

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, list[dict])
    """
    df = df.copy()
    atr = compute_atr(df, window=14)

    signals = pd.Series(None, index=df.index, dtype=object)
    details = [] if return_details else None
    bars_since_last = cooldown + 1

    for i in range(window, len(df)):
        bars_since_last += 1
        atr_i = atr.iloc[i]
        if pd.isna(atr_i) or atr_i <= 0:
            continue
        if bars_since_last <= cooldown:
            continue

        window_slice = df.iloc[i - window: i]
        highs = window_slice["High"].values
        lows = window_slice["Low"].values

        # --- Step 1: find swing highs / lows (order=3) ---
        sh_idx = find_swing_highs(highs, order=pivot_order)
        sl_idx = find_swing_lows(lows, order=pivot_order)

        # Need at least min_pivots per side, and at least 3 total
        # (matching notebook: "if xxmax.size<3 and xxmin.size<3: continue")
        if len(sh_idx) < min_pivots or len(sl_idx) < min_pivots:
            continue
        if (len(sh_idx) + len(sl_idx)) < 3:
            continue

        sh_x, sh_y = np.array(sh_idx, dtype=float), highs[sh_idx]
        sl_x, sl_y = np.array(sl_idx, dtype=float), lows[sl_idx]

        # --- Step 2: linregress on pivots → slope, intercept, r ---
        # Exactly as the notebook: slmax, intercmax, rmax, ...
        slmax, _, rmax, _, _ = linregress(sh_x, sh_y)
        slmin, _, rmin, _, _ = linregress(sl_x, sl_y)

        # --- Step 3: quality gate (|r| >= 0.9, notebook default) ---
        if abs(rmax) < min_r or abs(rmin) < min_r:
            continue

        # Adjust intercepts so the lines BOUND the pivots (not cut through).
        # Upper line: sits on top of the highest swing high.
        # Lower line: sits below the lowest swing low.
        # This matches the reference notebook's adjintercmax/adjintercmin
        # approach and produces the classic triangle shape where candles
        # are INSIDE the two converging lines.
        adj_intercmax = float(np.max(sh_y - slmax * sh_x))
        adj_intercmin = float(np.min(sl_y - slmin * sl_x))
        high_coeffs = [slmax, adj_intercmax]
        low_coeffs = [slmin, adj_intercmin]

        x = np.arange(window)
        upper_line = np.polyval(high_coeffs, x)
        lower_line = np.polyval(low_coeffs, x)

        # --- Step 4: convergence check ---
        range_start = upper_line[0] - lower_line[0]
        range_end = upper_line[-1] - lower_line[-1]
        if range_start <= 0 or range_end < 0:
            continue
        compression = (range_start - range_end) / range_start
        if compression < min_convergence_pct:
            continue

        # --- Step 5: classify triangle type (ATR-normalised slopes) ---
        # Notebook uses absolute thresholds; we normalise by ATR for
        # robustness across price levels.
        flat_threshold = 0.1 * atr_i / window
        is_ascending = abs(slmax) < flat_threshold and slmin > flat_threshold
        is_descending = slmax < -flat_threshold and abs(slmin) < flat_threshold
        is_symmetric = slmax < -flat_threshold and slmin > flat_threshold

        if not (is_ascending or is_descending or is_symmetric):
            continue

        # Containment (informational — not a hard gate with tight lines)
        tol = 0.1 * atr_i
        cr = containment_ratio(highs, lows, upper_line, lower_line,
                               tolerance=tol)

        # --- Step 6a: breakout signal ---
        current_high = df["High"].iloc[i]
        current_low = df["Low"].iloc[i]
        recent_high = highs[-3:].max()
        recent_low = lows[-3:].min()

        breakout_up = current_high > recent_high + 0.3 * atr_i
        breakout_down = current_low < recent_low - 0.3 * atr_i

        if breakout_up or breakout_down:
            if is_ascending:
                signals.iloc[i] = "ascending_triangle"
            elif is_descending:
                signals.iloc[i] = "descending_triangle"
            else:
                signals.iloc[i] = "symmetric_triangle"
            if return_details:
                details.append(_make_detail(
                    df, i, signals.iloc[i], high_coeffs, low_coeffs,
                    window, cr, rmax, rmin, sh_idx, sl_idx, atr_i,
                ))
            bars_since_last = 0
            continue

        # --- Step 6b: descending triangle upper-limit test ---
        if is_descending:
            upper_at_current = np.polyval(high_coeffs, window)
            current_close = df["Close"].iloc[i]
            if abs(current_close - upper_at_current) < 0.3 * atr_i:
                signals.iloc[i] = "desc_triangle_upper_test"
                if return_details:
                    details.append(_make_detail(
                        df, i, "desc_triangle_upper_test",
                        high_coeffs, low_coeffs,
                        window, cr, rmax, rmin, sh_idx, sl_idx,
                    ))
                bars_since_last = 0

    df["triangle_pattern"] = signals
    if return_details:
        return df, details
    return df


def _make_detail(df, i, pattern_type, high_coeffs, low_coeffs,
                 window, cr, r_upper, r_lower, sh_idx, sl_idx, atr_i):
    """Build a metadata dict for one detection, including touch statistics."""
    start = i - window
    highs = df["High"].values[start:i]
    lows = df["Low"].values[start:i]

    x = np.arange(window)
    upper_line = np.polyval(high_coeffs, x)
    lower_line = np.polyval(low_coeffs, x)

    # Touch counting with 0.15 × ATR tolerance (tight to avoid false marks)
    touch_tol = 0.15 * atr_i
    upper_touches = count_touches(highs, upper_line, touch_tol, side="upper")
    lower_touches = count_touches(lows, lower_line, touch_tol, side="lower")

    abs_highs = df["High"].values
    abs_lows = df["Low"].values
    return {
        "event_date": df.index[i],
        "pattern_type": pattern_type,
        "start_idx": start,
        "end_idx": i,
        "start_date": df.index[start],
        "end_date": df.index[i],
        "upper_slope": high_coeffs[0],
        "upper_intercept": high_coeffs[1],
        "lower_slope": low_coeffs[0],
        "lower_intercept": low_coeffs[1],
        "window": window,
        "containment_ratio": round(cr, 3),
        "r_upper": round(abs(r_upper), 3),
        "r_lower": round(abs(r_lower), 3),
        "pivot_highs": len(sh_idx),
        "pivot_lows": len(sl_idx),
        # Touch statistics
        "upper_touches": upper_touches["touch_count"],
        "lower_touches": lower_touches["touch_count"],
        "upper_touch_indices": [start + j for j in upper_touches["touch_indices"]],
        "lower_touch_indices": [start + j for j in lower_touches["touch_indices"]],
        "upper_mean_error": round(upper_touches["mean_error"], 4),
        "lower_mean_error": round(lower_touches["mean_error"], 4),
        "upper_violations": upper_touches["violations"],
        "lower_violations": lower_touches["violations"],
        # Pivot positions (absolute indices)
        "swing_high_idx": [start + int(j) for j in sh_idx],
        "swing_high_prices": [float(abs_highs[start + int(j)]) for j in sh_idx],
        "swing_low_idx": [start + int(j) for j in sl_idx],
        "swing_low_prices": [float(abs_lows[start + int(j)]) for j in sl_idx],
    }
