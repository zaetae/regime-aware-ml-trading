import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.data.utils import compute_atr
from src.patterns.pivots import find_swing_highs, find_swing_lows, containment_ratio


def detect_channel(df, window=30, slope_tolerance=0.20, cooldown=10,
                   return_details=False, pivot_order=3, min_pivots=2,
                   min_r=0.85, min_containment=0.60):
    """Detect price channels using pivot + linregress (same approach as triangles).

    Pipeline per window:

    1. Find swing highs / lows (±pivot_order bars).
    2. ``linregress`` on pivot points → slope, intercept, *r*.
    3. Require |r| >= min_r on each trendline.
    4. Adjusted intercepts → upper caps all swing highs, lower floors all lows.
    5. Parallelism check: slopes same direction, within tolerance.
    6. Width between 1–6× ATR.
    7. Signal when current price is near a boundary.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: High, Low, Close
    window : int
        Lookback window (default 30).
    slope_tolerance : float
        Max relative slope difference (default 0.20 = 20%).
    cooldown : int
        Minimum bars between signals (default 10).
    return_details : bool
        If True, return (df, details_list).
    pivot_order : int
        Half-width for swing detection (default 3).
    min_pivots : int
        Minimum swing points per trendline (default 2).
    min_r : float
        Minimum |r| on each trendline (default 0.85).
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

        # --- Step 1: find swing highs / lows ---
        sh_idx = find_swing_highs(highs, order=pivot_order)
        sl_idx = find_swing_lows(lows, order=pivot_order)

        if len(sh_idx) < min_pivots or len(sl_idx) < min_pivots:
            continue

        sh_x = np.array(sh_idx, dtype=float)
        sh_y = highs[sh_idx]
        sl_x = np.array(sl_idx, dtype=float)
        sl_y = lows[sl_idx]

        # --- Step 2: linregress on pivots ---
        slope_upper, _, r_upper, _, _ = linregress(sh_x, sh_y)
        slope_lower, _, r_lower, _, _ = linregress(sl_x, sl_y)

        # --- Step 3: quality gate ---
        if abs(r_upper) < min_r or abs(r_lower) < min_r:
            continue

        # --- Step 4: parallelism — slopes same direction, similar magnitude ---
        if abs(slope_upper) < 1e-9:
            continue
        if slope_upper * slope_lower < 0:
            continue
        if abs(slope_upper - slope_lower) / abs(slope_upper) > slope_tolerance:
            continue

        # --- Step 5: adjusted intercepts (lines bound the pivots) ---
        adj_int_upper = float(np.max(sh_y - slope_upper * sh_x))
        adj_int_lower = float(np.min(sl_y - slope_lower * sl_x))

        high_coeffs = [slope_upper, adj_int_upper]
        low_coeffs = [slope_lower, adj_int_lower]

        x = np.arange(window)
        upper_line = np.polyval(high_coeffs, x)
        lower_line = np.polyval(low_coeffs, x)

        # --- Step 6: channel width 1–6× ATR ---
        channel_width = (upper_line - lower_line).mean()
        if channel_width < atr_i or channel_width > atr_i * 6.0:
            continue

        # --- Step 7: containment ---
        tol = 0.1 * atr_i
        cr = containment_ratio(highs, lows, upper_line, lower_line,
                               tolerance=tol)
        if cr < min_containment:
            continue

        # --- Signal: current price near a boundary ---
        current_upper = np.polyval(high_coeffs, window)
        current_lower = np.polyval(low_coeffs, window)
        current_close = df["Close"].iloc[i]

        near_upper = abs(current_upper - current_close) < atr_i * 0.3
        near_lower = abs(current_close - current_lower) < atr_i * 0.3

        if near_upper or near_lower:
            signals.iloc[i] = "channel_up" if slope_upper > 0 else "channel_down"
            if return_details:
                start = i - window
                details.append({
                    "event_date": df.index[i],
                    "pattern_type": signals.iloc[i],
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
                    "swing_high_idx": [start + int(j) for j in sh_idx],
                    "swing_high_prices": [float(highs[int(j)]) for j in sh_idx],
                    "swing_low_idx": [start + int(j) for j in sl_idx],
                    "swing_low_prices": [float(lows[int(j)]) for j in sl_idx],
                })
            bars_since_last = 0

    df["channel_pattern"] = signals
    if return_details:
        return df, details
    return df
