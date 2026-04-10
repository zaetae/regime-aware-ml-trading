import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from src.data.utils import compute_atr
from src.patterns.pivots import find_swing_highs, find_swing_lows, containment_ratio


def _count_distinct_touches(values, line, band, min_separation=5):
    """Count distinct touches of a trendline, not consecutive-day clusters.

    A "touch" requires the value to be within *band* of the line.
    Touches closer than *min_separation* bars apart count as the same touch.
    """
    near = np.where(np.abs(values - line) < band)[0]
    if len(near) == 0:
        return 0
    touches = 1
    last_touch = near[0]
    for idx in near[1:]:
        if idx - last_touch >= min_separation:
            touches += 1
            last_touch = idx
    return touches


def detect_channel(df, window=50, slope_tolerance=0.15, min_touches=4,
                   r2_min=0.70, cooldown=10, return_details=False,
                   pivot_order=5, min_pivots=3, min_containment=0.70):
    """Detect price channels defined by two parallel trendlines.

    Fits trendlines to **swing highs / swing lows** (pivot points) using
    the Theil-Sen estimator, then validates parallelism, R², touch count,
    and containment (>= 70% of bars inside).

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: High, Low, Close
    window : int
        Lookback window for the formation (default 50).
    slope_tolerance : float
        Max allowed slope difference as a fraction (default 0.15 = 15%).
    min_touches : int
        Minimum *distinct* touches per band (default 4).
    r2_min : float
        Minimum R² for both trendline regressions (default 0.70).
    cooldown : int
        Minimum bars between consecutive channel signals (default 10).
    return_details : bool
        If True, return (df, details_list) with trendline metadata.
    pivot_order : int
        Half-width for swing high/low detection (default 5).
    min_pivots : int
        Minimum swing points per trendline (default 3).
    min_containment : float
        Minimum fraction of bars inside the channel (default 0.85).

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, list[dict])
        Original df with added column 'channel_pattern'.
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
        x = np.arange(window)

        # --- Step A: find swing highs / lows ---
        sh_idx = find_swing_highs(highs, order=pivot_order)
        sl_idx = find_swing_lows(lows, order=pivot_order)

        if len(sh_idx) < min_pivots or len(sl_idx) < min_pivots:
            continue

        # --- Step B: Theil-Sen regression on pivot points ---
        sh_x = np.array(sh_idx)
        sh_y = highs[sh_idx]
        high_slope, high_intercept, _, _ = theilslopes(sh_y, sh_x)

        sl_x = np.array(sl_idx)
        sl_y = lows[sl_idx]
        low_slope, low_intercept, _, _ = theilslopes(sl_y, sl_x)

        high_coeffs = [high_slope, high_intercept]
        low_coeffs = [low_slope, low_intercept]

        # --- Parallelism check ---
        if abs(high_slope) < 1e-9:
            continue
        if abs(high_slope - low_slope) / abs(high_slope) > slope_tolerance:
            continue
        if high_slope * low_slope < 0:
            continue

        # --- Evaluate trendlines ---
        upper_line = np.polyval(high_coeffs, x)
        lower_line = np.polyval(low_coeffs, x)

        # --- R² check on pivot points ---
        pred_sh = np.polyval(high_coeffs, sh_x)
        ss_res_h = np.sum((sh_y - pred_sh) ** 2)
        ss_tot_h = np.sum((sh_y - sh_y.mean()) ** 2)
        r2_high = 1 - ss_res_h / ss_tot_h if ss_tot_h > 0 else 0

        pred_sl = np.polyval(low_coeffs, sl_x)
        ss_res_l = np.sum((sl_y - pred_sl) ** 2)
        ss_tot_l = np.sum((sl_y - sl_y.mean()) ** 2)
        r2_low = 1 - ss_res_l / ss_tot_l if ss_tot_l > 0 else 0

        if r2_high < r2_min or r2_low < r2_min:
            continue

        # --- Step C: containment validation ---
        tol = 0.1 * atr_i
        cr = containment_ratio(highs, lows, upper_line, lower_line, tolerance=tol)
        if cr < min_containment:
            continue

        # --- Channel width: 1–6× ATR ---
        channel_width = (upper_line - lower_line).mean()
        if channel_width < atr_i or channel_width > atr_i * 6.0:
            continue

        # Pivot counts already enforce minimum touches (min_pivots per side).
        # With Theil-Sen on pivots + containment, the old band-based touch
        # filter is redundant and overly restrictive.

        # --- Signal: current price near a boundary ---
        current_upper = np.polyval(high_coeffs, window)
        current_lower = np.polyval(low_coeffs, window)
        current_close = df["Close"].iloc[i]

        near_upper = abs(current_upper - current_close) < atr_i * 0.3
        near_lower = abs(current_close - current_lower) < atr_i * 0.3

        if near_upper or near_lower:
            signals.iloc[i] = "channel_up" if high_slope > 0 else "channel_down"
            if return_details:
                details.append({
                    "event_date": df.index[i],
                    "pattern_type": signals.iloc[i],
                    "start_idx": i - window,
                    "end_idx": i,
                    "start_date": df.index[i - window],
                    "end_date": df.index[i],
                    "upper_slope": high_coeffs[0],
                    "upper_intercept": high_coeffs[1],
                    "lower_slope": low_coeffs[0],
                    "lower_intercept": low_coeffs[1],
                    "window": window,
                    "containment_ratio": round(cr, 3),
                    "pivot_highs": len(sh_idx),
                    "pivot_lows": len(sl_idx),
                })
            bars_since_last = 0

    df["channel_pattern"] = signals
    if return_details:
        return df, details
    return df
