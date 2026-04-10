import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.data.utils import compute_atr
from src.patterns.pivots import find_swing_highs, find_swing_lows, containment_ratio


def detect_triangle_pattern(df, window=50, min_convergence_pct=0.05,
                            cooldown=10, return_details=False,
                            pivot_order=3, min_pivots=2,
                            min_r=0.85, min_containment=0.70):
    """Detect triangle breakouts and upper-limit tests.

    Uses the pivot-point + linregress approach (cf. *TrianglePricePatterns*
    notebook) with envelope intercept adjustment and containment validation.

    Pipeline per window:

    1. Identify swing highs / lows (``pivot_order`` = 3).
    2. ``linregress`` on pivot points → slope + correlation *r*.
    3. Require ``|r| >= min_r`` on each trendline (pivot alignment quality).
    4. Shift intercepts to create an **envelope** (upper line caps all swing
       highs, lower line floors all swing lows).
    5. Validate containment >= ``min_containment``.
    6. Check convergence and classify triangle type.
    7. Fire breakout or upper-limit-test signal.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: High, Low, Close
    window : int
        Lookback window for the formation (default 50).
    min_convergence_pct : float
        Minimum range compression required (default 0.05 = 5%).
    cooldown : int
        Minimum bars between consecutive signals (default 10).
    return_details : bool
        If True, return (df, details_list) with trendline metadata.
    pivot_order : int
        Half-width for swing detection (default 3, per reference notebook).
    min_pivots : int
        Minimum swing points per trendline (default 2).
    min_r : float
        Minimum |correlation coefficient| on each trendline (default 0.85).
    min_containment : float
        Minimum fraction of bars inside the triangle (default 0.70).

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, list[dict])
        Original df with added column 'triangle_pattern'.
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

        # --- Step 1: find swing highs / lows (order=3, per reference) ---
        sh_idx = find_swing_highs(highs, order=pivot_order)
        sl_idx = find_swing_lows(lows, order=pivot_order)

        if len(sh_idx) < min_pivots or len(sl_idx) < min_pivots:
            continue

        sh_x = np.array(sh_idx)
        sh_y = highs[sh_idx]
        sl_x = np.array(sl_idx)
        sl_y = lows[sl_idx]

        # --- Step 2: linregress on pivots → slope + r ---
        slope_upper, _, r_upper, _, _ = linregress(sh_x, sh_y)
        slope_lower, _, r_lower, _, _ = linregress(sl_x, sl_y)

        # --- Step 3: quality gate — pivots must align tightly ---
        if abs(r_upper) < min_r or abs(r_lower) < min_r:
            continue

        # --- Step 4: envelope intercepts ---
        # Shift upper line UP so it caps ALL swing highs.
        # Shift lower line DOWN so it floors ALL swing lows.
        intercept_upper = float(np.max(sh_y - slope_upper * sh_x))
        intercept_lower = float(np.min(sl_y - slope_lower * sl_x))

        high_coeffs = [slope_upper, intercept_upper]
        low_coeffs = [slope_lower, intercept_lower]

        upper_line = np.polyval(high_coeffs, x)
        lower_line = np.polyval(low_coeffs, x)

        # --- Step 5: containment validation ---
        tol = 0.1 * atr_i
        cr = containment_ratio(highs, lows, upper_line, lower_line,
                               tolerance=tol)
        if cr < min_containment:
            continue

        # --- Step 6: convergence check ---
        range_start = upper_line[:5].mean() - lower_line[:5].mean()
        range_end = upper_line[-5:].mean() - lower_line[-5:].mean()

        if range_start <= 0:
            continue

        compression = (range_start - range_end) / range_start
        if compression < min_convergence_pct:
            continue

        # --- Classify triangle type (ATR-normalised slopes) ---
        flat_threshold = 0.1 * atr_i / window
        is_ascending = slope_upper < flat_threshold and slope_lower > flat_threshold
        is_descending = (slope_upper < -flat_threshold
                         and slope_lower > -flat_threshold)
        is_symmetric = (slope_upper < -flat_threshold
                        and slope_lower > flat_threshold)

        if not (is_ascending or is_descending or is_symmetric):
            continue

        # --- Step 7a: breakout signal ---
        current_high = df["High"].iloc[i]
        current_low = df["Low"].iloc[i]
        recent_high = highs[-5:].max()
        recent_low = lows[-5:].min()

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
                    window, cr, r_upper, r_lower, sh_idx, sl_idx,
                ))
            bars_since_last = 0
            continue

        # --- Step 7b: descending triangle upper-limit test ---
        if is_descending:
            upper_at_current = np.polyval(high_coeffs, window)
            current_close = df["Close"].iloc[i]
            if abs(current_close - upper_at_current) < 0.3 * atr_i:
                signals.iloc[i] = "desc_triangle_upper_test"
                if return_details:
                    details.append(_make_detail(
                        df, i, "desc_triangle_upper_test",
                        high_coeffs, low_coeffs,
                        window, cr, r_upper, r_lower, sh_idx, sl_idx,
                    ))
                bars_since_last = 0

    df["triangle_pattern"] = signals
    if return_details:
        return df, details
    return df


def _make_detail(df, i, pattern_type, high_coeffs, low_coeffs,
                 window, cr, r_upper, r_lower, sh_idx, sl_idx):
    """Build a metadata dict for one detection."""
    return {
        "event_date": df.index[i],
        "pattern_type": pattern_type,
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
        "r_upper": round(abs(r_upper), 3),
        "r_lower": round(abs(r_lower), 3),
        "pivot_highs": len(sh_idx),
        "pivot_lows": len(sl_idx),
    }
