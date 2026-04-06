import numpy as np
import pandas as pd

from src.data.utils import compute_atr


def _count_distinct_touches(values, line, band, min_separation=5):
    """Count distinct touches of a trendline, not consecutive-day clusters.

    A "touch" requires the value to be within *band* of the line.
    Touches closer than *min_separation* bars apart count as the same touch.
    This prevents 10 consecutive bars near the upper band from counting
    as 10 touches — they count as 1.

    Parameters
    ----------
    values : np.ndarray
        Highs (for upper line) or Lows (for lower line).
    line : np.ndarray
        Trendline values (same length as values).
    band : float
        Proximity threshold (e.g. 0.3 × ATR).
    min_separation : int
        Minimum bars between distinct touches.

    Returns
    -------
    int
        Number of distinct touches.
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
                   r2_min=0.70, cooldown=10, return_details=False):
    """Detect price channels defined by two parallel trendlines.

    Fits linear trendlines to highs and lows over *window* bars.  A valid
    channel requires parallel slopes, meaningful width, price touching both
    bands with distinct reversals, and good regression fit (R²).

    Improvements over the naive version:
    1. **R² check** — the regression must explain ≥70% of variance in both
       highs and lows, ensuring price actually follows a channel structure.
    2. **Distinct touch counting** — touches separated by ≥5 bars count as
       one, preventing inflated counts from consecutive bars near a line.
    3. **Cooldown** — after a signal fires, suppress for *cooldown* bars.

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
        If True, return (df, details_list) where details_list contains
        a metadata dict per detection with trendline coefficients.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, list[dict])
        Original df with added column 'channel_pattern'.
        When return_details=True, also returns a list of detail dicts.
    """
    df = df.copy()
    atr = compute_atr(df, window=14)

    signals = pd.Series(None, index=df.index, dtype=object)
    details = [] if return_details else None
    x = np.arange(window)
    bars_since_last = cooldown + 1  # allow first signal

    for i in range(window, len(df)):
        bars_since_last += 1
        atr_i = atr.iloc[i]
        if pd.isna(atr_i):
            continue

        # Cooldown: skip if too soon after last signal
        if bars_since_last <= cooldown:
            continue

        window_slice = df.iloc[i - window : i]
        highs = window_slice["High"].values
        lows = window_slice["Low"].values

        # Fit trendlines to highs and lows
        high_coeffs = np.polyfit(x, highs, 1)
        low_coeffs = np.polyfit(x, lows, 1)

        high_slope = high_coeffs[0]
        low_slope = low_coeffs[0]

        # Parallelism: slopes must be similar in magnitude and direction
        if abs(high_slope) < 1e-9:
            continue
        if abs(high_slope - low_slope) / abs(high_slope) > slope_tolerance:
            continue
        if high_slope * low_slope < 0:
            continue

        # R² check: trendlines must actually fit the data well
        upper_line = np.polyval(high_coeffs, x)
        lower_line = np.polyval(low_coeffs, x)

        ss_res_high = np.sum((highs - upper_line) ** 2)
        ss_tot_high = np.sum((highs - highs.mean()) ** 2)
        r2_high = 1 - ss_res_high / ss_tot_high if ss_tot_high > 0 else 0

        ss_res_low = np.sum((lows - lower_line) ** 2)
        ss_tot_low = np.sum((lows - lows.mean()) ** 2)
        r2_low = 1 - ss_res_low / ss_tot_low if ss_tot_low > 0 else 0

        if r2_high < r2_min or r2_low < r2_min:
            continue

        # Channel width: must be meaningful (1–6× ATR)
        channel_width = (upper_line - lower_line).mean()
        if channel_width < atr_i or channel_width > atr_i * 6.0:
            continue

        # Distinct touch count: touches ≥5 bars apart count as separate
        touch_band = atr_i * 0.3
        upper_touches = _count_distinct_touches(highs, upper_line, touch_band)
        lower_touches = _count_distinct_touches(lows, lower_line, touch_band)

        if upper_touches < min_touches or lower_touches < min_touches:
            continue

        # Only flag when current price is near a channel boundary
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
                })
            bars_since_last = 0

    df["channel_pattern"] = signals
    if return_details:
        return df, details
    return df
