import numpy as np
import pandas as pd

from src.data.utils import compute_atr


def detect_channel(df, window=50, slope_tolerance=0.15, min_touches=3):
    """Detect price channels defined by two parallel trendlines.

    Fits linear trendlines to highs and lows over *window* bars.  A valid
    channel requires parallel slopes, meaningful width (1–6× ATR), price
    touching both bands, and current price near a boundary.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: High, Low, Close
    window : int
        Lookback window for the formation (default 50)
    slope_tolerance : float
        Max allowed slope difference as a fraction (default 0.15 = 15%)
    min_touches : int
        Minimum times price must touch each band (default 2)

    Returns
    -------
    pd.DataFrame
        Original df with added column 'channel_pattern':
        - 'channel_up'
        - 'channel_down'
        - None otherwise
    """
    df = df.copy()
    atr = compute_atr(df, window=14)

    signals = pd.Series(None, index=df.index, dtype=object)
    x = np.arange(window)

    for i in range(window, len(df)):
        atr_i = atr.iloc[i]
        if pd.isna(atr_i):
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

        # Channel width: must be meaningful (1–6× ATR)
        upper_line = np.polyval(high_coeffs, x)
        lower_line = np.polyval(low_coeffs, x)
        channel_width = (upper_line - lower_line).mean()

        if channel_width < atr_i or channel_width > atr_i * 6.0:
            continue

        # Touch count: price must have touched each band
        touch_band = atr_i * 0.3
        upper_touches = (highs > upper_line - touch_band).sum()
        lower_touches = (lows < lower_line + touch_band).sum()

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

    df["channel_pattern"] = signals
    return df
