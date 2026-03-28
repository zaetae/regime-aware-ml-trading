import numpy as np
import pandas as pd

from src.data.utils import compute_atr


def detect_triangle_pattern(df, window=50, min_convergence_pct=0.05,
                            cooldown=10):
    """Detect triangle breakouts and upper-limit tests.

    Uses linear regression on highs/lows over *window* bars to identify
    converging price action.  Two signal types:

    1. **Breakout** — price exits the formation beyond recent range + ATR buffer.
       Labels: 'ascending_triangle', 'descending_triangle', 'symmetric_triangle'.
    2. **Upper-limit test** — within a descending triangle, price approaches
       the falling upper trendline (within 0.3×ATR).  This is the setup a
       discretionary trader would watch: repeated tests of descending resistance.
       Label: 'desc_triangle_upper_test'.

    Improvements:
    - **Stricter compression** — 5% minimum (was 3%) to reduce weak formations.
    - **Cooldown** — after any triangle signal fires, suppress for *cooldown*
      bars to prevent consecutive-day clusters from the same formation.

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

    Returns
    -------
    pd.DataFrame
        Original df with added column 'triangle_pattern':
        - 'ascending_triangle'
        - 'descending_triangle'
        - 'symmetric_triangle'
        - 'desc_triangle_upper_test'
        - None otherwise
    """
    df = df.copy()
    atr = compute_atr(df, window=14)

    signals = pd.Series(None, index=df.index, dtype=object)
    x = np.arange(window)
    bars_since_last = cooldown + 1

    for i in range(window, len(df)):
        bars_since_last += 1
        atr_i = atr.iloc[i]
        if pd.isna(atr_i):
            continue

        if bars_since_last <= cooldown:
            continue

        window_slice = df.iloc[i - window : i]

        highs = window_slice["High"].values
        lows = window_slice["Low"].values

        # Fit linear trend to highs and lows
        high_coeffs = np.polyfit(x, highs, 1)
        low_coeffs = np.polyfit(x, lows, 1)
        high_slope = high_coeffs[0]
        low_slope = low_coeffs[0]

        # Range at start vs end of window
        range_start = highs[:5].mean() - lows[:5].mean()
        range_end = highs[-5:].mean() - lows[-5:].mean()

        if range_start == 0:
            continue

        compression = (range_start - range_end) / range_start

        if compression < min_convergence_pct:
            continue

        # Classify triangle type
        is_ascending = high_slope < 0.01 and low_slope > 0.01
        is_descending = high_slope < -0.01 and low_slope < 0.01
        is_symmetric = high_slope < -0.01 and low_slope > 0.01

        if not (is_ascending or is_descending or is_symmetric):
            continue

        # --- Signal type 1: breakout ---
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
            bars_since_last = 0
            continue

        # --- Signal type 2: descending triangle upper-limit test ---
        # Price approaches the falling upper trendline (supervisor reference).
        # This is the "test of resistance" setup within the formation.
        if is_descending:
            upper_at_current = np.polyval(high_coeffs, window)
            current_close = df["Close"].iloc[i]
            if abs(current_close - upper_at_current) < 0.3 * atr_i:
                signals.iloc[i] = "desc_triangle_upper_test"
                bars_since_last = 0

    df["triangle_pattern"] = signals
    return df
