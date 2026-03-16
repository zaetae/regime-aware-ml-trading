import numpy as np
import pandas as pd

from src.data.utils import compute_atr


def detect_triangle_pattern(df, window=50, min_convergence_pct=0.03):
    """Detect ascending, descending, and symmetric triangle breakouts.

    Uses linear regression on highs/lows over *window* bars to identify
    converging price action, then fires only on the breakout bar — not
    during formation.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: High, Low, Close
    window : int
        Lookback window for the formation (default 50)
    min_convergence_pct : float
        Minimum range compression required (default 0.03 = 3%)

    Returns
    -------
    pd.DataFrame
        Original df with added column 'triangle_pattern':
        - 'ascending_triangle'
        - 'descending_triangle'
        - 'symmetric_triangle'
        - None otherwise
    """
    df = df.copy()
    atr = compute_atr(df, window=14)

    signals = pd.Series(None, index=df.index, dtype=object)
    x = np.arange(window)

    for i in range(window, len(df)):
        window_slice = df.iloc[i - window : i]

        highs = window_slice["High"].values
        lows = window_slice["Low"].values

        # Fit linear trend to highs and lows
        high_slope = np.polyfit(x, highs, 1)[0]
        low_slope = np.polyfit(x, lows, 1)[0]

        # Range at start vs end of window
        range_start = highs[:5].mean() - lows[:5].mean()
        range_end = highs[-5:].mean() - lows[-5:].mean()

        if range_start == 0:
            continue

        compression = (range_start - range_end) / range_start

        # Must have meaningful compression
        if compression < min_convergence_pct:
            continue

        # Classify triangle type
        is_ascending = high_slope < 0.01 and low_slope > 0.01
        is_descending = high_slope < -0.01 and low_slope < 0.01
        is_symmetric = high_slope < -0.01 and low_slope > 0.01

        if not (is_ascending or is_descending or is_symmetric):
            continue

        # Only flag the breakout bar
        current_high = df["High"].iloc[i]
        current_low = df["Low"].iloc[i]
        recent_high = highs[-5:].max()
        recent_low = lows[-5:].min()

        atr_i = atr.iloc[i]
        if pd.isna(atr_i):
            continue

        breakout_up = current_high > recent_high + 0.3 * atr_i
        breakout_down = current_low < recent_low - 0.3 * atr_i

        if breakout_up or breakout_down:
            if is_ascending:
                signals.iloc[i] = "ascending_triangle"
            elif is_descending:
                signals.iloc[i] = "descending_triangle"
            else:
                signals.iloc[i] = "symmetric_triangle"

    df["triangle_pattern"] = signals
    return df
