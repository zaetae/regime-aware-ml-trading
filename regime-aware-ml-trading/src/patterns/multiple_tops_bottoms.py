import numpy as np
import pandas as pd

from src.patterns.support_resistance import _apply_cooldown


def detect_multiple_tops_bottoms(df, window=20, confirm_bars=5, cooldown=10):
    """Detect multiple top and multiple bottom patterns.

    - Multiple Top: price highs are hitting a ceiling (rolling max of highs
      stays high) while recent closes confirm a downward trend.
    - Multiple Bottom: price lows are hitting a floor (rolling min of lows
      stays low) while recent closes confirm an upward trend.

    Improvements:
    - **Longer confirmation** — uses *confirm_bars* (default 5, was 3) for
      close-trend slope, reducing noise from short-term fluctuations.
    - **Cooldown** — after a signal fires, suppress the same type for
      *cooldown* bars to prevent clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: High, Low, Close
    window : int
        Rolling window size.
    confirm_bars : int
        Number of bars for close-trend slope confirmation (default 5).
    cooldown : int
        Minimum bars between consecutive signals of same type (default 10).

    Returns
    -------
    pd.DataFrame
        Original df with added column 'multiple_top_bottom_pattern':
        - 'multiple_top'
        - 'multiple_bottom'
        - None otherwise
    """
    df = df.copy()

    high_roll_max = df["High"].rolling(window=window).max()
    low_roll_min = df["Low"].rolling(window=window).min()
    close_roll_max = df["Close"].rolling(window=window).max()
    close_roll_min = df["Close"].rolling(window=window).min()

    # Base conditions: highs hitting ceiling / lows hitting floor
    top_base = (high_roll_max >= high_roll_max.shift(1)) & (
        close_roll_max < close_roll_max.shift(1)
    )
    bottom_base = (low_roll_min <= low_roll_min.shift(1)) & (
        close_roll_min > close_roll_min.shift(1)
    )

    # Close-trend confirmation over last confirm_bars bars
    x_confirm = np.arange(float(confirm_bars))
    closes = df["Close"].values
    close_slope = pd.Series(np.nan, index=df.index)

    for i in range(confirm_bars, len(df)):
        close_slope.iloc[i] = np.polyfit(
            x_confirm, closes[i - confirm_bars : i], 1
        )[0]

    # Raw masks
    raw_top = top_base & (close_slope < 0)
    raw_bottom = bottom_base & (close_slope > 0)

    # Apply cooldown independently to each type
    top_filtered = _apply_cooldown(raw_top, cooldown)
    bottom_filtered = _apply_cooldown(raw_bottom, cooldown)

    df["multiple_top_bottom_pattern"] = None
    df.loc[top_filtered, "multiple_top_bottom_pattern"] = "multiple_top"
    df.loc[bottom_filtered, "multiple_top_bottom_pattern"] = "multiple_bottom"

    return df
