import numpy as np
import pandas as pd


def detect_multiple_tops_bottoms(df, window=20):
    """Detect multiple top and multiple bottom patterns.

    - Multiple Top: price highs are hitting a ceiling (rolling max of highs
      stays high) while the last 3 closes confirm a downward trend.
    - Multiple Bottom: price lows are hitting a floor (rolling min of lows
      stays low) while the last 3 closes confirm an upward trend.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: High, Low, Close
    window : int
        Rolling window size

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

    # Close-trend confirmation over last 3 bars
    x3 = np.array([0.0, 1.0, 2.0])
    closes = df["Close"].values
    close_slope = pd.Series(np.nan, index=df.index)

    for i in range(3, len(df)):
        close_slope.iloc[i] = np.polyfit(x3, closes[i - 3 : i], 1)[0]

    # Multiple Top: closes must be trending down
    top_mask = top_base & (close_slope < 0)

    # Multiple Bottom: closes must be trending up
    bottom_mask = bottom_base & (close_slope > 0)

    df["multiple_top_bottom_pattern"] = None
    df.loc[top_mask, "multiple_top_bottom_pattern"] = "multiple_top"
    df.loc[bottom_mask, "multiple_top_bottom_pattern"] = "multiple_bottom"

    return df
