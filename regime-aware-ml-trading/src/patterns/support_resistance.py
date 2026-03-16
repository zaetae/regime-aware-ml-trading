import pandas as pd

from src.data.utils import compute_atr


def calculate_support_resistance(df, window=50, atr_mult=0.3):
    """Calculate dynamic support and resistance levels.

    Uses rolling max of High / rolling min of Low over *window* bars to set
    resistance and support.  Flags when price is within *atr_mult* × ATR(14)
    of either level.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: High, Low, Close
    window : int
        Rolling window size (default 50 for ~2.5 months of trading days)
    atr_mult : float
        ATR multiplier for the proximity band (default 0.5)

    Returns
    -------
    pd.DataFrame
        Original df with added columns:
        - support: rolling low-based support level
        - resistance: rolling high-based resistance level
        - near_support: True when Close is within atr_mult × ATR of support
        - near_resistance: True when Close is within atr_mult × ATR of resistance
    """
    df = df.copy()

    df["resistance"] = df["High"].rolling(window=window).max()
    df["support"] = df["Low"].rolling(window=window).min()

    atr = compute_atr(df)
    band = atr_mult * atr

    df["near_support"] = (df["Close"] - df["support"]).abs() <= band
    df["near_resistance"] = (df["Close"] - df["resistance"]).abs() <= band

    return df
