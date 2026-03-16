import pandas as pd


def compute_atr(df, window=14):
    """Average True Range.

    Parameters
    ----------
    df : pd.DataFrame
        Must have High, Low, Close columns.
    window : int
        Rolling window size (default 14).

    Returns
    -------
    pd.Series
        ATR values.
    """
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()
