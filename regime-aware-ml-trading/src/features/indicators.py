"""Technical indicator calculations for feature engineering.

All functions take a DataFrame with OHLCV columns and return Series or
DataFrames. No lookahead: every value at bar *i* uses data up to and
including bar *i*.
"""

import numpy as np
import pandas as pd
from src.data.utils import compute_atr


def atr(df, window=14):
    return compute_atr(df, window)


def atr_ratio(df, window=14):
    """ATR / Close — normalised volatility."""
    return compute_atr(df, window) / df["Close"]


def rolling_volatility(df, window=20):
    """Annualised rolling volatility of log returns."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def returns(df, periods=None):
    """Simple returns over multiple horizons.

    Returns a DataFrame with columns like ret_1, ret_5, ret_10, ret_20.
    """
    if periods is None:
        periods = [1, 5, 10, 20]
    out = pd.DataFrame(index=df.index)
    for p in periods:
        out[f"ret_{p}"] = df["Close"].pct_change(p)
    return out


def moving_averages(df, windows=None):
    """Simple moving averages + distance from price."""
    if windows is None:
        windows = [10, 20, 50, 100, 200]
    out = pd.DataFrame(index=df.index)
    for w in windows:
        ma = df["Close"].rolling(w).mean()
        out[f"sma_{w}"] = ma
        out[f"sma_{w}_dist"] = (df["Close"] - ma) / ma  # relative distance
    return out


def ma_spreads(df):
    """Spreads between key moving average pairs."""
    sma_10 = df["Close"].rolling(10).mean()
    sma_20 = df["Close"].rolling(20).mean()
    sma_50 = df["Close"].rolling(50).mean()
    sma_200 = df["Close"].rolling(200).mean()
    out = pd.DataFrame(index=df.index)
    out["ma_spread_10_50"] = (sma_10 - sma_50) / sma_50
    out["ma_spread_20_200"] = (sma_20 - sma_200) / sma_200
    out["ma_spread_50_200"] = (sma_50 - sma_200) / sma_200
    return out


def rsi(df, window=14):
    """Relative Strength Index."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(df, fast=12, slow=26, signal=9):
    """MACD line, signal line, and histogram."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    out = pd.DataFrame(index=df.index)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = histogram
    return out


def bollinger_bands(df, window=20, num_std=2):
    """Bollinger Band width and %B."""
    sma = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    out = pd.DataFrame(index=df.index)
    out["bb_width"] = (upper - lower) / sma
    out["bb_pctb"] = (df["Close"] - lower) / (upper - lower)
    return out


def momentum(df, periods=None):
    """Rate of change (momentum) over multiple horizons."""
    if periods is None:
        periods = [5, 10, 20]
    out = pd.DataFrame(index=df.index)
    for p in periods:
        out[f"mom_{p}"] = df["Close"] / df["Close"].shift(p) - 1
    return out


def volume_features(df, window=20):
    """Volume-related features."""
    out = pd.DataFrame(index=df.index)
    vol_ma = df["Volume"].rolling(window).mean()
    out["volume_ratio"] = df["Volume"] / vol_ma
    out["volume_std"] = df["Volume"].rolling(window).std() / vol_ma
    # On-balance indicator: cumulative sign * volume
    price_dir = np.sign(df["Close"].diff())
    out["obv_norm"] = (price_dir * df["Volume"]).cumsum() / df["Volume"].rolling(window).sum()
    return out


def compute_all_indicators(df):
    """Compute all technical indicators and return a single DataFrame.

    Only uses information at or before each bar (no lookahead).
    """
    out = pd.DataFrame(index=df.index)

    out["atr_14"] = atr(df, 14)
    out["atr_ratio"] = atr_ratio(df, 14)
    out["rvol_20"] = rolling_volatility(df, 20)

    out = out.join(returns(df, [1, 5, 10, 20]))
    out = out.join(moving_averages(df, [10, 20, 50, 100, 200]))
    out = out.join(ma_spreads(df))

    out["rsi_14"] = rsi(df, 14)
    out = out.join(macd(df))
    out = out.join(bollinger_bands(df))
    out = out.join(momentum(df, [5, 10, 20]))
    out = out.join(volume_features(df))

    return out
