"""Shared helpers for experiment notebooks 21+."""
import numpy as np
import pandas as pd
import yfinance as yf

from src.data.utils import compute_atr
from src.patterns.scanner import scan_all_patterns
from src.features.build_features import build_feature_matrix, compute_all_indicators

TICKERS = ['SPY', 'QQQ', 'NVDA', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'META', 'JPM', 'XOM']


def download_ohlcv(ticker, start='2010-01-01', end='2026-01-01'):
    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']]
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw.index.name = 'Date'
    return raw


def bars_since_last_event(event_mask, max_val=10):
    event_mask = event_mask.values if hasattr(event_mask, 'values') else event_mask
    result = np.full(len(event_mask), max_val, dtype=int)
    last_seen = -1
    for i in range(len(event_mask)):
        if event_mask[i]:
            last_seen = i
            result[i] = 0
        elif last_seen >= 0:
            result[i] = min(i - last_seen, max_val)
    return result


def label_every_bar(df, pt_mult=2.0, sl_mult=2.0, max_holding=10, atr_window=14):
    """Direction-agnostic triple-barrier label for every bar."""
    atr = compute_atr(df, window=atr_window)
    rows = []
    for i in range(len(df) - max_holding):
        entry = df["Close"].iloc[i]
        a = atr.iloc[i]
        if pd.isna(a) or a <= 0:
            continue
        upper, lower = entry + pt_mult * a, entry - sl_mult * a
        label = "no_trade"
        for j in range(i + 1, i + max_holding + 1):
            hit_u = df["High"].iloc[j] >= upper
            hit_l = df["Low"].iloc[j] <= lower
            if hit_u and hit_l:
                break
            elif hit_u:
                label = "long"; break
            elif hit_l:
                label = "short"; break
        rows.append({"event_date": df.index[i], "label": label})
    return pd.DataFrame(rows)


def build_event_data(ticker, raw=None):
    """Event-based dataset (detector-selected bars only)."""
    raw = download_ohlcv(ticker) if raw is None else raw
    feats, labels, labeled = build_feature_matrix(raw.copy())
    return {'ticker': ticker, 'raw': raw, 'features': feats,
            'labels': labels, 'labeled_df': labeled}


def build_bar_data(ticker, raw=None):
    """Bar-by-bar dataset: indicators + recency features for every bar."""
    raw = download_ohlcv(ticker) if raw is None else raw
    scanned = scan_all_patterns(raw.copy())
    labeled = label_every_bar(raw.copy())
    ind = compute_all_indicators(raw.copy())
    labeled = labeled[labeled['event_date'].isin(ind.index)].reset_index(drop=True)
    dates = pd.DatetimeIndex(labeled['event_date'])

    feats = ind.loc[dates].reset_index(drop=True)
    abs_sma = [c for c in feats.columns if c.startswith("sma_") and "_dist" not in c]
    feats = feats.drop(columns=abs_sma, errors="ignore").dropna(axis=1, how="all")

    rec = pd.DataFrame({
        'bars_since_near_support': bars_since_last_event(scanned['near_support']),
        'bars_since_near_resistance': bars_since_last_event(scanned['near_resistance']),
        'bars_since_triangle': bars_since_last_event(scanned['triangle_pattern'].notna()),
        'bars_since_channel': bars_since_last_event(scanned['channel_pattern'].notna()),
        'bars_since_mtb': bars_since_last_event(scanned['multiple_top_bottom_pattern'].notna()),
    }, index=scanned.index).loc[dates].reset_index(drop=True)

    return {'ticker': ticker, 'raw': raw,
            'features': pd.concat([feats, rec], axis=1),
            'labels': labeled['label'], 'dates': dates}


def align_columns(data_dict):
    """Give every ticker the same feature columns (missing ones filled with 0)."""
    cols = sorted(set().union(*[set(d['features'].columns) for d in data_dict.values()]))
    for d in data_dict.values():
        d['features'] = d['features'].reindex(columns=cols, fill_value=0)
    return cols


def split_per_ticker(d, train=0.6, val=0.2):
    """Chronological 60/20/20 split inside ONE ticker's own timeline."""
    n = len(d['features'])
    a, b = int(n * train), int(n * (train + val))
    X, y = d['features'].fillna(0), d['labels']
    return (X.iloc[:a], y.iloc[:a]), (X.iloc[a:b], y.iloc[a:b]), (X.iloc[b:], y.iloc[b:])