import os
import pandas as pd
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _clean_ohlcv(df):
    """Standardise an OHLCV DataFrame: tz-strip, sort, drop NaN, validate."""
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df = df.dropna()
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df


def load_spy(source="csv"):
    """Load SPY data.

    Parameters
    ----------
    source : str
        "csv" — load from data/raw/spy.csv (default, existing behaviour).
        "yfinance" — download from Yahoo Finance (requires yfinance).
        "alphavantage" — download from Alpha Vantage (requires API key in
            env var ALPHAVANTAGE_API_KEY or passed to load_spy_alphavantage).
    """
    if source == "csv":
        return _load_csv()
    elif source == "yfinance":
        return _load_yfinance()
    elif source == "alphavantage":
        return load_spy_alphavantage()
    else:
        raise ValueError(f"Unknown source: {source!r}. Use 'csv', 'yfinance', or 'alphavantage'.")


def _load_csv():
    """Load SPY data from the raw CSV file."""
    path = DATA_DIR / "raw" / "spy.csv"
    if not path.is_file():
        path = Path.cwd() / "data" / "raw" / "spy.csv"
    if not path.is_file():
        return _load_yfinance(save=True)

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return _clean_ohlcv(df)


def _load_yfinance(save=False):
    """Download SPY from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    df = yf.download("SPY", start="2010-01-01", end="2026-01-01", auto_adjust=False)
    if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
        df.columns = df.columns.get_level_values(0)
    df = df[REQUIRED_COLS]
    df.index.name = "Date"
    df = _clean_ohlcv(df)

    if save:
        path = DATA_DIR / "raw" / "spy.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        print(f"Saved {len(df)} rows to {path}")
    return df


def load_spy_alphavantage(api_key=None, save=True):
    """Download SPY daily data from Alpha Vantage.

    Parameters
    ----------
    api_key : str, optional
        Alpha Vantage API key. If None, reads from env var
        ALPHAVANTAGE_API_KEY.
    save : bool
        If True, cache to data/raw/spy_alphavantage.csv.
    """
    cache_path = DATA_DIR / "raw" / "spy_alphavantage.csv"
    if cache_path.is_file():
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        return _clean_ohlcv(df)

    if api_key is None:
        api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Alpha Vantage API key required. Set ALPHAVANTAGE_API_KEY env var "
            "or pass api_key='...'."
        )

    import requests
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol=SPY&outputsize=full"
        f"&apikey={api_key}&datatype=csv"
    )
    print("Downloading SPY from Alpha Vantage...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    from io import StringIO
    raw = pd.read_csv(StringIO(resp.text))

    # Alpha Vantage CSV columns: timestamp, open, high, low, close, volume
    rename = {"timestamp": "Date", "open": "Open", "high": "High",
              "low": "Low", "close": "Close", "volume": "Volume"}
    raw = raw.rename(columns=rename)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("Date")
    raw = raw[REQUIRED_COLS]
    df = _clean_ohlcv(raw)

    # Filter to 2010+ to match yfinance range
    df = df.loc["2010-01-01":]

    if save:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
        print(f"Saved {len(df)} rows to {cache_path}")
    return df


def compare_sources(df_yf, df_av):
    """Compare yfinance and Alpha Vantage DataFrames.

    Returns a dict with comparison statistics.
    """
    common_idx = df_yf.index.intersection(df_av.index)
    stats = {
        "yf_rows": len(df_yf),
        "av_rows": len(df_av),
        "common_dates": len(common_idx),
        "yf_only_dates": len(df_yf.index.difference(df_av.index)),
        "av_only_dates": len(df_av.index.difference(df_yf.index)),
        "yf_start": df_yf.index[0],
        "yf_end": df_yf.index[-1],
        "av_start": df_av.index[0],
        "av_end": df_av.index[-1],
    }

    if len(common_idx) > 0:
        yf_c = df_yf.loc[common_idx]
        av_c = df_av.loc[common_idx]
        close_diff = (yf_c["Close"] - av_c["Close"]).abs()
        stats["close_mean_diff"] = round(close_diff.mean(), 4)
        stats["close_max_diff"] = round(close_diff.max(), 4)
        stats["close_corr"] = round(yf_c["Close"].corr(av_c["Close"]), 6)
        vol_diff = (yf_c["Volume"] - av_c["Volume"]).abs()
        stats["volume_mean_diff"] = round(vol_diff.mean(), 0)
        stats["volume_corr"] = round(yf_c["Volume"].corr(av_c["Volume"]), 6)

    return stats


if __name__ == "__main__":
    df = load_spy()
    print(f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    print(df.head())
