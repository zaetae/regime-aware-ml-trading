import yfinance as yf
from pathlib import Path

# Import the DATA_DIR from load_data for consistency
from .load_data import DATA_DIR

RAW_DIR = DATA_DIR / "raw"


def download_spy(start="2010-01-01", end="2025-12-31"):
    """Download SPY daily OHLCV data and save to data/raw/spy.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = yf.download("SPY", start=start, end=end, auto_adjust=True)

    # yfinance returns multi-level columns for single ticker; flatten them
    if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
        df.columns = df.columns.get_level_values(0)

    path = RAW_DIR / "spy.csv"
    df.to_csv(path)
    print(f"Saved {len(df)} rows to {path}")
    return df


if __name__ == "__main__":
    download_spy()
