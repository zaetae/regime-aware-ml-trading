import yfinance as yf
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def download_spy(start="2010-01-01", end="2025-12-31"):
    """Download SPY daily OHLCV data and save to data/raw/spy.csv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = yf.download("SPY", start=start, end=end, auto_adjust=True)

    # yfinance returns multi-level columns for single ticker; flatten them
    if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
        df.columns = df.columns.get_level_values(0)

    path = os.path.join(DATA_DIR, "spy.csv")
    df.to_csv(path)
    print(f"Saved {len(df)} rows to {path}")
    return df


if __name__ == "__main__":
    download_spy()
