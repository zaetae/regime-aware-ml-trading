import pandas as pd
from pathlib import Path

# Resolve data directory relative to this file's location.
# __file__ is always correct regardless of the working directory.
# src/data/load_data.py  →  ../../data  =  <project_root>/data
_THIS_DIR = Path(__file__).resolve().parent          # src/data/
PROJECT_ROOT = _THIS_DIR.parent.parent               # <project_root>
DATA_DIR = PROJECT_ROOT / "data"


def load_spy():
    """Load SPY data from raw CSV, clean it, and return a DataFrame."""
    path = DATA_DIR / "raw" / "spy.csv"

    # Fallback: if the path doesn't exist, try relative to current working directory
    if not path.is_file():
        path = Path.cwd() / "data" / "raw" / "spy.csv"
    
    # If still not found, try to download it to the canonical location
    if not path.is_file():
        path = DATA_DIR / "raw" / "spy.csv"
        print(f"SPY data not found, downloading to {path}...")
        try:
            import yfinance as yf
            df = yf.download("SPY", start="2010-01-01", end="2026-01-01", auto_adjust=False)
            # Flatten MultiIndex columns if present
            if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
                df.columns = df.columns.get_level_values(0)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.index.name = 'Date'
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path)
            print(f"Downloaded {len(df)} rows to {path}")
        except ImportError:
            raise FileNotFoundError(f"SPY data not found at {path} and yfinance not available to download")
    
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    # yfinance may save timezone-aware dates — strip tz for consistent plotting
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df = df.dropna()
    # Ensure expected columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df


if __name__ == "__main__":
    df = load_spy()
    print(f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    print(df.head())
