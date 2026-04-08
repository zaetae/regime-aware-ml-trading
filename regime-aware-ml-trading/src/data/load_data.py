import pandas as pd
from pathlib import Path

# Find project root by looking for src/ directory
def _find_project_root():
    """Find the project root directory containing src/."""
    current = Path.cwd()
    for _ in range(10):  # Prevent infinite loop
        if (current / "src").is_dir():
            return current
        current = current.parent
    # Fallback: assume we're in the project root or notebooks/ subdirectory
    return Path.cwd().parent if (Path.cwd().parent / "src").is_dir() else Path.cwd()

PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"


def load_spy():
    """Load SPY data from raw CSV, clean it, and return a DataFrame."""
    path = DATA_DIR / "raw" / "spy.csv"

    # Fallback: if the path doesn't exist, try relative to current working directory
    if not path.is_file():
        path = Path.cwd() / "data" / "raw" / "spy.csv"
    
    # If still not found, try to download it
    if not path.is_file():
        print(f"SPY data not found at {path}, downloading...")
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
