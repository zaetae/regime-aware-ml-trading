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
