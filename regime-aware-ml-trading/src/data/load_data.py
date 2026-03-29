import pandas as pd
import os

# Resolve data directory relative to this file's location.
# os.path.realpath resolves symlinks so the ../.. traversal always lands
# at the project root, even when the module is imported via sys.path on Colab.
DATA_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..", "..", "data")


def load_spy():
    """Load SPY data from raw CSV, clean it, and return a DataFrame."""
    path = os.path.join(DATA_DIR, "raw", "spy.csv")

    # Fallback: if the __file__-based path doesn't exist, try cwd/data/
    if not os.path.isfile(path):
        path = os.path.join(os.getcwd(), "data", "raw", "spy.csv")
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
