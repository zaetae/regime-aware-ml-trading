import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def load_spy():
    """Load SPY data from raw CSV, clean it, and return a DataFrame."""
    path = os.path.join(DATA_DIR, "raw", "spy.csv")
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
