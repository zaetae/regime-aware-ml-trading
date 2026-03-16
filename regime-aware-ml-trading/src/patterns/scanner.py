import pandas as pd

from src.data.utils import compute_atr
from src.patterns.support_resistance import calculate_support_resistance
from src.patterns.triangles import detect_triangle_pattern
from src.patterns.multiple_tops_bottoms import detect_multiple_tops_bottoms
from src.patterns.channels import detect_channel


def scan_all_patterns(df, window=20):
    """Run all 4 pattern detectors and return the enriched DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: Open, High, Low, Close, Volume
    window : int
        Rolling window size for all detectors

    Returns
    -------
    pd.DataFrame
        Original df with pattern columns added:
        - support, resistance, near_support, near_resistance
        - triangle_pattern
        - multiple_top_bottom_pattern
        - channel_pattern
        - has_event (True if any pattern detected on that row)
    """
    df = calculate_support_resistance(df, window=window)
    df = detect_triangle_pattern(df, window=window)
    df = detect_multiple_tops_bottoms(df, window=window)
    df = detect_channel(df, window=window)

    # Unified event flag: True if any pattern signal fires
    df["has_event"] = (
        df["near_support"]
        | df["near_resistance"]
        | df["triangle_pattern"].notna()
        | df["multiple_top_bottom_pattern"].notna()
        | df["channel_pattern"].notna()
    )

    return df


def get_events(df, window=20):
    """Return only the rows where a pattern event was detected."""
    df = scan_all_patterns(df, window=window)
    return df[df["has_event"]].copy()


if __name__ == "__main__":
    from src.data.load_data import load_spy

    df = load_spy()
    df = scan_all_patterns(df)

    event_count = df["has_event"].sum()
    total = len(df)
    print(f"Total bars: {total}")
    print(f"Event bars: {event_count} ({100 * event_count / total:.1f}%)")
    print()

    # Breakdown by pattern type
    print("Pattern breakdown:")
    print(f"  Near support:     {df['near_support'].sum()}")
    print(f"  Near resistance:  {df['near_resistance'].sum()}")
    print(f"  Triangles:        {df['triangle_pattern'].notna().sum()}")
    print(f"  Multi top/bottom: {df['multiple_top_bottom_pattern'].notna().sum()}")
    print(f"  Channels:         {df['channel_pattern'].notna().sum()}")
