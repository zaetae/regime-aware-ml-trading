import pandas as pd

from src.patterns.support_resistance import calculate_support_resistance
from src.patterns.triangles import detect_triangle_pattern
from src.patterns.multiple_tops_bottoms import detect_multiple_tops_bottoms
from src.patterns.channels import detect_channel


def scan_all_patterns(df, sr_window=50, tri_window=25, mtb_window=50):
    """Run all 4 pattern detectors and return the enriched DataFrame.

    Each detector uses its own tuned default parameters internally.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: Open, High, Low, Close, Volume
    sr_window : int
        Lookback for support/resistance (default 50).
    tri_window : int
        Lookback for triangle detection (default 25).
    mtb_window : int
        Lookback for multiple tops/bottoms (default 50).

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
    df = calculate_support_resistance(df, window=sr_window)
    df = detect_triangle_pattern(df, window=tri_window)
    df = detect_multiple_tops_bottoms(df, window=mtb_window)
    df = detect_channel(df)

    # Resolve direction using the same priority as _get_event_type() in the
    # labeling module.  A triangle upper-test is directionless by design and
    # therefore cannot fall through to a lower-priority signal on the bar.
    direction = pd.Series(pd.NA, index=df.index, dtype="object")
    tri_active = df["triangle_pattern"].notna()
    direction.loc[tri_active] = df.loc[tri_active, "triangle_breakout_direction"].map(
        {"up": "long", "down": "short"}
    )

    no_tri = ~tri_active
    channel_dir = df["channel_pattern"].map({"channel_up": "long", "channel_down": "short"})
    channel_active = no_tri & channel_dir.notna()
    direction.loc[channel_active] = channel_dir.loc[channel_active]

    mtb_dir = df["multiple_top_bottom_pattern"].map(
        {"multiple_top": "short", "multiple_bottom": "long"}
    )
    mtb_active = no_tri & ~channel_active & mtb_dir.notna()
    direction.loc[mtb_active] = mtb_dir.loc[mtb_active]

    sr_active = no_tri & ~channel_active & ~mtb_active
    direction.loc[sr_active & df["near_resistance"]] = "short"
    direction.loc[sr_active & df["near_support"]] = "long"
    df["intended_direction"] = direction

    # Unified event flag: True if any pattern signal fires
    df["has_event"] = (
        df["near_support"]
        | df["near_resistance"]
        | df["triangle_pattern"].notna()
        | df["multiple_top_bottom_pattern"].notna()
        | df["channel_pattern"].notna()
    )

    return df


def get_events(df, **kwargs):
    """Return only the rows where a pattern event was detected."""
    df = scan_all_patterns(df, **kwargs)
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
