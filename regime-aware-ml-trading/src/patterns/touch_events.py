"""Touch-based event generation.

Generates additional trading events when price directly touches key levels:
- Support / resistance boundaries
- Channel boundaries (upper and lower trendlines)

These complement the existing strict pattern detectors (triangles, channels,
multiple tops/bottoms, S/R proximity) by capturing moments of direct
boundary interaction that the original detectors may not flag as events.

Conservative filtering is applied:
- ATR proximity threshold (configurable, default 0.2 * ATR)
- Minimum spacing between touch events (cooldown)
- Touch events are tracked separately from original events

Supervisor feedback: "start sequences from direct touch of trend lines."
"""

import numpy as np
import pandas as pd

from src.data.utils import compute_atr


def generate_sr_touch_events(df, atr_mult=0.2, cooldown=10,
                             sr_window=50, stability_window=5):
    """Generate events when price directly touches support or resistance.

    Unlike the existing near_support/near_resistance signals (which use
    0.3 * ATR proximity), this uses a tighter threshold and focuses on
    *direct touches* where a wick reaches the level.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data. Must have 'support' and 'resistance' columns
        (from calculate_support_resistance), or they will be computed.
    atr_mult : float
        ATR multiplier for touch proximity (default 0.2 — tighter than
        the 0.3 used by the scanner).
    cooldown : int
        Minimum bars between consecutive touch events of the same type.
    sr_window : int
        Rolling window for S/R calculation if not already present.
    stability_window : int
        Minimum bars the level must be unchanged.

    Returns
    -------
    pd.DataFrame
        Boolean columns 'touch_support' and 'touch_resistance' added.
    """
    df = df.copy()
    atr = compute_atr(df)
    band = atr_mult * atr

    # Compute S/R if not already present
    if "resistance" not in df.columns or "support" not in df.columns:
        from src.patterns.support_resistance import calculate_support_resistance
        df = calculate_support_resistance(df, window=sr_window)

    # Stability filter
    res_stable = df["resistance"] == df["resistance"].shift(stability_window)
    sup_stable = df["support"] == df["support"].shift(stability_window)

    # Direct touch: wick reaches within band of level
    # For resistance: High must be close to resistance level
    raw_touch_res = ((df["resistance"] - df["High"]).abs() <= band) & res_stable
    # For support: Low must be close to support level
    raw_touch_sup = ((df["Low"] - df["support"]).abs() <= band) & sup_stable

    # Exclude bars already flagged by existing detectors
    if "near_resistance" in df.columns:
        raw_touch_res = raw_touch_res & ~df["near_resistance"]
    if "near_support" in df.columns:
        raw_touch_sup = raw_touch_sup & ~df["near_support"]

    # Apply cooldown
    df["touch_resistance"] = _apply_cooldown(raw_touch_res, cooldown)
    df["touch_support"] = _apply_cooldown(raw_touch_sup, cooldown)

    return df


def generate_channel_touch_events(df, atr_mult=0.2, cooldown=10):
    """Generate events when price directly touches channel boundaries.

    Runs the channel detector with details, then walks through the data
    checking if price wicks touch the computed trendlines.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    atr_mult : float
        ATR multiplier for touch proximity.
    cooldown : int
        Minimum bars between consecutive touch events.

    Returns
    -------
    df : pd.DataFrame
        With 'touch_channel_upper' and 'touch_channel_lower' columns.
    channel_details : list
        Channel detection details for geometry features.
    """
    from src.patterns.channels import detect_channel

    df = df.copy()
    df_with_channels, ch_details = detect_channel(df, return_details=True)

    atr = compute_atr(df)

    # Build trendline values for each detected channel
    touch_upper = pd.Series(False, index=df.index)
    touch_lower = pd.Series(False, index=df.index)

    if ch_details:
        for det in ch_details:
            event_idx = det.get("event_idx")
            window = det.get("window", 40)
            upper_coeff = det.get("upper_coeff")
            lower_coeff = det.get("lower_coeff")

            if upper_coeff is None or lower_coeff is None:
                continue
            if event_idx is None:
                continue

            start_idx = max(0, event_idx - window)
            end_idx = min(len(df), event_idx + 1)

            for j in range(start_idx, end_idx):
                rel_pos = j - start_idx
                upper_val = upper_coeff[0] * rel_pos + upper_coeff[1]
                lower_val = lower_coeff[0] * rel_pos + lower_coeff[1]
                atr_val = atr.iloc[j] if j < len(atr) else np.nan

                if pd.isna(atr_val) or atr_val <= 0:
                    continue

                band = atr_mult * atr_val

                # Upper touch: High is within band of upper trendline
                if abs(df["High"].iloc[j] - upper_val) <= band:
                    touch_upper.iloc[j] = True

                # Lower touch: Low is within band of lower trendline
                if abs(df["Low"].iloc[j] - lower_val) <= band:
                    touch_lower.iloc[j] = True

    # Exclude bars already flagged as channel events
    if "channel_pattern" in df_with_channels.columns:
        existing_channel = df_with_channels["channel_pattern"].notna()
        touch_upper = touch_upper & ~existing_channel
        touch_lower = touch_lower & ~existing_channel

    df["touch_channel_upper"] = _apply_cooldown(touch_upper, cooldown)
    df["touch_channel_lower"] = _apply_cooldown(touch_lower, cooldown)

    return df, ch_details


def generate_all_touch_events(df, atr_mult=0.2, cooldown=10):
    """Generate all touch-based events and merge with existing patterns.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data (optionally already scanned with scan_all_patterns).
    atr_mult : float
        ATR multiplier for touch proximity.
    cooldown : int
        Minimum bars between touch events.

    Returns
    -------
    df : pd.DataFrame
        With touch event columns and 'has_touch_event' flag.
    stats : dict
        Summary of touch events generated.
    """
    from src.patterns.scanner import scan_all_patterns

    # Run standard pattern scanning if not done
    if "has_event" not in df.columns:
        df = scan_all_patterns(df)

    n_original = int(df["has_event"].sum())

    # Generate S/R touch events
    df = generate_sr_touch_events(df, atr_mult=atr_mult, cooldown=cooldown)

    # Generate channel touch events
    df, _ = generate_channel_touch_events(df, atr_mult=atr_mult, cooldown=cooldown)

    # Combined touch event flag
    touch_cols = ["touch_support", "touch_resistance",
                  "touch_channel_upper", "touch_channel_lower"]
    existing_cols = [c for c in touch_cols if c in df.columns]

    if existing_cols:
        df["has_touch_event"] = df[existing_cols].any(axis=1)
    else:
        df["has_touch_event"] = False

    # Combined flag: original OR touch
    df["has_any_event"] = df["has_event"] | df["has_touch_event"]

    n_touch_only = int(df["has_touch_event"].sum() & ~df["has_event"].sum()
                       if False else
                       int((df["has_touch_event"] & ~df["has_event"]).sum()))
    n_combined = int(df["has_any_event"].sum())

    stats = {
        "n_original_events": n_original,
        "n_touch_events": int(df["has_touch_event"].sum()),
        "n_new_touch_only": n_touch_only,
        "n_combined_events": n_combined,
        "touch_support": int(df.get("touch_support", pd.Series(0)).sum()),
        "touch_resistance": int(df.get("touch_resistance", pd.Series(0)).sum()),
        "touch_channel_upper": int(df.get("touch_channel_upper", pd.Series(0)).sum()),
        "touch_channel_lower": int(df.get("touch_channel_lower", pd.Series(0)).sum()),
    }

    return df, stats


def _apply_cooldown(signal: pd.Series, cooldown: int) -> pd.Series:
    """Keep only the first True in each cluster, then suppress for cooldown bars."""
    result = pd.Series(False, index=signal.index)
    bars_since_last = cooldown + 1

    for i in range(len(signal)):
        if signal.iloc[i] and bars_since_last > cooldown:
            result.iloc[i] = True
            bars_since_last = 0
        else:
            bars_since_last += 1

    return result


def _get_touch_event_type(row):
    """Extract the touch event type from a row with touch columns."""
    if row.get("touch_channel_upper", False):
        return "touch_channel_upper"
    if row.get("touch_channel_lower", False):
        return "touch_channel_lower"
    if row.get("touch_resistance", False):
        return "touch_resistance"
    if row.get("touch_support", False):
        return "touch_support"
    return "unknown_touch"
