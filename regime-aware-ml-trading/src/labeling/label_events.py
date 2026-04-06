"""Triple-barrier labeling for detected technical events.

Assigns directional labels (long / short / no_trade) to each event based on
which price barrier is touched first after the event bar.

The three barriers:
    Upper — entry_price + pt_mult * ATR  (profit target for longs)
    Lower — entry_price - sl_mult * ATR  (profit target for shorts)
    Time  — max_holding bars ahead       (expiry, no significant move)

References
----------
Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 3.
"""

import numpy as np
import pandas as pd

from src.data.utils import compute_atr


# ------------------------------------------------------------------
# Event type extraction
# ------------------------------------------------------------------

def _get_event_type(row):
    """Extract the primary event type from a scanner output row.

    Priority: pattern-specific signals > support/resistance.
    """
    if pd.notna(row.get("triangle_pattern")):
        return row["triangle_pattern"]
    if pd.notna(row.get("channel_pattern")):
        return row["channel_pattern"]
    if pd.notna(row.get("multiple_top_bottom_pattern")):
        return row["multiple_top_bottom_pattern"]
    if row.get("near_resistance"):
        return "near_resistance"
    if row.get("near_support"):
        return "near_support"
    return "unknown"


# ------------------------------------------------------------------
# Core labeling function
# ------------------------------------------------------------------

def triple_barrier_label(df, events, pt_mult=2.0, sl_mult=2.0,
                         max_holding=10, atr_window=14):
    """Apply triple-barrier labeling to detected events.

    For each event bar the algorithm walks forward through subsequent bars
    and checks whether price breaches the upper or lower barrier before the
    time limit.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV dataset with DatetimeIndex.
    events : pd.DataFrame
        Subset of *df* where ``has_event == True`` (from the scanner).
    pt_mult : float
        Upper barrier distance in ATR multiples (default 2.0).
    sl_mult : float
        Lower barrier distance in ATR multiples (default 2.0).
    max_holding : int
        Maximum bars to hold before the time barrier (default 10).
    atr_window : int
        ATR lookback period (default 14).

    Returns
    -------
    pd.DataFrame
        One row per labeled event with columns:

        ============== ==============================================
        event_date     Timestamp of the event bar
        event_type     Which pattern fired (e.g. "channel_up")
        entry_price    Close at event bar
        atr            ATR value at event bar
        upper_barrier  entry + pt_mult * ATR
        lower_barrier  entry - sl_mult * ATR
        exit_date      Date when a barrier was touched
        exit_price     Price at exit (barrier level or Close at expiry)
        bars_held      Number of bars from entry to exit (1..max_holding)
        label          "long", "short", or "no_trade"
        return_pct     (exit_price - entry_price) / entry_price * 100
        ============== ==============================================
    """
    atr = compute_atr(df, window=atr_window)

    results = []
    for event_date, row in events.iterrows():
        pos = df.index.get_loc(event_date)
        entry_price = df["Close"].iloc[pos]
        atr_val = atr.iloc[pos]

        if pd.isna(atr_val) or atr_val <= 0:
            continue

        upper = entry_price + pt_mult * atr_val
        lower = entry_price - sl_mult * atr_val

        # Default: time barrier (expiry)
        end_pos = min(pos + max_holding, len(df) - 1)
        label = "no_trade"
        exit_price = df["Close"].iloc[end_pos]
        exit_date = df.index[end_pos]
        bars_held = end_pos - pos

        # Walk forward, check High/Low against barriers
        for j in range(pos + 1, min(pos + max_holding + 1, len(df))):
            high_j = df["High"].iloc[j]
            low_j = df["Low"].iloc[j]

            hit_upper = high_j >= upper
            hit_lower = low_j <= lower

            if hit_upper and hit_lower:
                # Both barriers breached on the same bar — use Close
                close_j = df["Close"].iloc[j]
                label = "long" if close_j >= entry_price else "short"
                exit_price = close_j
                exit_date = df.index[j]
                bars_held = j - pos
                break
            elif hit_upper:
                label = "long"
                exit_price = upper
                exit_date = df.index[j]
                bars_held = j - pos
                break
            elif hit_lower:
                label = "short"
                exit_price = lower
                exit_date = df.index[j]
                bars_held = j - pos
                break

        return_pct = (exit_price - entry_price) / entry_price * 100

        results.append({
            "event_date": event_date,
            "event_type": _get_event_type(row),
            "entry_price": round(entry_price, 2),
            "atr": round(atr_val, 4),
            "upper_barrier": round(upper, 2),
            "lower_barrier": round(lower, 2),
            "exit_date": exit_date,
            "exit_price": round(exit_price, 2),
            "bars_held": bars_held,
            "label": label,
            "return_pct": round(return_pct, 4),
        })

    return pd.DataFrame(results)


# ------------------------------------------------------------------
# Convenience wrapper
# ------------------------------------------------------------------

def label_events(df, pt_mult=2.0, sl_mult=2.0, max_holding=10,
                 atr_window=14):
    """Run the scanner (if needed) and apply triple-barrier labels.

    This is the main entry point for downstream code.  It:

    1. Runs ``scan_all_patterns`` if ``has_event`` is not already in *df*.
    2. Extracts event rows.
    3. Applies :func:`triple_barrier_label`.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV dataset (with or without pattern columns).
    pt_mult, sl_mult, max_holding, atr_window
        Forwarded to :func:`triple_barrier_label`.

    Returns
    -------
    pd.DataFrame
        Labeled events — see :func:`triple_barrier_label`.
    """
    from src.patterns.scanner import scan_all_patterns

    if "has_event" not in df.columns:
        df = scan_all_patterns(df)

    events = df[df["has_event"]].copy()
    return triple_barrier_label(df, events, pt_mult, sl_mult,
                                max_holding, atr_window)
