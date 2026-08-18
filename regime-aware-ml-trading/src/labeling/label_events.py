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
                         max_holding=10, atr_window=14,
                         direction_col="intended_direction"):
    """Apply triple-barrier labeling to detected events.

    For each event bar the algorithm walks forward through subsequent bars
    and checks position-specific profit-target and stop-loss barriers. The
    event's intended direction must be determined causally by its detector.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV dataset with DatetimeIndex.
    events : pd.DataFrame
        Subset of *df* where ``has_event == True`` (from the scanner).
    pt_mult : float
        Favorable-direction profit-target distance in ATR multiples.
    sl_mult : float
        Adverse-direction stop-loss distance in ATR multiples.
    max_holding : int
        Maximum bars to hold before the time barrier (default 10).
    atr_window : int
        ATR lookback period (default 14).
    direction_col : str
        Event column containing ``"long"`` or ``"short"``.

    Returns
    -------
    pd.DataFrame
        One row per labeled event with columns:

        ============== ==============================================
        event_date     Timestamp of the event bar
        event_type     Which pattern fired (e.g. "channel_up")
        entry_price    Close at event bar
        atr            ATR value at event bar
        intended_direction  Detector-defined proposed trade direction
        tp_price       Position-specific profit-target price
        sl_price       Position-specific stop-loss price
        exit_date      Date when a barrier was touched
        exit_price     Price at exit (barrier level or Close at expiry)
        bars_held      Number of bars from entry to exit (1..max_holding)
        label          "long", "short", or "no_trade"
        return_pct     (exit_price - entry_price) / entry_price * 100
        ============== ==============================================
    """
    if direction_col not in events.columns:
        raise ValueError(
            f"events must include '{direction_col}' with values 'long' or 'short'"
        )

    atr = compute_atr(df, window=atr_window)

    results = []
    for event_date, row in events.iterrows():
        intended_direction = row[direction_col]
        if intended_direction not in {"long", "short"}:
            raise ValueError(
                f"Invalid {direction_col}={intended_direction!r} at {event_date}"
            )

        pos = df.index.get_loc(event_date)
        entry_price = df["Close"].iloc[pos]
        atr_val = atr.iloc[pos]

        if pd.isna(atr_val) or atr_val <= 0:
            continue

        if intended_direction == "long":
            tp_price = entry_price + pt_mult * atr_val
            sl_price = entry_price - sl_mult * atr_val
        else:
            tp_price = entry_price - pt_mult * atr_val
            sl_price = entry_price + sl_mult * atr_val

        # Default: time barrier (expiry)
        end_pos = min(pos + max_holding, len(df) - 1)
        label = "no_trade"
        exit_price = df["Close"].iloc[end_pos]
        exit_date = df.index[end_pos]
        bars_held = end_pos - pos

        exit_reason = "time_exit"

        # Walk forward, using the exact position-specific convention used by
        # backtest.simulator.simulate_trades().
        for j in range(pos + 1, min(pos + max_holding + 1, len(df))):
            high_j = df["High"].iloc[j]
            low_j = df["Low"].iloc[j]

            if intended_direction == "long":
                hit_tp = high_j >= tp_price
                hit_sl = low_j <= sl_price
            else:
                hit_tp = low_j <= tp_price
                hit_sl = high_j >= sl_price

            if hit_tp and hit_sl:
                # Daily OHLC does not reveal barrier-hit order. Preserve the
                # simulator's close exit, but leave the label as no_trade.
                exit_price = df["Close"].iloc[j]
                exit_date = df.index[j]
                exit_reason = "both_barriers"
                bars_held = j - pos
                break
            elif hit_tp:
                label = intended_direction
                exit_price = tp_price
                exit_date = df.index[j]
                exit_reason = "take_profit"
                bars_held = j - pos
                break
            elif hit_sl:
                exit_price = sl_price
                exit_date = df.index[j]
                exit_reason = "stop_loss"
                bars_held = j - pos
                break

        raw_return = (
            (exit_price - entry_price) / entry_price
            if intended_direction == "long"
            else (entry_price - exit_price) / entry_price
        )

        results.append({
            "event_date": event_date,
            "event_type": _get_event_type(row),
            "intended_direction": intended_direction,
            "entry_price": round(entry_price, 2),
            "atr": round(atr_val, 4),
            "tp_price": round(tp_price, 2),
            "sl_price": round(sl_price, 2),
            "exit_date": exit_date,
            "exit_price": round(exit_price, 2),
            "exit_reason": exit_reason,
            "bars_held": bars_held,
            "label": label,
            "return_pct": round(raw_return * 100, 4),
        })

    return pd.DataFrame(results)


# ------------------------------------------------------------------
# Convenience wrapper
# ------------------------------------------------------------------

def label_events(df, pt_mult=2.0, sl_mult=2.0, max_holding=10,
                 atr_window=14, exclude_patterns=None):
    """Run the scanner (if needed) and apply triple-barrier labels.

    This is the main entry point for downstream code.  It:

    1. Runs ``scan_all_patterns`` if ``has_event`` is not already in *df*.
    2. Extracts event rows (optionally filtering out excluded pattern types).
    3. Applies :func:`triple_barrier_label`.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV dataset (with or without pattern columns).
    pt_mult, sl_mult, max_holding, atr_window
        Forwarded to :func:`triple_barrier_label`.
    exclude_patterns : list[str], optional
        Pattern column names to exclude, e.g.
        ``['triangle_pattern', 'channel_pattern']``.
        Rows whose *only* active signal comes from an excluded column
        are dropped.

    Returns
    -------
    pd.DataFrame
        Labeled events — see :func:`triple_barrier_label`.
    """
    from src.patterns.scanner import scan_all_patterns

    if "has_event" not in df.columns:
        df = scan_all_patterns(df)

    events = df[df["has_event"]].copy()

    if exclude_patterns:
        # Null-out excluded columns so _get_event_type picks the next
        # available signal, then drop rows with no remaining signal.
        for col in exclude_patterns:
            if col in events.columns:
                if events[col].dtype == bool:
                    events[col] = False
                else:
                    events[col] = pd.NA
        # Keep only rows that still have at least one active signal
        all_pattern_cols = ["near_support", "near_resistance",
                            "triangle_pattern",
                            "multiple_top_bottom_pattern",
                            "channel_pattern"]
        kept_signals = []
        for col in all_pattern_cols:
            if col not in events.columns:
                continue
            if events[col].dtype == bool:
                kept_signals.append(events[col])
            else:
                kept_signals.append(events[col].notna())
        if kept_signals:
            has_any = kept_signals[0]
            for s in kept_signals[1:]:
                has_any = has_any | s
            events = events[has_any]

    # Directionless events (currently descending-triangle upper tests) are
    # retained in detector output but cannot define a directional trade label.
    events = events[events["intended_direction"].isin(["long", "short"])]

    return triple_barrier_label(df, events, pt_mult, sl_mult,
                                max_holding, atr_window)
