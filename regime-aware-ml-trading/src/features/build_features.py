"""Build event-based feature matrix for ML training.

Each row represents one labeled event. Features use only information
available at or before the event bar (no lookahead).

Supports two event modes:
- "standard": only events from the original pattern detectors
- "with_touch": adds touch-based events (price touching S/R, channels)
"""

import numpy as np
import pandas as pd

from src.features.indicators import compute_all_indicators
from src.data.utils import compute_atr
from src.patterns.scanner import scan_all_patterns
from src.patterns.triangles import detect_triangle_pattern
from src.patterns.channels import detect_channel
from src.labeling.label_events import label_events, triple_barrier_label


def _pattern_geometry_features(labeled_df, tri_details, ch_details):
    """Extract pattern geometry features for each labeled event.

    Maps detection metadata (touch counts, containment, slopes, etc.)
    to each event row.
    """
    # Index details by event_date for fast lookup
    tri_map = {d["event_date"]: d for d in tri_details} if tri_details else {}
    ch_map = {d["event_date"]: d for d in ch_details} if ch_details else {}

    rows = []
    for _, ev in labeled_df.iterrows():
        dt = ev["event_date"]
        feat = {}

        tri = tri_map.get(dt, None)
        ch = ch_map.get(dt, None)
        det = tri or ch  # whichever matched

        if det:
            feat["upper_slope"] = det.get("upper_slope", np.nan)
            feat["lower_slope"] = det.get("lower_slope", np.nan)
            feat["containment"] = det.get("containment_ratio", np.nan)
            feat["upper_touches"] = det.get("upper_touches", 0)
            feat["lower_touches"] = det.get("lower_touches", 0)
            feat["total_touches"] = feat["upper_touches"] + feat["lower_touches"]
            feat["upper_mean_error"] = det.get("upper_mean_error", np.nan)
            feat["lower_mean_error"] = det.get("lower_mean_error", np.nan)
            feat["pattern_window"] = det.get("window", np.nan)
            feat["channel_width_atr"] = det.get("channel_width_atr", np.nan)
            feat["r_upper"] = det.get("r_upper", np.nan)
            feat["r_lower"] = det.get("r_lower", np.nan)
        else:
            for k in ["upper_slope", "lower_slope", "containment",
                       "upper_touches", "lower_touches", "total_touches",
                       "upper_mean_error", "lower_mean_error",
                       "pattern_window", "channel_width_atr",
                       "r_upper", "r_lower"]:
                feat[k] = np.nan

        rows.append(feat)

    return pd.DataFrame(rows, index=labeled_df.index)


def _event_type_dummies(labeled_df):
    """One-hot encode the event type."""
    return pd.get_dummies(labeled_df["event_type"], prefix="etype")


def build_feature_matrix(df, exclude_patterns=None,
                         pt_mult=2.0, sl_mult=2.0, max_holding=10,
                         include_touch_events=False, touch_atr_mult=0.2,
                         touch_cooldown=10):
    """Build the full feature matrix for labeled events.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data.
    exclude_patterns : list, optional
        Pattern types to exclude from labeling.
    pt_mult, sl_mult, max_holding
        Forwarded to label_events().
    include_touch_events : bool
        If True, also generate touch-based events (price touching S/R
        and channel boundaries) and label them with triple-barrier.
    touch_atr_mult : float
        ATR multiplier for touch proximity (default 0.2).
    touch_cooldown : int
        Cooldown between touch events (default 10).

    Returns
    -------
    features : pd.DataFrame
        Feature matrix (one row per event, columns = features).
    labels : pd.Series
        Target labels ("long", "short", "no_trade").
    labeled_df : pd.DataFrame
        Full labeled events DataFrame.
    """
    # Step 1: Compute all bar-level indicators
    indicators = compute_all_indicators(df)

    # Step 2: Run detectors with details for touch counting
    _, tri_details = detect_triangle_pattern(df, return_details=True)
    _, ch_details = detect_channel(df, return_details=True)

    # Step 3: Label events (original detector events)
    labeled = label_events(df, pt_mult=pt_mult, sl_mult=sl_mult,
                           max_holding=max_holding,
                           exclude_patterns=exclude_patterns)

    # Mark original events
    if len(labeled) > 0:
        labeled["event_source"] = "detector"

    # Step 3b: Optionally add touch-based events
    if include_touch_events and len(labeled) > 0:
        touch_labeled = _generate_touch_labels(
            df, pt_mult=pt_mult, sl_mult=sl_mult,
            max_holding=max_holding, atr_mult=touch_atr_mult,
            cooldown=touch_cooldown,
        )
        if len(touch_labeled) > 0:
            touch_labeled["event_source"] = "touch"
            labeled = pd.concat([labeled, touch_labeled],
                                ignore_index=True)
            # Remove duplicates (same event_date)
            labeled = labeled.drop_duplicates(subset="event_date", keep="first")
            labeled = labeled.sort_values("event_date").reset_index(drop=True)

    if len(labeled) == 0:
        return pd.DataFrame(), pd.Series(dtype=str), labeled

    # Step 4: For each event, pull bar-level indicators at event date
    event_dates = pd.DatetimeIndex(labeled["event_date"])
    # Only keep dates that exist in the indicator index
    valid_mask = event_dates.isin(indicators.index)
    labeled = labeled.loc[valid_mask].reset_index(drop=True)
    event_dates = pd.DatetimeIndex(labeled["event_date"])

    bar_features = indicators.loc[event_dates].reset_index(drop=True)

    # Step 5: Pattern geometry features
    geo_features = _pattern_geometry_features(labeled, tri_details, ch_details)

    # Step 6: Event type dummies
    type_dummies = _event_type_dummies(labeled)

    # Step 7: Combine all feature groups.
    # NOTE: entry_price and event_atr are NOT included — both scale with
    # SPY's price level / time trend and would leak temporal information.
    # Volatility regime is already captured by atr_ratio (ATR/Close).
    features = pd.concat([bar_features, geo_features, type_dummies],
                         axis=1)

    # Drop absolute SMA values — they trend with price and are time proxies.
    # The _dist (relative distance) versions are kept.
    abs_sma_cols = [c for c in features.columns
                    if c.startswith("sma_") and "_dist" not in c]
    features = features.drop(columns=abs_sma_cols, errors="ignore")

    # Drop any columns that are all NaN
    features = features.dropna(axis=1, how="all")

    labels = labeled["label"]

    return features, labels, labeled


def _generate_touch_labels(df, pt_mult=2.0, sl_mult=2.0, max_holding=10,
                           atr_mult=0.2, cooldown=10):
    """Generate and label touch-based events.

    Returns a labeled DataFrame in the same format as label_events().
    """
    from src.patterns.touch_events import generate_all_touch_events, _get_touch_event_type

    df_touch, stats = generate_all_touch_events(
        df, atr_mult=atr_mult, cooldown=cooldown,
    )

    # Get touch-only events (not already flagged by standard detectors)
    touch_only = df_touch[df_touch["has_touch_event"] & ~df_touch["has_event"]].copy()

    if len(touch_only) == 0:
        return pd.DataFrame()

    # Label with triple-barrier
    from src.data.utils import compute_atr
    atr = compute_atr(df, window=14)

    results = []
    for event_date, row in touch_only.iterrows():
        pos = df.index.get_loc(event_date)
        entry_price = df["Close"].iloc[pos]
        atr_val = atr.iloc[pos]

        if pd.isna(atr_val) or atr_val <= 0:
            continue

        upper = entry_price + pt_mult * atr_val
        lower = entry_price - sl_mult * atr_val

        end_pos = min(pos + max_holding, len(df) - 1)
        label = "no_trade"
        exit_price = df["Close"].iloc[end_pos]
        exit_date = df.index[end_pos]
        bars_held = end_pos - pos

        for j in range(pos + 1, min(pos + max_holding + 1, len(df))):
            high_j = df["High"].iloc[j]
            low_j = df["Low"].iloc[j]
            hit_upper = high_j >= upper
            hit_lower = low_j <= lower

            if hit_upper and hit_lower:
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
            "event_type": _get_touch_event_type(row),
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
