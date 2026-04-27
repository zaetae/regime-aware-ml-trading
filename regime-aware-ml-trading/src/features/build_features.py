"""Build event-based feature matrix for ML training.

Each row represents one labeled event. Features use only information
available at or before the event bar (no lookahead).
"""

import numpy as np
import pandas as pd

from src.features.indicators import compute_all_indicators
from src.data.utils import compute_atr
from src.patterns.scanner import scan_all_patterns
from src.patterns.triangles import detect_triangle_pattern
from src.patterns.channels import detect_channel
from src.labeling.label_events import label_events


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
                         pt_mult=2.0, sl_mult=2.0, max_holding=10):
    """Build the full feature matrix for labeled events.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data.
    exclude_patterns : list, optional
        Pattern types to exclude from labeling.
    pt_mult, sl_mult, max_holding
        Forwarded to label_events().

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

    # Step 3: Label events
    labeled = label_events(df, pt_mult=pt_mult, sl_mult=sl_mult,
                           max_holding=max_holding,
                           exclude_patterns=exclude_patterns)

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

    # Step 7: Event-level features from the labeled DataFrame.
    # NOTE: entry_price is NOT included — it is a proxy for time (SPY
    # trends upward) and would leak temporal information into the model.
    # event_atr is kept because it captures current volatility regime.
    event_meta = pd.DataFrame({
        "event_atr": labeled["atr"].values,
    })

    # Combine all feature groups
    features = pd.concat([bar_features, geo_features, type_dummies, event_meta],
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
