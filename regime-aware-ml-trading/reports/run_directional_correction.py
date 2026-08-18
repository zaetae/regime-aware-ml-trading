"""Reproduce the directional-label correction in an isolated results folder.

This script intentionally does not modify ``outputs/`` or existing notebooks.
It covers the workflow used by notebooks 12 and 13: detection, corrected
labeling, feature construction, two 100-trial barrier grids, and walk-forward
evaluation. Run from the nested project root:

    PYTHONPATH=. python reports/run_directional_correction.py \
        --results-dir results/directional_label_correction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.load_data import load_spy
from src.features.build_features import build_feature_matrix
from src.models.optimize import grid_search
from src.models.train import run_training_pipeline, walk_forward_cv


EXCLUDE = ["triangle_pattern", "channel_pattern"]
GRID = {
    "pt_range": [1.0, 1.5, 2.0, 2.5, 3.0],
    "sl_range": [1.0, 1.5, 2.0, 2.5, 3.0],
    "holding_range": [5, 10, 15, 20],
}


def _serialise(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _save_json(path: Path, value) -> None:
    path.write_text(json.dumps(_serialise(value), indent=2, default=str) + "\n")


def _event_split_dates(labeled_df: pd.DataFrame) -> dict:
    dates = pd.DatetimeIndex(labeled_df["event_date"]).sort_values()
    n = len(dates)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    return {
        "n_events": n,
        "train": {"n": train_end, "start": dates[0], "end": dates[train_end - 1]},
        "validation": {
            "n": val_end - train_end,
            "start": dates[train_end],
            "end": dates[val_end - 1],
        },
        "test": {"n": n - val_end, "start": dates[val_end], "end": dates[-1]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.results_dir
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing results directory: {out}")
    out.mkdir(parents=True)

    df = load_spy()
    features, labels, labeled = build_feature_matrix(
        df, exclude_patterns=EXCLUDE,
    )
    features = features.fillna(0)

    label_distribution = (
        labels.value_counts().rename_axis("label").reset_index(name="count")
    )
    label_distribution["share"] = label_distribution["count"] / len(labels)
    label_distribution.to_csv(out / "label_distribution_corrected.csv", index=False)
    labeled.to_csv(out / "labeled_events_corrected.csv", index=False)
    _save_json(out / "split_dates.json", _event_split_dates(labeled))

    best_f1_params, best_f1_score, f1_grid = grid_search(
        df, target="f1_macro", exclude_patterns=EXCLUDE,
        n_estimators=100, random_state=42, verbose=True, **GRID,
    )
    f1_grid.to_csv(out / "grid_f1_macro.csv", index=False)

    best_profit_params, best_profit_score, profit_grid = grid_search(
        df, target="cumulative_return", exclude_patterns=EXCLUDE,
        n_estimators=100, random_state=42, verbose=True, **GRID,
    )
    profit_grid.to_csv(out / "grid_cumulative_return.csv", index=False)

    best_features, best_labels, best_labeled = build_feature_matrix(
        df, exclude_patterns=EXCLUDE, **best_profit_params,
    )
    best_features = best_features.fillna(0)
    training = run_training_pipeline(
        best_features, best_labels, best_labeled,
        df_ohlcv=df, random_state=42, **best_profit_params,
    )

    wf = walk_forward_cv(
        best_features, best_labels, best_labeled,
        n_splits=5, n_estimators=200, max_depth=8, random_state=42,
        df_ohlcv=df, **best_profit_params,
    )
    if wf is not None:
        wf["folds"].to_csv(out / "walk_forward_best_profit.csv", index=False)

    test_profit = training["profitability"]["rf"]["test"]
    comparison = pd.DataFrame([
        {
            "version": "old_notebook_12_recorded",
            "label_distribution": "not persisted for pt=2.0/sl=3.0/mh=20",
            "f1_macro": None,
            "cumulative_return": 0.2866,
            "win_rate": None,
            "scope": "validation grid selection; notebook cell 8",
        },
        {
            "version": "corrected",
            "label_distribution": labels.value_counts().to_dict(),
            "f1_macro": training["test_results"]["rf"]["f1_macro"],
            "cumulative_return": test_profit["cumulative_return"],
            "win_rate": test_profit["win_rate"],
            "scope": "held-out test, best corrected-profit configuration",
        },
    ])
    comparison.to_csv(out / "before_after_comparison.csv", index=False)

    _save_json(out / "summary.json", {
        "data_start": df.index.min(),
        "data_end": df.index.max(),
        "direction_mapping": {
            "near_support": "long", "multiple_bottom": "long",
            "near_resistance": "short", "multiple_top": "short",
            "channel_up": "long", "channel_down": "short",
            "triangle_breakout_up": "long", "triangle_breakout_down": "short",
        },
        "best_f1": {"params": best_f1_params, "validation_score": best_f1_score},
        "best_profit": {
            "params": best_profit_params,
            "validation_cumulative_return": best_profit_score,
            "test_profitability_rf": test_profit,
        },
        "test_classification_rf": training["test_results"]["rf"],
        "walk_forward": (
            {key: value for key, value in wf.items() if key not in {"folds", "fold_details"}}
            if wf is not None else None
        ),
    })


if __name__ == "__main__":
    main()
