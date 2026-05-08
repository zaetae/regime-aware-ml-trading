"""Hyperparameter optimization for triple-barrier labeling parameters.

Treats pt_mult, sl_mult, and max_holding as tunable hyperparameters rather
than fixed constants. Supports two optimization backends:

1. Optuna (if installed) — Bayesian TPE sampler, efficient for small budgets
2. Grid search fallback — exhaustive search over a discrete grid

Optimization targets:
- "f1_macro"         : macro-averaged F1 on validation set
- "accuracy"         : classification accuracy on validation set
- "cumulative_return": total return from simulated trading
- "sharpe"           : Sharpe ratio of simulated trades
- "profit_factor"    : gross profit / gross loss

Performance optimization: pattern scanning and indicator computation are
done ONCE upfront.  Only the labeling step (which depends on pt_mult,
sl_mult, max_holding) is re-run per trial.

No look-ahead leakage: labeling walks forward from signal-bar close,
and the train/val split is strictly chronological.
"""

import warnings
import numpy as np
import pandas as pd

from src.features.indicators import compute_all_indicators
from src.features.build_features import _pattern_geometry_features, _event_type_dummies
from src.patterns.scanner import scan_all_patterns
from src.patterns.triangles import detect_triangle_pattern
from src.patterns.channels import detect_channel
from src.labeling.label_events import label_events
from src.models.train import (
    temporal_split, train_random_forest, evaluate_model,
)
from src.backtest.simulator import evaluate_profitability


# ------------------------------------------------------------------
# Pre-computation cache
# ------------------------------------------------------------------

def _precompute(df, exclude_patterns=None):
    """Pre-compute expensive operations that don't depend on barrier params.

    Returns indicators, scanned df, and pattern details.
    """
    indicators = compute_all_indicators(df)
    df_scanned = scan_all_patterns(df)
    _, tri_details = detect_triangle_pattern(df, return_details=True)
    _, ch_details = detect_channel(df, return_details=True)
    return {
        "indicators": indicators,
        "df_scanned": df_scanned,
        "tri_details": tri_details,
        "ch_details": ch_details,
        "exclude_patterns": exclude_patterns,
    }


def _build_features_fast(df, cache, pt_mult, sl_mult, max_holding):
    """Build feature matrix using pre-computed data.

    Only the labeling step (which depends on barrier params) is re-run.
    """
    indicators = cache["indicators"]
    tri_details = cache["tri_details"]
    ch_details = cache["ch_details"]
    exclude_patterns = cache["exclude_patterns"]

    # Re-label with candidate params (uses pre-scanned df)
    labeled = label_events(
        cache["df_scanned"], pt_mult=pt_mult, sl_mult=sl_mult,
        max_holding=max_holding, exclude_patterns=exclude_patterns,
    )

    if len(labeled) == 0:
        return pd.DataFrame(), pd.Series(dtype=str), labeled

    # Pull bar-level indicators at event dates
    event_dates = pd.DatetimeIndex(labeled["event_date"])
    valid_mask = event_dates.isin(indicators.index)
    labeled = labeled.loc[valid_mask].reset_index(drop=True)
    event_dates = pd.DatetimeIndex(labeled["event_date"])

    bar_features = indicators.loc[event_dates].reset_index(drop=True)

    # Pattern geometry
    geo_features = _pattern_geometry_features(labeled, tri_details, ch_details)

    # Event type dummies
    type_dummies = _event_type_dummies(labeled)

    # Combine
    features = pd.concat([bar_features, geo_features, type_dummies], axis=1)

    # Drop absolute SMAs
    abs_sma_cols = [c for c in features.columns
                    if c.startswith("sma_") and "_dist" not in c]
    features = features.drop(columns=abs_sma_cols, errors="ignore")
    features = features.dropna(axis=1, how="all")

    labels = labeled["label"]
    return features, labels, labeled


# ------------------------------------------------------------------
# Objective function (shared by Optuna and grid search)
# ------------------------------------------------------------------

def _run_trial(df, cache, pt_mult, sl_mult, max_holding, target,
               train_frac=0.6, val_frac=0.2,
               n_estimators=100, max_depth=8, random_state=42):
    """Run one optimization trial with given barrier parameters.

    Uses pre-computed cache for speed.

    Returns
    -------
    score : float
        The target metric value (higher is better for all targets).
    info : dict
        Detailed results for logging.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        features, labels, labeled_df = _build_features_fast(
            df, cache, pt_mult, sl_mult, max_holding,
        )

        if len(features) < 20:
            return -999.0, {"error": "too_few_events"}

        features = features.fillna(0)

        # Chronological split
        split = temporal_split(features, labels, labeled_df,
                               train_frac=train_frac, val_frac=val_frac)

        X_train = split["X_train"]
        X_val = split["X_val"]
        y_train = split["y_train"]
        y_val = split["y_val"]

        if len(y_train.unique()) < 2 or len(y_val) < 5:
            return -999.0, {"error": "degenerate_split"}

        # Train RF (smaller ensemble for speed)
        rf = train_random_forest(X_train, y_train,
                                 n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 random_state=random_state)

        # Classification metrics
        eval_res = evaluate_model(rf, X_val, y_val, "RF")

        info = {
            "pt_mult": pt_mult,
            "sl_mult": sl_mult,
            "max_holding": max_holding,
            "n_events": len(features),
            "accuracy": eval_res["accuracy"],
            "f1_macro": eval_res["f1_macro"],
            "label_dist": labels.value_counts().to_dict(),
        }

        # Profitability metrics (only computed if needed)
        if target in ("cumulative_return", "sharpe", "profit_factor",
                       "win_rate"):
            dates = pd.DatetimeIndex(labeled_df["event_date"])
            sort_idx = dates.argsort()
            sorted_labeled = labeled_df.iloc[sort_idx].reset_index(drop=True)

            n = len(sorted_labeled)
            n_train = int(n * train_frac)
            n_val = int(n * (train_frac + val_frac))
            val_labeled = sorted_labeled.iloc[n_train:n_val]

            y_pred_val = rf.predict(X_val)
            profit_metrics, _ = evaluate_profitability(
                df, val_labeled, y_pred_val,
                pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding,
            )
            info.update({
                "cumulative_return": profit_metrics["cumulative_return"],
                "sharpe": profit_metrics["sharpe_ratio"],
                "profit_factor": profit_metrics["profit_factor"],
                "win_rate": profit_metrics["win_rate"],
                "n_trades": profit_metrics["n_trades"],
            })

        # Select score based on target
        if target == "f1_macro":
            score = eval_res["f1_macro"]
        elif target == "accuracy":
            score = eval_res["accuracy"]
        elif target in ("cumulative_return", "sharpe", "profit_factor",
                        "win_rate"):
            score = info.get(target, -999.0)
            if score == "inf":
                score = 10.0  # cap infinite profit factor
        else:
            score = eval_res["f1_macro"]  # default

        return float(score), info


# ------------------------------------------------------------------
# Grid search
# ------------------------------------------------------------------

def grid_search(df, target="f1_macro",
                pt_range=None, sl_range=None, holding_range=None,
                exclude_patterns=None, include_touch_events=False,
                train_frac=0.6, val_frac=0.2,
                n_estimators=100, max_depth=8, random_state=42,
                verbose=True):
    """Exhaustive grid search over triple-barrier parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV data.
    target : str
        Metric to maximize. One of: "f1_macro", "accuracy",
        "cumulative_return", "sharpe", "profit_factor".
    pt_range : list[float]
        Profit target multipliers to try (default [1.0, 1.5, 2.0, 2.5, 3.0]).
    sl_range : list[float]
        Stop loss multipliers to try (default [1.0, 1.5, 2.0, 2.5, 3.0]).
    holding_range : list[int]
        Max holding periods to try (default [5, 10, 15, 20]).
    verbose : bool
        Print progress.

    Returns
    -------
    best_params : dict
        Best parameter combination found.
    best_score : float
        Best score achieved.
    all_results : pd.DataFrame
        All trial results.
    """
    if pt_range is None:
        pt_range = [1.0, 1.5, 2.0, 2.5, 3.0]
    if sl_range is None:
        sl_range = [1.0, 1.5, 2.0, 2.5, 3.0]
    if holding_range is None:
        holding_range = [5, 10, 15, 20]

    total = len(pt_range) * len(sl_range) * len(holding_range)

    if verbose:
        print(f"  Pre-computing indicators and pattern scans...")

    cache = _precompute(df, exclude_patterns=exclude_patterns)

    if verbose:
        print(f"  Pre-computation done. Running {total} trials...")

    results = []
    best_score = -999.0
    best_params = {}
    trial_num = 0

    for pt in pt_range:
        for sl in sl_range:
            for mh in holding_range:
                trial_num += 1
                score, info = _run_trial(
                    df, cache, pt, sl, mh, target,
                    train_frac=train_frac, val_frac=val_frac,
                    n_estimators=n_estimators, max_depth=max_depth,
                    random_state=random_state,
                )

                info["score"] = score
                results.append(info)

                if score > best_score:
                    best_score = score
                    best_params = {"pt_mult": pt, "sl_mult": sl,
                                   "max_holding": mh}

                if verbose and trial_num % 10 == 0:
                    print(f"  Trial {trial_num}/{total}: "
                          f"pt={pt}, sl={sl}, mh={mh} -> {target}={score:.4f}")

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\nBest {target}: {best_score:.4f}")
        print(f"Best params: {best_params}")

    return best_params, best_score, results_df


# ------------------------------------------------------------------
# Optuna-based optimization
# ------------------------------------------------------------------

def optuna_search(df, target="f1_macro", n_trials=50,
                  pt_range=(1.0, 3.0), sl_range=(1.0, 3.0),
                  holding_range=(5, 20),
                  exclude_patterns=None, include_touch_events=False,
                  train_frac=0.6, val_frac=0.2,
                  n_estimators=100, max_depth=8, random_state=42,
                  verbose=True):
    """Bayesian optimization using Optuna's TPE sampler.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV data.
    target : str
        Metric to maximize.
    n_trials : int
        Number of trials (default 50).
    pt_range : tuple
        (min, max) for pt_mult.
    sl_range : tuple
        (min, max) for sl_mult.
    holding_range : tuple
        (min, max) for max_holding.

    Returns
    -------
    best_params : dict
    best_score : float
    study : optuna.Study
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is not installed. Install with: pip install optuna\n"
            "Falling back to grid_search() is recommended."
        )

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if verbose:
        print("  Pre-computing indicators and pattern scans...")

    cache = _precompute(df, exclude_patterns=exclude_patterns)

    if verbose:
        print("  Pre-computation done. Running Optuna optimization...")

    def objective(trial):
        pt = trial.suggest_float("pt_mult", pt_range[0], pt_range[1], step=0.25)
        sl = trial.suggest_float("sl_mult", sl_range[0], sl_range[1], step=0.25)
        mh = trial.suggest_int("max_holding", holding_range[0], holding_range[1],
                               step=5)

        score, info = _run_trial(
            df, cache, pt, sl, mh, target,
            train_frac=train_frac, val_frac=val_frac,
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state,
        )

        trial.set_user_attr("info", info)
        return score

    study = optuna.create_study(direction="maximize",
                                study_name=f"barrier_opt_{target}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best_params = study.best_params
    best_params["max_holding"] = int(best_params["max_holding"])
    best_score = study.best_value

    if verbose:
        print(f"\nBest {target}: {best_score:.4f}")
        print(f"Best params: {best_params}")

    return best_params, best_score, study


# ------------------------------------------------------------------
# Convenience wrapper
# ------------------------------------------------------------------

def optimize_barriers(df, target="f1_macro", method="auto", n_trials=50,
                      pt_range=None, sl_range=None, holding_range=None,
                      exclude_patterns=None, include_touch_events=False,
                      train_frac=0.6, val_frac=0.2,
                      n_estimators=100, max_depth=8, random_state=42,
                      verbose=True):
    """Optimize triple-barrier parameters.

    Parameters
    ----------
    method : str
        "optuna" — use Optuna TPE sampler
        "grid"   — exhaustive grid search
        "auto"   — Optuna if available, otherwise grid search

    Returns
    -------
    best_params : dict
        {"pt_mult": ..., "sl_mult": ..., "max_holding": ...}
    best_score : float
    details : object
        Grid search DataFrame or Optuna study.
    """
    if method == "auto":
        try:
            import optuna  # noqa: F401
            method = "optuna"
        except ImportError:
            method = "grid"

    if method == "optuna":
        if pt_range is None:
            pt_range = (1.0, 3.0)
        if sl_range is None:
            sl_range = (1.0, 3.0)
        if holding_range is None:
            holding_range = (5, 20)

        return optuna_search(
            df, target=target, n_trials=n_trials,
            pt_range=pt_range, sl_range=sl_range,
            holding_range=holding_range,
            exclude_patterns=exclude_patterns,
            include_touch_events=include_touch_events,
            train_frac=train_frac, val_frac=val_frac,
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, verbose=verbose,
        )
    else:
        # Convert ranges to lists for grid search
        if pt_range is None:
            pt_list = [1.0, 1.5, 2.0, 2.5, 3.0]
        elif isinstance(pt_range, (list, tuple)) and len(pt_range) == 2:
            pt_list = list(np.arange(pt_range[0], pt_range[1] + 0.01, 0.5))
        else:
            pt_list = list(pt_range)

        if sl_range is None:
            sl_list = [1.0, 1.5, 2.0, 2.5, 3.0]
        elif isinstance(sl_range, (list, tuple)) and len(sl_range) == 2:
            sl_list = list(np.arange(sl_range[0], sl_range[1] + 0.01, 0.5))
        else:
            sl_list = list(sl_range)

        if holding_range is None:
            holding_list = [5, 10, 15, 20]
        elif isinstance(holding_range, (list, tuple)) and len(holding_range) == 2:
            holding_list = list(range(holding_range[0], holding_range[1] + 1, 5))
        else:
            holding_list = list(holding_range)

        return grid_search(
            df, target=target,
            pt_range=pt_list, sl_range=sl_list,
            holding_range=holding_list,
            exclude_patterns=exclude_patterns,
            include_touch_events=include_touch_events,
            train_frac=train_frac, val_frac=val_frac,
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, verbose=verbose,
        )
