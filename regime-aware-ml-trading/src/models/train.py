"""Random Forest and Bagging classifier training for event-based labeling.

Uses time-aware train/validation/test splits to avoid temporal leakage.
Provides comprehensive evaluation: confusion matrix, classification report,
feature importance, tree complexity stats, and comparison between models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder


# ------------------------------------------------------------------
# Time-aware splitting
# ------------------------------------------------------------------

def temporal_split(features, labels, labeled_df,
                   train_frac=0.6, val_frac=0.2):
    """Split data chronologically into train / validation / test.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix (rows aligned with labeled_df).
    labels : pd.Series
        Target labels.
    labeled_df : pd.DataFrame
        Contains 'event_date' for ordering.
    train_frac, val_frac : float
        Fractions for train and validation. Test = remainder.

    Returns
    -------
    dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                    dates_train, dates_val, dates_test
    """
    dates = pd.DatetimeIndex(labeled_df["event_date"])
    sort_idx = dates.argsort()

    features = features.iloc[sort_idx].reset_index(drop=True)
    labels = labels.iloc[sort_idx].reset_index(drop=True)
    dates = dates[sort_idx]

    n = len(features)
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))

    split = {
        "X_train": features.iloc[:n_train],
        "X_val": features.iloc[n_train:n_val],
        "X_test": features.iloc[n_val:],
        "y_train": labels.iloc[:n_train],
        "y_val": labels.iloc[n_train:n_val],
        "y_test": labels.iloc[n_val:],
        "dates_train": dates[:n_train],
        "dates_val": dates[n_train:n_val],
        "dates_test": dates[n_val:],
    }
    return split


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_random_forest(X_train, y_train, n_estimators=200,
                        max_depth=8, random_state=42, **kwargs):
    """Train a Random Forest classifier."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
        **kwargs,
    )
    clf.fit(X_train, y_train)
    return clf


def train_bagging(X_train, y_train, n_estimators=200,
                  max_depth=8, random_state=42, **kwargs):
    """Train a Bagging classifier with decision tree base estimators.

    Unlike Random Forest, Bagging uses all features at each split
    (no random feature subset). The only source of diversity is the
    bootstrap sample.
    """
    base_tree = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
    )
    clf = BaggingClassifier(
        estimator=base_tree,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )
    clf.fit(X_train, y_train)
    return clf


def train_baseline(X_train, y_train, strategy="stratified", random_state=42):
    """Train a DummyClassifier as a baseline."""
    clf = DummyClassifier(strategy=strategy, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_model(clf, X, y, model_name="Model"):
    """Evaluate a classifier and return a results dict."""
    y_pred = clf.predict(X)
    y_proba = None
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X)

    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=sorted(y.unique()))
    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y, y_pred, average="weighted", zero_division=0)

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "labels": sorted(y.unique()),
    }


def tree_complexity_stats(clf, model_name="Model"):
    """Extract tree depth and complexity statistics from ensemble.

    Works for RandomForestClassifier and BaggingClassifier (with trees).
    """
    if hasattr(clf, "estimators_"):
        trees = clf.estimators_
    else:
        return None

    # For BaggingClassifier, estimators_ may contain other estimators
    depths = []
    n_leaves = []
    n_nodes = []

    for t in trees:
        tree = t.tree_ if hasattr(t, "tree_") else None
        if tree is None:
            continue
        depths.append(tree.max_depth)
        n_leaves.append(tree.n_leaves)
        n_nodes.append(tree.node_count)

    if not depths:
        return None

    return {
        "model_name": model_name,
        "n_trees": len(trees),
        "min_depth": int(np.min(depths)),
        "mean_depth": round(float(np.mean(depths)), 1),
        "max_depth": int(np.max(depths)),
        "min_leaves": int(np.min(n_leaves)),
        "mean_leaves": round(float(np.mean(n_leaves)), 1),
        "max_leaves": int(np.max(n_leaves)),
        "min_nodes": int(np.min(n_nodes)),
        "mean_nodes": round(float(np.mean(n_nodes)), 1),
        "max_nodes": int(np.max(n_nodes)),
    }


def feature_importance_table(clf, feature_names, top_n=20):
    """Return a DataFrame of feature importances (sorted descending).

    Works for RandomForestClassifier. For Bagging, averages across trees.
    """
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "estimators_"):
        # Bagging: average importances across constituent trees
        imp_list = []
        for t in clf.estimators_:
            if hasattr(t, "feature_importances_"):
                imp_list.append(t.feature_importances_)
        if not imp_list:
            return pd.DataFrame()
        importances = np.mean(imp_list, axis=0)
    else:
        return pd.DataFrame()

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return fi.head(top_n) if top_n else fi


# ------------------------------------------------------------------
# Walk-forward cross-validation
# ------------------------------------------------------------------

def walk_forward_cv(features, labels, labeled_df,
                    n_splits=5, min_train_frac=0.30,
                    n_estimators=200, max_depth=8, random_state=42):
    """Expanding-window walk-forward cross-validation.

    Splits the chronologically ordered events into ``n_splits`` folds.
    For fold *k*, training uses all events up to fold *k*, and testing
    uses fold *k+1*.  The training window grows with each fold.

    Returns
    -------
    dict with per-fold and aggregate results.
    """
    dates = pd.DatetimeIndex(labeled_df["event_date"])
    sort_idx = dates.argsort()
    features = features.iloc[sort_idx].reset_index(drop=True).fillna(0)
    labels = labels.iloc[sort_idx].reset_index(drop=True)
    dates = dates[sort_idx]

    n = len(features)
    fold_size = n // (n_splits + 1)
    if fold_size < 5:
        return None  # too few events for meaningful CV

    fold_results = []
    for k in range(n_splits):
        train_end = fold_size * (k + 1)
        test_start = train_end
        test_end = min(train_end + fold_size, n)
        if test_end <= test_start or train_end < int(n * min_train_frac):
            continue

        X_tr = features.iloc[:train_end]
        y_tr = labels.iloc[:train_end]
        X_te = features.iloc[test_start:test_end]
        y_te = labels.iloc[test_start:test_end]

        if len(y_te.unique()) < 2:
            continue

        rf = train_random_forest(X_tr, y_tr, n_estimators=n_estimators,
                                 max_depth=max_depth, random_state=random_state)
        baseline = train_baseline(X_tr, y_tr, random_state=random_state)

        rf_res = evaluate_model(rf, X_te, y_te, f"RF fold-{k}")
        base_res = evaluate_model(baseline, X_te, y_te, f"Base fold-{k}")

        fold_results.append({
            "fold": k,
            "train_size": len(X_tr),
            "test_size": len(X_te),
            "train_end_date": dates[train_end - 1],
            "test_end_date": dates[test_end - 1],
            "rf_accuracy": rf_res["accuracy"],
            "rf_f1_macro": rf_res["f1_macro"],
            "base_accuracy": base_res["accuracy"],
            "base_f1_macro": base_res["f1_macro"],
        })

    if not fold_results:
        return None

    wf_df = pd.DataFrame(fold_results)
    return {
        "folds": wf_df,
        "rf_mean_acc": round(wf_df["rf_accuracy"].mean(), 4),
        "rf_mean_f1": round(wf_df["rf_f1_macro"].mean(), 4),
        "base_mean_acc": round(wf_df["base_accuracy"].mean(), 4),
        "base_mean_f1": round(wf_df["base_f1_macro"].mean(), 4),
        "n_folds": len(fold_results),
    }


# ------------------------------------------------------------------
# Full training pipeline
# ------------------------------------------------------------------

def run_training_pipeline(features, labels, labeled_df,
                          train_frac=0.6, val_frac=0.2,
                          n_estimators=200, max_depth=8,
                          random_state=42):
    """Run the complete training and evaluation pipeline.

    Returns
    -------
    results : dict with all models, evaluations, and statistics.
    """
    # Handle NaN in features
    features = features.fillna(0)

    # Split
    split = temporal_split(features, labels, labeled_df,
                           train_frac=train_frac, val_frac=val_frac)

    X_train = split["X_train"]
    X_val = split["X_val"]
    X_test = split["X_test"]
    y_train = split["y_train"]
    y_val = split["y_val"]
    y_test = split["y_test"]

    # Train models
    rf = train_random_forest(X_train, y_train, n_estimators=n_estimators,
                             max_depth=max_depth, random_state=random_state)
    bag = train_bagging(X_train, y_train, n_estimators=n_estimators,
                        max_depth=max_depth, random_state=random_state)
    baseline = train_baseline(X_train, y_train, random_state=random_state)

    # Evaluate on validation set
    rf_val = evaluate_model(rf, X_val, y_val, "Random Forest (val)")
    bag_val = evaluate_model(bag, X_val, y_val, "Bagging (val)")
    base_val = evaluate_model(baseline, X_val, y_val, "Baseline (val)")

    # Evaluate on test set
    rf_test = evaluate_model(rf, X_test, y_test, "Random Forest (test)")
    bag_test = evaluate_model(bag, X_test, y_test, "Bagging (test)")
    base_test = evaluate_model(baseline, X_test, y_test, "Baseline (test)")

    # Tree stats
    rf_trees = tree_complexity_stats(rf, "Random Forest")
    bag_trees = tree_complexity_stats(bag, "Bagging")

    # Feature importance
    feature_names = list(features.columns)
    rf_fi = feature_importance_table(rf, feature_names)
    bag_fi = feature_importance_table(bag, feature_names)

    # Walk-forward cross-validation
    wf = walk_forward_cv(features, labels, labeled_df,
                         n_splits=5, n_estimators=n_estimators,
                         max_depth=max_depth, random_state=random_state)

    return {
        "split": split,
        "models": {"rf": rf, "bagging": bag, "baseline": baseline},
        "val_results": {"rf": rf_val, "bagging": bag_val, "baseline": base_val},
        "test_results": {"rf": rf_test, "bagging": bag_test, "baseline": base_test},
        "tree_stats": {"rf": rf_trees, "bagging": bag_trees},
        "feature_importance": {"rf": rf_fi, "bagging": bag_fi},
        "feature_names": feature_names,
        "walk_forward": wf,
        "hyperparams": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "train_frac": train_frac,
            "val_frac": val_frac,
        },
    }
