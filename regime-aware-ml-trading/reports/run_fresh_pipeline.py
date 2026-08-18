from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.load_data import load_spy
from src.features.build_features import build_feature_matrix
from src.labeling.label_events import label_events
from src.models.optimize import grid_search
from src.models.train import run_training_pipeline, walk_forward_cv
from src.patterns.scanner import scan_all_patterns
from src.patterns.touch_events import generate_all_touch_events


ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / 'results'
GRID = {
    'pt_range': [1.0, 1.5, 2.0, 2.5, 3.0],
    'sl_range': [1.0, 1.5, 2.0, 2.5, 3.0],
    'holding_range': [5, 10, 15, 20],
}
EXCLUDE = ['triangle_pattern', 'channel_pattern']
HISTORIC = {'long': 58, 'short': 43, 'no_trade': 36}


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str) + '\n')


def main():
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = RESULTS_ROOT / f'fresh_pipeline_run_{run_id}'
    out_dir.mkdir(parents=True, exist_ok=False)

    raw = load_spy()

    scanned = scan_all_patterns(raw)
    scanned.to_csv(out_dir / '01_scan_all_patterns.csv', index=True)

    touch_df, touch_stats = generate_all_touch_events(raw.copy())
    touch_df.to_csv(out_dir / '02_touch_events.csv', index=True)
    save_json(out_dir / '02_touch_stats.json', touch_stats)

    labeled = label_events(raw.copy())
    labeled.to_csv(out_dir / '03_labeled_events.csv', index=False)
    label_dist = labeled['label'].value_counts().rename_axis('label').reset_index(name='count')
    label_dist = label_dist.sort_values('label').reset_index(drop=True)
    label_dist['share'] = label_dist['count'] / len(labeled)
    label_dist.to_csv(out_dir / '03_label_distribution.csv', index=False)

    features, labels, labeled_full = build_feature_matrix(raw.copy(), exclude_patterns=EXCLUDE)
    features = features.fillna(0)
    features.to_csv(out_dir / '04_features.csv', index=False)
    labels.to_csv(out_dir / '04_labels.csv', header=['label'], index=True)
    labeled_full.to_csv(out_dir / '04_labeled_feature_rows.csv', index=False)

    best_f1_params, best_f1_score, f1_grid = grid_search(
        raw.copy(),
        target='f1_macro',
        exclude_patterns=EXCLUDE,
        n_estimators=100,
        random_state=42,
        verbose=False,
        **GRID,
    )
    f1_grid.to_csv(out_dir / '05_grid_f1_macro.csv', index=False)

    best_profit_params, best_profit_score, profit_grid = grid_search(
        raw.copy(),
        target='cumulative_return',
        exclude_patterns=EXCLUDE,
        n_estimators=100,
        random_state=42,
        verbose=False,
        **GRID,
    )
    profit_grid.to_csv(out_dir / '05_grid_cumulative_return.csv', index=False)

    best_features, best_labels, best_labeled = build_feature_matrix(
        raw.copy(),
        exclude_patterns=EXCLUDE,
        **best_profit_params,
    )
    best_features = best_features.fillna(0)

    training = run_training_pipeline(
        best_features,
        best_labels,
        best_labeled,
        df_ohlcv=raw.copy(),
        random_state=42,
        **best_profit_params,
    )
    save_json(out_dir / '06_train_summary.json', training)

    wf = walk_forward_cv(
        best_features,
        best_labels,
        best_labeled,
        n_splits=5,
        n_estimators=200,
        max_depth=8,
        random_state=42,
        df_ohlcv=raw.copy(),
        **best_profit_params,
    )
    if wf is not None:
        if isinstance(wf, dict) and 'folds' in wf:
            pd.DataFrame(wf['folds']).to_csv(out_dir / '07_walk_forward.csv', index=False)
        save_json(out_dir / '07_walk_forward_summary.json', {
            k: v for k, v in wf.items() if k not in {'folds', 'fold_details'}
        } if isinstance(wf, dict) else wf)

    summary = {
        'run_id': run_id,
        'step_order': [
            'data_loading',
            'scan_all_patterns',
            'generate_all_touch_events',
            'label_events',
            'build_feature_matrix',
            'grid_search_f1_macro',
            'grid_search_cumulative_return',
            'walk_forward_cv',
        ],
        'label_distribution': label_dist.set_index('label')['count'].to_dict(),
        'label_distribution_share': label_dist.set_index('label')['share'].to_dict(),
        'historic_report_distribution': HISTORIC,
        'best_f1_params': best_f1_params,
        'best_f1_score': float(best_f1_score),
        'best_profit_params': best_profit_params,
        'best_profit_score': float(best_profit_score),
        'test_classification_rf': training['test_results']['rf'],
        'test_profitability_rf': training['profitability']['rf']['test'],
    }
    save_json(out_dir / 'summary_metrics.json', summary)

    comparison = pd.DataFrame([
        {
            'version': 'report_cited',
            'long': HISTORIC['long'],
            'short': HISTORIC['short'],
            'no_trade': HISTORIC['no_trade'],
        },
        {
            'version': 'fresh_run',
            'long': int(label_dist.loc[label_dist['label'] == 'long', 'count'].sum()) if 'long' in label_dist['label'].values else 0,
            'short': int(label_dist.loc[label_dist['label'] == 'short', 'count'].sum()) if 'short' in label_dist['label'].values else 0,
            'no_trade': int(label_dist.loc[label_dist['label'] == 'no_trade', 'count'].sum()) if 'no_trade' in label_dist['label'].values else 0,
        },
    ])
    comparison.to_csv(out_dir / 'comparison_vs_report_cited.csv', index=False)

    print(f'Fresh pipeline results saved to: {out_dir}')
    print(label_dist.to_string(index=False))
    print('Historic report distribution:', HISTORIC)
    print('Fresh distribution:', label_dist.set_index('label')['count'].to_dict())
    print('Best F1 score:', best_f1_score)
    print('Best cumulative return score:', best_profit_score)


if __name__ == '__main__':
    main()
