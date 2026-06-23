"""
CPU stage: fuse cached DINOv3 features across views and score with ridge regression.

Reads the caches written by extract_features_dinov3.py, concatenates the per-view
feature matrices (late fusion), fits RidgeCV per fold, and reports test MAE with a
standard error across folds — reproducing the single-view and multi-view tables.

The fused feature matrices are also saved per fold, so the BLR / conformal /
calibration / active-learning steps can load them directly (no GPU, no re-extraction).

Examples:
    # single view
    python modelling/dino/evaluate_dinov3.py --views 4-3-2 --folds 1 2 3 4 5

    # the four-view multi-view model
    python modelling/dino/evaluate_dinov3.py --views 4-3-2 8-4-2 12-1-3 11-8-2 --folds 1 2 3 4 5
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


def feature_dir(root, mode, source, fold, bands):
    """Must match extract_features_dinov3.feature_dir."""
    band_tag = '-'.join(str(b) for b in bands)
    return os.path.join(root, f'{mode}_{source}_fold{fold}_bands{band_tag}')


def load_view(root, mode, source, fold, bands):
    d = feature_dir(root, mode, source, fold, bands)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"Missing feature cache: {d}\n"
            f"Run extract_features_dinov3.py for fold {fold}, bands {bands} first."
        )
    return (np.load(os.path.join(d, 'X_train.npy')),
            np.load(os.path.join(d, 'y_train.npy')),
            np.load(os.path.join(d, 'X_test.npy')),
            np.load(os.path.join(d, 'y_test.npy')))


def fuse_views(root, mode, source, fold, views):
    """Concatenate per-view feature matrices (late fusion). y is shared across views."""
    Xtr_parts, Xte_parts = [], []
    y_train = y_test = None
    for bands in views:
        Xtr, ytr, Xte, yte = load_view(root, mode, source, fold, bands)
        Xtr_parts.append(Xtr)
        Xte_parts.append(Xte)
        if y_train is None:
            y_train, y_test = ytr, yte
        elif not (np.allclose(y_train, ytr) and np.allclose(y_test, yte)):
            raise ValueError(
                f"Target mismatch across views for fold {fold} — were all views extracted "
                f"from the same fold split? ({bands})"
            )
    return np.hstack(Xtr_parts), y_train, np.hstack(Xte_parts), y_test


def ridge_eval(X_train, y_train, X_test, y_test):
    alphas = np.logspace(-6, 6, 20)
    pipeline = Pipeline([('ridge', RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_absolute_error'))])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
    pipeline.fit(X_train, y_train)
    test_mae = float(np.mean(np.abs(pipeline.predict(X_test) - y_test)))
    return test_mae, -cv.mean()


def main(views, folds, mode, source, features_root, save_fused, fused_root):
    print(f"Views: {views}  folds: {folds}  mode: {mode}  source: {source}")
    maes = []
    for fold in folds:
        X_train, y_train, X_test, y_test = fuse_views(features_root, mode, source, fold, views)
        test_mae, cv_mae = ridge_eval(X_train, y_train, X_test, y_test)
        maes.append(test_mae)
        print(f"  fold {fold}: test MAE = {test_mae:.4f}  (train CV MAE = {cv_mae:.4f}, "
              f"dim = {X_train.shape[1]})")

        if save_fused:
            tag = '_'.join('-'.join(str(b) for b in v) for v in views)
            out = os.path.join(fused_root, f'{mode}_{source}_fold{fold}_views_{tag}')
            os.makedirs(out, exist_ok=True)
            np.save(os.path.join(out, 'X_train.npy'), X_train)
            np.save(os.path.join(out, 'y_train.npy'), y_train)
            np.save(os.path.join(out, 'X_test.npy'), X_test)
            np.save(os.path.join(out, 'y_test.npy'), y_test)

    maes = np.array(maes)
    se = maes.std(ddof=0) / np.sqrt(len(maes)) if len(maes) > 1 else 0.0
    print(f"\nMAE over {len(maes)} folds: {maes.mean():.4f} ± {se:.4f} (SE)")
    return maes.mean(), se


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuse cached DINOv3 features and score with ridge.')
    parser.add_argument('--views', type=str, nargs='+', required=True,
                        help='One or more views as dash-joined bands, e.g. 4-3-2 8-4-2 12-1-3 11-8-2')
    parser.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--mode', type=str, default='spatial', choices=['spatial', 'temporal', 'one_country'])
    parser.add_argument('--imagery_source', type=str, default='S')
    parser.add_argument('--features_root', type=str, default='modelling/dino/features')
    parser.add_argument('--save_fused', action='store_true',
                        help='Also save the fused multi-view feature matrices (input for BLR/conformal/etc.)')
    parser.add_argument('--fused_root', type=str, default='modelling/dino/fused_features')
    args = parser.parse_args()

    views = [[int(b) for b in v.split('-')] for v in args.views]
    main(views, args.folds, args.mode, args.imagery_source,
         args.features_root, args.save_fused, args.fused_root)
