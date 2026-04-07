"""
Trains a Random Forest classifier on 55D REP profiles from the IAUG experiment.

Input:  rep_profiles_train_iaug.pt
Output: rep_meta_classifier_iaug.pkl

Usage:
    python train_meta_classifier_iaug.py

Environment variables:
    OUT_DIR — where .pt files are and where to save classifier (default: ./)
"""

import os
import sys
import torch
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = os.environ.get("OUT_DIR", str(BASE_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v2_cropped_optimized"))
from cropped_data_generator import BASE_CLASSES, CLASS_TO_IDX

IDX_TO_CLASS = {v: BASE_CLASSES[k] for k, v in CLASS_TO_IDX.items()}


def load_profiles(filename, out_dir):
    path = os.path.join(out_dir, filename)
    data = torch.load(path, map_location='cpu')
    X = data['profiles'].numpy()
    y = data['labels'].numpy()
    global_feats = data.get('global_features', 1)
    return X, y, global_feats


def build_feature_names(n_total, global_feats):
    names = []
    n_rec = n_total - global_feats
    n_per_spec = n_rec // 5

    for cls_name in sorted(BASE_CLASSES.values()):
        for i in range(n_per_spec):
            patch_i = i // 2
            stat    = 'mean' if i % 2 == 0 else 'std'
            names.append(f"{cls_name}_p{patch_i}_{stat}")

    if global_feats >= 1: names.append('cy_norm')
    if global_feats >= 2: names.append('aspect_ratio')
    if global_feats >= 3: names.append('bbox_area_norm')
    if global_feats >= 4: names.append('bbox_w_norm')
    if global_feats >= 5: names.append('bbox_h_norm')

    return names


def main():
    print(f"Loading profiles from: {OUT_DIR}")

    X_train, y_train, global_feats = load_profiles('rep_profiles_train_iaug.pt', OUT_DIR)
    X_val,   y_val,   _            = load_profiles('rep_profiles_val_iaug.pt',   OUT_DIR)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Global features: {global_feats}")

    feature_names = build_feature_names(X_train.shape[1], global_feats)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    print("Training Random Forest (IAUG)...")
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\nValidation accuracy: {acc:.4f} ({acc*100:.2f}%)")

    class_names = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))

    # Feature importance
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:15]
    print("\nTop 15 features:")
    for i in top_indices:
        name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        print(f"  {name:30s}  {importances[i]:.4f}")

    clf_path = os.path.join(OUT_DIR, 'rep_meta_classifier_iaug.pkl')
    with open(clf_path, 'wb') as f:
        pickle.dump({
            'classifier': rf,
            'feature_names': feature_names,
            'global_features': global_feats,
            'val_accuracy': acc,
        }, f)
    print(f"\nClassifier saved: {clf_path}")

    # Feature importance plot
    try:
        import matplotlib.pyplot as plt
        n_show = min(20, len(importances))
        top_n  = np.argsort(importances)[::-1][:n_show]
        names  = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in top_n]
        plt.figure(figsize=(12, 6))
        plt.barh(range(n_show), importances[top_n][::-1])
        plt.yticks(range(n_show), names[::-1])
        plt.xlabel('Feature Importance')
        plt.title('IAUG REP Profile — Top Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'IAUG_REP_Feature_Importance.png'))
        print("Feature importance plot saved.")
    except ImportError:
        pass


if __name__ == '__main__':
    main()
