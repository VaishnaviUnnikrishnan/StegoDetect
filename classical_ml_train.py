"""
Classical ML Steganalysis Trainer
===================================
Trains a Random Forest classifier to detect LSB steganography.

Feature extraction mirrors the pixel-level analysis used during
manual inspection: LSB statistics, even/odd ratios, DCT energy,
histogram features, and inter-channel correlations.

Usage:
    python classical_ml_train.py \
        --train  ./Stego-pvd-dataset/stego_train \
        --val    ./Stego-pvd-dataset/stego_val \
        --test   ./Stego-pvd-dataset/stego_test \
        --output ./ml_output
"""

import os
import sys
import argparse
import warnings
import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.fftpack import dct
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (pixel-level, same as manual analysis)
# ══════════════════════════════════════════════════════════════

def extract_features(img_path: str) -> np.ndarray | None:
    """
    Extract 52 pixel-level statistical features from one image.

    Feature groups:
      1. LSB mean & std per channel          (6 features)
      2. Even/odd pixel value ratio          (3 features)
      3. Chi-square LSB uniformity score     (3 features)
      4. Histogram bin entropy per channel   (3 features)
      5. DCT coefficient statistics          (12 features)
      6. Pixel difference (adjacent)         (6 features)
      7. Cross-channel correlation           (3 features)
      8. Bit-plane noise metrics             (4 features)
      9. Overall image statistics            (8 features)
                                    TOTAL = 48 features
    """
    try:
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
    except Exception as e:
        print(f"  [SKIP] {img_path} — {e}")
        return None

    feats = []
    H, W, _ = img.shape

    for c in range(3):
        ch = img[:, :, c]
        ch_int = ch.astype(np.uint8)

        # ── 1. LSB mean & std ──────────────────────────────
        lsb = ch_int & 1
        feats += [lsb.mean(), lsb.std()]

        # ── 2. Even/odd ratio ──────────────────────────────
        hist = np.bincount(ch_int.flatten(), minlength=256)
        even = hist[0::2].sum()
        odd  = hist[1::2].sum() + 1e-9
        feats.append(even / odd)

        # ── 3. Chi-square LSB uniformity ──────────────────
        counts  = np.array([np.sum(lsb == 0), np.sum(lsb == 1)], dtype=float)
        expected = np.full(2, counts.sum() / 2)
        chi2 = float(np.sum((counts - expected) ** 2 / (expected + 1e-9)))
        feats.append(chi2)

        # ── 4. Histogram entropy ───────────────────────────
        p = hist / (hist.sum() + 1e-9)
        entropy = -np.sum(p * np.log2(p + 1e-9))
        feats.append(entropy)

        # ── 5. DCT statistics ──────────────────────────────
        #   Sample a central 64×64 block for speed
        cy, cx = H // 2, W // 2
        block = ch[max(0,cy-32):cy+32, max(0,cx-32):cx+32]
        if block.shape[0] < 2 or block.shape[1] < 2:
            block = ch
        d = dct(dct(block.T, norm='ortho').T, norm='ortho')
        d_flat = d.flatten()
        feats += [
            float(d_flat.mean()),
            float(d_flat.std()),
            float(np.percentile(d_flat, 25)),
            float(np.percentile(d_flat, 75)),
        ]

        # ── 6. Adjacent pixel differences ─────────────────
        diff_h = np.abs(np.diff(ch, axis=1)).mean()   # horizontal
        diff_v = np.abs(np.diff(ch, axis=0)).mean()   # vertical
        feats += [float(diff_h), float(diff_v)]

    # ── 7. Cross-channel correlation ───────────────────────
    r = img[:, :, 0].flatten()
    g = img[:, :, 1].flatten()
    b = img[:, :, 2].flatten()
    feats.append(float(np.corrcoef(r, g)[0, 1]))
    feats.append(float(np.corrcoef(r, b)[0, 1]))
    feats.append(float(np.corrcoef(g, b)[0, 1]))

    # ── 8. Bit-plane noise (planes 0–3) ────────────────────
    gray = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]).astype(np.uint8)
    for bit in range(4):
        plane = (gray >> bit) & 1
        feats.append(float(plane.std()))

    # ── 9. Global statistics ────────────────────────────────
    flat = img.flatten()
    feats += [
        float(flat.mean()),
        float(flat.std()),
        float(np.percentile(flat, 5)),
        float(np.percentile(flat, 95)),
        float(flat.max() - flat.min()),          # dynamic range
        float(np.median(flat)),
        float(img[:,:,0].mean() - img[:,:,1].mean()),  # R-G channel diff
        float(img[:,:,1].mean() - img[:,:,2].mean()),  # G-B channel diff
    ]

    return np.array(feats, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
#  DATASET LOADER
# ══════════════════════════════════════════════════════════════

def load_split(split_dir: str, split_name: str) -> tuple[np.ndarray, np.ndarray, list]:
    """Load features and labels from a split directory containing labels.csv."""
    split_path = Path(split_dir)
    csv_path   = split_path / 'labels.csv'

    if not csv_path.exists():
        sys.exit(f"[ERROR] labels.csv not found in '{split_dir}'")

    df = pd.read_csv(csv_path)
    print(f"\n  Loading {split_name}: {len(df)} samples "
          f"(clean={len(df[df.label==0])}, stego={len(df[df.label==1])})")

    X, y, paths = [], [], []
    skipped = 0

    for _, row in df.iterrows():
        # Try path from CSV first, then relative to split_dir
        img_path = Path(row['path'])
        if not img_path.exists():
            # Build path from split_dir + label_str folder + filename
            img_path = split_path / row['label_str'] / row['filename']
        if not img_path.exists():
            skipped += 1
            continue

        feats = extract_features(str(img_path))
        if feats is None:
            skipped += 1
            continue

        X.append(feats)
        y.append(int(row['label']))
        paths.append(str(img_path))

    if skipped:
        print(f"  [WARN] Skipped {skipped} files (not found or unreadable)")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), paths


# ══════════════════════════════════════════════════════════════
#  METRICS HELPER
# ══════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, split_name: str) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred).tolist()

    print(f"\n  ── {split_name} Results ──────────────────────")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Confusion matrix (TN FP / FN TP):")
    print(f"    {cm[0]}  ← clean")
    print(f"    {cm[1]}  ← stego")
    print(f"\n{classification_report(y_true, y_pred, target_names=['clean','stego'])}")

    return {
        'split'    : split_name,
        'accuracy' : round(float(acc),  4),
        'precision': round(float(prec), 4),
        'recall'   : round(float(rec),  4),
        'f1_score' : round(float(f1),   4),
        'confusion_matrix': cm,
        'n_samples': int(len(y_true)),
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Classical ML Steganalysis Trainer')
    parser.add_argument('--train',  required=True, help='Path to stego_train directory')
    parser.add_argument('--val',    required=True, help='Path to stego_val directory')
    parser.add_argument('--test',   required=True, help='Path to stego_test directory')
    parser.add_argument('--output', default='./ml_output', help='Output directory for model + metrics')
    parser.add_argument('--model',  default='rf',
                        choices=['rf', 'gb', 'svm'],
                        help='Classifier: rf=RandomForest, gb=GradientBoosting, svm=SVM (default: rf)')
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*55)
    print("  Classical ML Steganalysis Trainer")
    print("="*55)

    # ── 1. Load all splits ─────────────────────────────────
    print("\n[1/4] Extracting features from all splits...")
    t0 = time.time()

    X_train, y_train, _ = load_split(args.train, 'Train')
    X_val,   y_val,   _ = load_split(args.val,   'Val')
    X_test,  y_test,  _ = load_split(args.test,  'Test')

    print(f"\n  Feature extraction done in {time.time()-t0:.1f}s")
    print(f"  Feature vector size : {X_train.shape[1]} features per image")
    print(f"  Train shape : {X_train.shape}")
    print(f"  Val   shape : {X_val.shape}")
    print(f"  Test  shape : {X_test.shape}")

    # Save feature matrices for QML reuse
    np.save(out_dir / 'X_train.npy', X_train)
    np.save(out_dir / 'y_train.npy', y_train)
    np.save(out_dir / 'X_val.npy',   X_val)
    np.save(out_dir / 'y_val.npy',   y_val)
    np.save(out_dir / 'X_test.npy',  X_test)
    np.save(out_dir / 'y_test.npy',  y_test)
    print(f"\n  Feature arrays saved to '{out_dir}/' (reusable for QML)")

    # ── 2. Build model pipeline ────────────────────────────
    print(f"\n[2/4] Building {args.model.upper()} pipeline...")

    if args.model == 'rf':
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        # RF doesn't need scaling but pipeline keeps interface consistent
        pipeline = Pipeline([('clf', clf)])

    elif args.model == 'gb':
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf)])

    else:  # svm
        clf = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf)])

    # ── 3. Train ───────────────────────────────────────────
    print(f"\n[3/4] Training...")
    t1 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"  Training complete in {time.time()-t1:.1f}s")

    # ── 4. Evaluate on all splits ──────────────────────────
    print(f"\n[4/4] Evaluating...")
    all_metrics = {}

    for split_name, X, y in [
        ('train', X_train, y_train),
        ('val',   X_val,   y_val),
        ('test',  X_test,  y_test),
    ]:
        y_pred = pipeline.predict(X)
        all_metrics[split_name] = compute_metrics(y, y_pred, split_name.capitalize())

    # ── 5. Save model ──────────────────────────────────────
    model_path = out_dir / f'stego_classifier_{args.model}.pkl'
    joblib.dump(pipeline, model_path)
    print(f"\n  Model saved → {model_path}")

    # ── 6. Save metrics JSON ───────────────────────────────
    metrics_out = {
        'model_type'      : args.model,
        'feature_count'   : int(X_train.shape[1]),
        'train_samples'   : int(len(y_train)),
        'val_samples'     : int(len(y_val)),
        'test_samples'    : int(len(y_test)),
        'training_time_s' : round(time.time() - t1, 2),
        'splits'          : all_metrics,
    }

    metrics_path = out_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    # ── 7. Feature importance (RF / GB only) ──────────────
    if args.model in ('rf', 'gb'):
        importances = pipeline.named_steps['clf'].feature_importances_
        feat_names = (
            [f'{ch}_{name}' for ch in ['R','G','B']
             for name in ['lsb_mean','lsb_std','even_odd','chi2','entropy',
                          'dct_mean','dct_std','dct_p25','dct_p75',
                          'diff_h','diff_v']] +
            ['corr_RG','corr_RB','corr_GB'] +
            [f'bitplane_{i}_std' for i in range(4)] +
            ['global_mean','global_std','pct5','pct95',
             'dynamic_range','median','RG_diff','GB_diff']
        )
        fi_df = pd.DataFrame({
            'feature'   : feat_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)

        fi_path = out_dir / 'feature_importance.csv'
        fi_df.to_csv(fi_path, index=False)
        print(f"  Feature importances → {fi_path}")
        print(f"\n  Top 10 most important features:")
        print(fi_df.head(10).to_string(index=False))

    # ── Final summary ──────────────────────────────────────
    test_acc = all_metrics['test']['accuracy']
    test_f1  = all_metrics['test']['f1_score']
    print(f"\n{'='*55}")
    print(f"  FINAL TEST ACCURACY : {test_acc*100:.2f}%")
    print(f"  FINAL TEST F1 SCORE : {test_f1:.4f}")
    print(f"{'='*55}\n")

    if test_acc >= 0.85:
        print("  Target accuracy (85-90%) ACHIEVED.")
    else:
        print("  Tip: Try --model gb (GradientBoosting) for higher accuracy.")

    print(f"\n  Output files in '{out_dir}/':")
    for f in sorted(out_dir.iterdir()):
        print(f"    {f.name}")


if __name__ == '__main__':
    main()