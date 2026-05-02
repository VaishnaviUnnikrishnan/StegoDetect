"""
QML Metrics Repair Script
==========================
Run this ONCE if qml_metrics.json is missing from your qml_output folder.
It reconstructs the metrics file from the existing pkl + npy files
without needing to retrain.

Usage:
    python repair_qml_metrics.py --qml-output ./qml_output --ml-output ./ml_output
"""

import json, joblib, argparse, warnings
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.fftpack import dct
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


def extract_features(img_array):
    img = img_array.astype(np.float32)
    feats = []
    H, W, _ = img.shape
    for c in range(3):
        ch = img[:, :, c]; ch_int = ch.astype(np.uint8)
        lsb = ch_int & 1
        feats += [lsb.mean(), lsb.std()]
        hist = np.bincount(ch_int.flatten(), minlength=256)
        feats.append(hist[0::2].sum() / (hist[1::2].sum() + 1e-9))
        counts = np.array([np.sum(lsb==0), np.sum(lsb==1)], dtype=float)
        expected = np.full(2, counts.sum()/2)
        feats.append(float(np.sum((counts-expected)**2/(expected+1e-9))))
        p = hist/(hist.sum()+1e-9)
        feats.append(-np.sum(p*np.log2(p+1e-9)))
        cy, cx = H//2, W//2
        block = ch[max(0,cy-32):cy+32, max(0,cx-32):cx+32]
        if block.shape[0]<2 or block.shape[1]<2: block=ch
        d = dct(dct(block.T,norm='ortho').T,norm='ortho').flatten()
        feats += [float(d.mean()),float(d.std()),float(np.percentile(d,25)),float(np.percentile(d,75))]
        feats += [float(np.abs(np.diff(ch,axis=1)).mean()),float(np.abs(np.diff(ch,axis=0)).mean())]
    r=img[:,:,0].flatten(); g=img[:,:,1].flatten(); b=img[:,:,2].flatten()
    feats += [float(np.corrcoef(r,g)[0,1]),float(np.corrcoef(r,b)[0,1]),float(np.corrcoef(g,b)[0,1])]
    gray=(0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2]).astype(np.uint8)
    for bit in range(4): feats.append(float(((gray>>bit)&1).std()))
    flat=img.flatten()
    feats += [float(flat.mean()),float(flat.std()),float(np.percentile(flat,5)),
              float(np.percentile(flat,95)),float(flat.max()-flat.min()),
              float(np.median(flat)),float(img[:,:,0].mean()-img[:,:,1].mean()),
              float(img[:,:,1].mean()-img[:,:,2].mean())]
    return np.array(feats, dtype=np.float32)


def compute_metrics(y_true, y_pred, split):
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    cm   = confusion_matrix(y_true, y_pred).tolist()
    print(f"  {split:6s} → acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")
    return dict(split=split, accuracy=round(acc,4), precision=round(prec,4),
                recall=round(rec,4), f1_score=round(f1,4),
                confusion_matrix=cm, n_samples=int(len(y_true)))


def qml_predict_batch(payload, weights, X_reduced):
    """Reconstruct VQC and predict on a feature-reduced array."""
    from qiskit.circuit.library import zz_feature_map, real_amplitudes
    from qiskit.primitives import StatevectorSampler
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.optimizers import COBYLA

    n    = payload['n_qubits']
    reps = payload['reps']
    fm   = zz_feature_map(feature_dimension=n, reps=1)
    ans  = real_amplitudes(num_qubits=n, reps=reps)
    vqc  = VQC(sampler=StatevectorSampler(), feature_map=fm,
               ansatz=ans, optimizer=COBYLA(maxiter=1))
    # Dummy fit to initialise internal state
    vqc.fit(X_reduced[:2], np.array([0, 1]))
    vqc._fit_result.x = weights
    return vqc.predict(X_reduced)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qml-output', default='./qml_output')
    parser.add_argument('--ml-output',  default='./ml_output')
    args = parser.parse_args()

    qml_p = Path(args.qml_output)
    ml_p  = Path(args.ml_output)

    print("\n" + "="*52)
    print("  QML Metrics Repair")
    print("="*52)

    # ── Load existing pkl and weights ─────────────────────
    print("\n[1/4] Loading existing model files...")
    payload = joblib.load(qml_p / 'qml_vqc_model.pkl')
    weights = np.load(str(qml_p / 'vqc_weights.npy'))
    print(f"  Payload keys : {list(payload.keys())}")
    print(f"  Weights shape: {weights.shape}")

    scaler = payload['scaler']
    pca    = payload['pca']
    n      = payload['n_qubits']
    reps   = payload['reps']

    # ── Load feature arrays from ml_output ───────────────
    print("\n[2/4] Loading feature arrays from ml_output...")
    X_train = np.load(ml_p / 'X_train.npy')
    y_train = np.load(ml_p / 'y_train.npy')
    X_val   = np.load(ml_p / 'X_val.npy')
    y_val   = np.load(ml_p / 'y_val.npy')
    X_test  = np.load(ml_p / 'X_test.npy')
    y_test  = np.load(ml_p / 'y_test.npy')
    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # ── Apply same PCA/scaler transform ──────────────────
    print("\n[3/4] Applying PCA transform and predicting...")
    X_train_r = pca.transform(scaler.transform(X_train))
    X_val_r   = pca.transform(scaler.transform(X_val))
    X_test_r  = pca.transform(scaler.transform(X_test))

    # Subsample train to match original training set size
    train_size = payload.get('train_samples', 300)
    rng  = np.random.default_rng(42)
    n_cls = min(train_size//2, (y_train==0).sum(), (y_train==1).sum())
    idx0 = rng.choice(np.where(y_train==0)[0], size=n_cls, replace=False)
    idx1 = rng.choice(np.where(y_train==1)[0], size=n_cls, replace=False)
    idx  = np.concatenate([idx0, idx1])
    X_tr = X_train_r[idx]; y_tr = y_train[idx]

    all_metrics = {}
    for split_name, X, y in [('train', X_tr, y_tr),
                               ('val',   X_val_r, y_val),
                               ('test',  X_test_r, y_test)]:
        print(f"  Predicting {split_name} ({len(y)} samples)...")
        pred = qml_predict_batch(payload, weights, X)
        all_metrics[split_name] = compute_metrics(y, pred, split_name)

    # ── Build and save qml_metrics.json ──────────────────
    print("\n[4/4] Writing qml_metrics.json...")

    circuit_info = {
        'n_qubits'             : n,
        'reps'                 : reps,
        'feature_map'          : payload.get('feature_map', 'ZZFeatureMap'),
        'ansatz'               : payload.get('ansatz', 'RealAmplitudes'),
        'optimizer'            : payload.get('optimizer', 'COBYLA'),
        'ansatz_params'        : len(weights),
        'training_samples'     : int(train_size),
        'training_time_s'      : payload.get('training_time_s', None),
        'backend'              : 'AerSimulator (StatevectorSampler)',
        'trained_weights'      : weights.tolist(),
        'pca_variance_retained': float(pca.explained_variance_ratio_.sum()),
        'repaired'             : True,
    }

    metrics_out = {
        'model_type'  : 'VQC',
        'circuit_info': circuit_info,
        'splits'      : all_metrics,
    }

    out_path = qml_p / 'qml_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\n  Saved → {out_path}")
    print(f"\n{'='*52}")
    print(f"  Done! Restart your Streamlit app now.")
    print(f"  Test accuracy: {all_metrics['test']['accuracy']*100:.2f}%")
    print(f"{'='*52}\n")


if __name__ == '__main__':
    main()