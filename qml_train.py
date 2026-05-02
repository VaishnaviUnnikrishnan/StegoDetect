"""
QML Steganalysis Trainer  —  Quantum SVM (PegasosQSVC)
=======================================================
Uses a quantum kernel (ZZFeatureMap fidelity) to classify clean vs stego images.

Usage:
    python qml_train.py --data ./ml_output --output ./qml_output
    python qml_train.py --data ./ml_output --output ./qml_output --ibm-token TOKEN
"""

import os, sys, json, time, joblib, warnings, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_qsphere
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.algorithms import PegasosQSVC

warnings.filterwarnings('ignore')


def load_features(data_dir):
    d = Path(data_dir)
    keys = ['X_train','y_train','X_val','y_val','X_test','y_test']
    for k in keys:
        if not (d / f'{k}.npy').exists():
            sys.exit(f"[ERROR] Missing {d/k}.npy — run classical_ml_train.py first.")
    return {k: np.load(str(d / f'{k}.npy')) for k in keys}


def pca_reduce(X_train, X_val, X_test, n):
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    Xs  = scaler.fit_transform(X_train)
    pca = PCA(n_components=n, random_state=42)
    Xr  = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {X_train.shape[1]} → {n} | variance retained: {var*100:.1f}%")
    return (Xr,
            pca.transform(scaler.transform(X_val)),
            pca.transform(scaler.transform(X_test)),
            scaler, pca)


def subsample_balanced(X, y, n_per_class, seed=42):
    rng  = np.random.default_rng(seed)
    n0   = min(n_per_class, int((y == 0).sum()))
    n1   = min(n_per_class, int((y == 1).sum()))
    idx0 = rng.choice(np.where(y == 0)[0], size=n0, replace=False)
    idx1 = rng.choice(np.where(y == 1)[0], size=n1, replace=False)
    idx  = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    print(f"  Balanced: {n0} clean + {n1} stego = {len(idx)} training samples")
    return X[idx], y[idx]


def predict_batched(qsvc, X, batch_size=25, label=""):
    """Predict in small batches with a progress bar — avoids Windows deadlock."""
    preds = []
    total = len(X)
    for i in range(0, total, batch_size):
        batch = X[i:i + batch_size]
        p = qsvc.predict(batch)
        preds.extend(p.tolist())
        done = min(i + batch_size, total)
        filled = done * 40 // total
        bar = "#" * filled + "-" * (40 - filled)
        print(f"\r    [{bar}] {done}/{total} {label}", end="", flush=True)
    print()
    return np.array(preds)


def compute_metrics(y_true, y_pred, split):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred).tolist()
    print(f"\n  ── {split} ──────────────────────────────")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"  CM: {cm[0]}  ← clean")
    print(f"      {cm[1]}  ← stego")
    print(classification_report(y_true, y_pred, target_names=['clean','stego']))
    return dict(split=split, accuracy=round(float(acc), 4),
                precision=round(float(prec), 4), recall=round(float(rec), 4),
                f1_score=round(float(f1), 4), confusion_matrix=cm,
                n_samples=int(len(y_true)))


def save_circuit_diagram(fm, out_dir):
    print("  Generating circuit diagram...")
    try:
        c = fm.copy()
        c.measure_all()
        fig = c.draw(output='mpl', fold=-1, style='iqp')
        fig.savefig(out_dir / 'circuit_diagram.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        print("  Saved → circuit_diagram.png")
    except Exception as e:
        print(f"  [WARN] Circuit: {e}")


def save_qsphere(fm, sample, out_dir):
    print("  Generating Q-sphere...")
    try:
        bound = fm.assign_parameters(sample)
        sv    = Statevector(bound)
        fig   = plot_state_qsphere(sv, figsize=(6, 6))
        fig.suptitle("Q-sphere — QSVM feature map (single sample)", fontsize=11)
        fig.savefig(out_dir / 'qsphere.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        print("  Saved → qsphere.png")
    except Exception as e:
        print(f"  [WARN] Q-sphere: {e}")


def save_kernel_matrix(kernel, X_sub, out_dir):
    print("  Generating kernel matrix...")
    try:
        n  = min(40, len(X_sub))
        K  = kernel.evaluate(X_sub[:n])
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0e1520')
        ax.set_facecolor('#0e1520')
        im = ax.imshow(K, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_title('Quantum Kernel Matrix', color='white')
        ax.set_xlabel('Sample index', color='#94a3b8')
        ax.set_ylabel('Sample index', color='#94a3b8')
        ax.tick_params(colors='#64748b')
        fig.tight_layout()
        fig.savefig(out_dir / 'measurement_histogram.png', dpi=120,
                    bbox_inches='tight', facecolor='#0e1520')
        plt.close(fig)
        print("  Saved → measurement_histogram.png")
    except Exception as e:
        print(f"  [WARN] Kernel matrix: {e}")


def save_confusion_heatmap(cm_list, split, out_dir):
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor('#0e1520')
    ax.set_facecolor('#0e1520')
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Clean', 'Pred: Stego'], color='#94a3b8')
    ax.set_yticklabels(['True: Clean', 'True: Stego'], color='#94a3b8')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else '#94a3b8',
                    fontsize=14, fontweight='bold')
    ax.set_title(f'Confusion Matrix — {split}', color='white', pad=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f'confusion_matrix_{split.lower()}.png',
                dpi=120, bbox_inches='tight', facecolor='#0e1520')
    plt.close(fig)
    print(f"  Saved → confusion_matrix_{split.lower()}.png")


def save_training_curve(C, num_steps, out_dir):
    print("  Generating training curve...")
    try:
        steps = np.arange(1, num_steps + 1)
        curve = 1.0 / np.sqrt(steps)
        curve = curve / curve[0]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor('#0e1520')
        ax.set_facecolor('#0e1520')
        ax.plot(steps, curve, color='#7c3aed', linewidth=1.8)
        ax.fill_between(steps, curve, alpha=0.15, color='#7c3aed')
        ax.set_xlabel('Pegasos step', color='#94a3b8')
        ax.set_ylabel('Normalised objective', color='#94a3b8')
        ax.set_title('QSVM Pegasos convergence (theoretical)', color='white')
        ax.tick_params(colors='#64748b')
        ax.grid(True, alpha=0.15, color='#1e2d45')
        for sp in ax.spines.values():
            sp.set_edgecolor('#1e2d45')
        fig.tight_layout()
        fig.savefig(out_dir / 'training_curve.png', dpi=120,
                    bbox_inches='tight', facecolor='#0e1520')
        plt.close(fig)
        print("  Saved → training_curve.png")
    except Exception as e:
        print(f"  [WARN] Training curve: {e}")


def get_ibm_sampler(token, backend_name):
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session
        print(f"  Connecting to IBM Quantum ({backend_name})...")
        svc     = QiskitRuntimeService(channel='ibm_quantum', token=token)
        backend = svc.backend(backend_name)
        print(f"  Backend: {backend.name} | qubits: {backend.num_qubits}")
        session = Session(backend=backend)
        return SamplerV2(mode=session), session
    except ImportError:
        sys.exit("[ERROR] pip install qiskit-ibm-runtime")
    except Exception as e:
        sys.exit(f"[ERROR] IBM: {e}")


def main():
    parser = argparse.ArgumentParser(description='QML QSVM Steganalysis')
    parser.add_argument('--data',        required=True,
                        help='ml_output/ folder with .npy feature arrays')
    parser.add_argument('--output',      default='./qml_output')
    parser.add_argument('--n-qubits',    type=int,   default=4,
                        help='PCA components / qubits (default: 4)')
    parser.add_argument('--reps',        type=int,   default=2,
                        help='Feature map repetitions (default: 2)')
    parser.add_argument('--C',           type=float, default=100.0,
                        help='SVM regularisation C (default: 100)')
    parser.add_argument('--num-steps',   type=int,   default=200,
                        help='Pegasos steps (default: 200)')
    parser.add_argument('--train-size',  type=int,   default=50,
                        help='Samples per class for training (default: 50)')
    parser.add_argument('--batch-size',  type=int,   default=25,
                        help='Prediction batch size to avoid freeze (default: 25)')
    parser.add_argument('--ibm-token',   default=None)
    parser.add_argument('--ibm-backend', default='ibm_brisbane')
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  QML Steganalysis  —  Quantum SVM (PegasosQSVC)")
    print("=" * 60)

    # ── 1. Load ────────────────────────────────────────────
    print("\n[1/5] Loading features...")
    d = load_features(args.data)
    X_train, y_train = d['X_train'], d['y_train']
    X_val,   y_val   = d['X_val'],   d['y_val']
    X_test,  y_test  = d['X_test'],  d['y_test']
    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # ── 2. PCA ─────────────────────────────────────────────
    print(f"\n[2/5] PCA → {args.n_qubits} qubits...")
    X_tr_r, X_val_r, X_te_r, scaler, pca = pca_reduce(
        X_train, X_val, X_test, args.n_qubits)
    X_sub, y_sub = subsample_balanced(X_tr_r, y_train, args.train_size)
    joblib.dump(scaler, out_dir / 'pca_scaler.pkl')
    joblib.dump(pca,    out_dir / 'pca_transform.pkl')

    # ── 3. Build kernel ────────────────────────────────────
    print(f"\n[3/5] Building quantum kernel "
          f"({args.n_qubits} qubits, reps={args.reps})...")
    fm = zz_feature_map(feature_dimension=args.n_qubits, reps=args.reps)

    if args.ibm_token:
        sampler, ibm_session = get_ibm_sampler(args.ibm_token, args.ibm_backend)
    else:
        sampler, ibm_session = StatevectorSampler(), None
        print("  [SIM] Local StatevectorSampler")

    fidelity = ComputeUncompute(sampler=sampler)
    kernel   = FidelityQuantumKernel(fidelity=fidelity, feature_map=fm)
    qsvc     = PegasosQSVC(quantum_kernel=kernel, C=args.C,
                            num_steps=args.num_steps)

    print(f"  Circuit depth : {fm.decompose().depth()}")
    print(f"  C={args.C}  steps={args.num_steps}  batch={args.batch_size}")

    # ── 4. Train ───────────────────────────────────────────
    print(f"\n[4/5] Training QSVM on {len(y_sub)} samples...")
    t0 = time.time()
    qsvc.fit(X_sub, y_sub)
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s")
    if ibm_session:
        ibm_session.close()

    # ── 5. Evaluate (batched to avoid Windows freeze) ──────
    print(f"\n[5/5] Evaluating (batch_size={args.batch_size})...")
    all_metrics = {}
    for sname, X, y in [
        ('train', X_sub,   y_sub),
        ('val',   X_val_r, y_val),
        ('test',  X_te_r,  y_test),
    ]:
        print(f"  Predicting {sname} ({len(y)} samples)...")
        pred = predict_batched(qsvc, X, batch_size=args.batch_size,
                               label=sname)
        m = compute_metrics(y, pred, sname.capitalize())
        all_metrics[sname] = m
        save_confusion_heatmap(m['confusion_matrix'], sname.capitalize(), out_dir)

    # ── Save metrics JSON ──────────────────────────────────
    circuit_info = {
        'model'              : 'QSVM (PegasosQSVC)',
        'n_qubits'           : args.n_qubits,
        'reps'               : args.reps,
        'feature_map'        : 'ZZFeatureMap',
        'kernel'             : 'FidelityQuantumKernel',
        'optimizer'          : f'Pegasos (C={args.C}, steps={args.num_steps})',
        'ansatz_params'      : 0,
        'circuit_depth'      : int(fm.decompose().depth()),
        'circuit_gates'      : int(fm.decompose().size()),
        'feature_map_params' : int(fm.num_parameters),
        'training_samples'   : int(len(y_sub)),
        'training_time_s'    : round(train_time, 2),
        'backend'            : (args.ibm_backend if args.ibm_token
                                else 'AerSimulator (StatevectorSampler)'),
        'pca_variance_retained': float(pca.explained_variance_ratio_.sum()),
        'C'                  : args.C,
        'num_steps'          : args.num_steps,
    }
    with open(out_dir / 'qml_metrics.json', 'w') as f:
        json.dump({'model_type': 'QSVM',
                   'circuit_info': circuit_info,
                   'splits': all_metrics}, f, indent=2)
    print(f"  Metrics saved → qml_metrics.json")

    # ── Save model pkl ─────────────────────────────────────
    payload = {
        'model_type'      : 'QSVM',
        'qsvc'            : qsvc,
        'support_vectors' : X_sub,
        'support_labels'  : y_sub,
        'scaler'          : scaler,
        'pca'             : pca,
        'n_qubits'        : args.n_qubits,
        'reps'            : args.reps,
        'C'               : args.C,
        'num_steps'       : args.num_steps,
        'training_time_s' : round(train_time, 2),
        'feature_map'     : 'ZZFeatureMap',
        'optimizer'       : 'Pegasos',
    }
    joblib.dump(payload, out_dir / 'qml_vqc_model.pkl')
    print(f"  Model saved → qml_vqc_model.pkl")

    # Dual coefficients as weights (app.py compatibility)
    try:
        dual = np.array(list(qsvc.dual_coef_))
    except Exception:
        dual = np.array([0.0])
    np.save(out_dir / 'vqc_weights.npy', dual)
    print(f"  Weights saved → vqc_weights.npy")

    # ── Visualisations ─────────────────────────────────────
    save_circuit_diagram(fm, out_dir)
    save_qsphere(fm, X_sub[0], out_dir)
    save_kernel_matrix(kernel, X_sub, out_dir)
    save_training_curve(args.C, args.num_steps, out_dir)

    # ── Summary ────────────────────────────────────────────
    test_acc = all_metrics['test']['accuracy']
    test_f1  = all_metrics['test']['f1_score']
    print(f"\n{'=' * 60}")
    print(f"  FINAL TEST ACCURACY : {test_acc * 100:.2f}%")
    print(f"  FINAL TEST F1       : {test_f1:.4f}")
    print(f"{'=' * 60}")
    print(f"\n  For IBM hardware: add --ibm-token YOUR_KEY\n")


if __name__ == '__main__':
    main()