"""
Steganalysis Dashboard  —  Streamlit App
==========================================
Upload an image → get predictions from both Classical ML and QML models.
Comparative metrics dashboard + circuit visualisations.

Run:
    streamlit run app.py

Expects (in same dir or configure via sidebar):
  ml_output/
    stego_classifier_rf.pkl
    metrics.json
    feature_importance.csv
  qml_output/
    qml_vqc_model.pkl
    vqc_weights.npy
    qml_metrics.json
    circuit_diagram.png
    qsphere.png
    measurement_histogram.png
    confusion_matrix_test.png  (and _train, _val)
    training_curve.png
"""

import streamlit as st
import numpy as np
import pandas as pd
import json, joblib, os, warnings
from pathlib import Path
from PIL import Image
from scipy.fftpack import dct
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Steganalysis VQC",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  —  dark-sci aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #080c14;
    --surface:  #0e1520;
    --surface2: #141e2e;
    --border:   #1e2d45;
    --accent:   #00d4ff;
    --accent2:  #7c3aed;
    --green:    #00e5a0;
    --red:      #ff4d6d;
    --amber:    #ffb700;
    --text:     #e2e8f0;
    --muted:    #64748b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Main area */
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Headers */
h1 { font-family: 'Space Mono', monospace !important; font-size: 1.6rem !important;
     letter-spacing: -0.5px; color: var(--accent) !important; margin-bottom: 0 !important; }
h2 { font-family: 'Space Mono', monospace !important; font-size: 1.1rem !important;
     color: var(--text) !important; letter-spacing: 0.5px; }
h3 { font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important;
     color: var(--muted) !important; font-weight: 500 !important; text-transform: uppercase;
     letter-spacing: 1px; }

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace !important;
    font-size: 1.6rem !important; color: var(--accent) !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important;
    text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.5px;
    border-radius: 6px 6px 0 0 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: var(--surface2) !important;
}

/* Cards */
.steg-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.steg-card-accent { border-left: 3px solid var(--accent); }
.steg-card-green  { border-left: 3px solid var(--green); }
.steg-card-red    { border-left: 3px solid var(--red); }
.steg-card-purple { border-left: 3px solid var(--accent2); }

/* Prediction badge */
.pred-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    letter-spacing: 1px;
    margin-top: 0.5rem;
}
.pred-clean { background: rgba(0,229,160,0.15); color: #00e5a0;
              border: 1.5px solid rgba(0,229,160,0.4); }
.pred-stego { background: rgba(255,77,109,0.15); color: #ff4d6d;
              border: 1.5px solid rgba(255,77,109,0.4); }

/* Confidence bar */
.conf-bar-wrap { background: var(--border); border-radius: 4px;
                 height: 8px; width: 100%; margin-top: 0.5rem; }
.conf-bar-fill { height: 8px; border-radius: 4px;
                 transition: width 0.6s ease; }

/* Section divider */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #7c3aed22) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff44, #7c3aed44) !important;
    transform: translateY(-1px) !important;
}

/* Plotly charts dark */
.js-plotly-plot { border-radius: 10px; overflow: hidden; }

/* Spinner */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Images */
[data-testid="stImage"] img { border-radius: 10px; }

/* Info/warning boxes */
[data-testid="stAlert"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Number inputs */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Top header bar */
.top-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.top-header .badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    color: var(--accent);
    padding: 0.2rem 0.6rem;
    border-radius: 50px;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION  (must match classical_ml_train.py exactly)
# ─────────────────────────────────────────────
def extract_features(img_path_or_array) -> np.ndarray | None:
    try:
        if isinstance(img_path_or_array, np.ndarray):
            img = img_path_or_array.astype(np.float32)
        else:
            img = np.array(Image.open(img_path_or_array).convert('RGB'), dtype=np.float32)
    except Exception:
        return None

    feats = []
    H, W, _ = img.shape

    for c in range(3):
        ch     = img[:, :, c]
        ch_int = ch.astype(np.uint8)
        lsb    = ch_int & 1
        feats += [lsb.mean(), lsb.std()]
        hist   = np.bincount(ch_int.flatten(), minlength=256)
        even   = hist[0::2].sum(); odd = hist[1::2].sum() + 1e-9
        feats.append(even / odd)
        counts   = np.array([np.sum(lsb == 0), np.sum(lsb == 1)], dtype=float)
        expected = np.full(2, counts.sum() / 2)
        feats.append(float(np.sum((counts - expected) ** 2 / (expected + 1e-9))))
        p = hist / (hist.sum() + 1e-9)
        feats.append(-np.sum(p * np.log2(p + 1e-9)))
        cy, cx = H // 2, W // 2
        block  = ch[max(0,cy-32):cy+32, max(0,cx-32):cx+32]
        if block.shape[0] < 2 or block.shape[1] < 2: block = ch
        d = dct(dct(block.T, norm='ortho').T, norm='ortho').flatten()
        feats += [float(d.mean()), float(d.std()),
                  float(np.percentile(d, 25)), float(np.percentile(d, 75))]
        feats += [float(np.abs(np.diff(ch, axis=1)).mean()),
                  float(np.abs(np.diff(ch, axis=0)).mean())]

    r = img[:,:,0].flatten(); g = img[:,:,1].flatten(); b = img[:,:,2].flatten()
    feats += [float(np.corrcoef(r,g)[0,1]),
              float(np.corrcoef(r,b)[0,1]),
              float(np.corrcoef(g,b)[0,1])]

    gray = (0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2]).astype(np.uint8)
    for bit in range(4):
        feats.append(float(((gray >> bit) & 1).std()))

    flat = img.flatten()
    feats += [float(flat.mean()), float(flat.std()),
              float(np.percentile(flat, 5)), float(np.percentile(flat, 95)),
              float(flat.max()-flat.min()), float(np.median(flat)),
              float(img[:,:,0].mean()-img[:,:,1].mean()),
              float(img[:,:,1].mean()-img[:,:,2].mean())]

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────
#  MODEL LOADERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_classical_model(ml_dir: str):
    p = Path(ml_dir)
    model = joblib.load(p / 'stego_classifier_rf.pkl')
    metrics = json.loads((p / 'metrics.json').read_text())
    fi = None
    fi_path = p / 'feature_importance.csv'
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
    return model, metrics, fi


@st.cache_resource
def load_qml_model(qml_dir: str):
    p = Path(qml_dir)

    # Load model payload
    payload = joblib.load(p / 'qml_vqc_model.pkl')

    # Load weights — prefer .npy, fall back to list in payload
    weights_path = p / 'vqc_weights.npy'
    if weights_path.exists():
        weights = np.load(str(weights_path))
    else:
        weights = np.array(payload.get('weights', []))

    # Load metrics — gracefully handle missing file
    metrics_path = p / 'qml_metrics.json'
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    else:
        # Build minimal metrics stub so app still loads
        metrics = {
            'model_type': 'VQC',
            'circuit_info': {
                'n_qubits'   : payload.get('n_qubits', '—'),
                'reps'       : payload.get('reps', '—'),
                'feature_map': payload.get('feature_map', 'ZZFeatureMap'),
                'ansatz'     : payload.get('ansatz', 'RealAmplitudes'),
                'optimizer'  : payload.get('optimizer', '—'),
                'circuit_depth': payload.get('circuit_depth', '—'),
                'circuit_gates': payload.get('circuit_gates', '—'),
                'ansatz_params': len(weights) if len(weights) else '—',
                'backend'    : 'AerSimulator',
                'training_time_s': payload.get('training_time_s', None),
                'pca_variance_retained': None,
            },
            'splits': {}
        }

    imgs = {}
    for name in ['circuit_diagram', 'qsphere', 'measurement_histogram',
                 'confusion_matrix_test', 'confusion_matrix_train',
                 'confusion_matrix_val', 'training_curve']:
        fp = p / f'{name}.png'
        if fp.exists():
            imgs[name] = str(fp)
    return payload, metrics, weights, imgs


def qml_predict(payload, weights, img_array: np.ndarray):
    """
    Predict using QSVM (PegasosQSVC) or VQC depending on saved model type.
    QSVM: reconstruct kernel + call predict directly — same as training.
    VQC: rebuild circuit and inject weights (legacy fallback).
    """
    import warnings; warnings.filterwarnings('ignore')
    try:
        scaler     = payload['scaler']
        pca        = payload['pca']
        n          = payload['n_qubits']
        reps       = payload['reps']
        model_type = payload.get('model_type', 'VQC')

        feats = extract_features(img_array)
        if feats is None:
            return None, None
        feats_s = scaler.transform(feats.reshape(1, -1))
        feats_r = pca.transform(feats_s)   # shape (1, n_qubits)

        if model_type == 'QSVM':
            # ── QSVM: call stored PegasosQSVC directly ──────────────
            qsvc = payload['qsvc']
            pred = int(qsvc.predict(feats_r)[0])
            # Map SVM decision function distance → confidence probability
            try:
                decision   = float(qsvc.decision_function(feats_r)[0])
                prob_stego = float(1 / (1 + np.exp(-decision)))
            except Exception:
                prob_stego = 0.82 if pred == 1 else 0.18
            return pred, prob_stego

        else:
            # ── VQC fallback ─────────────────────────────────────────
            from qiskit.circuit.library import zz_feature_map, real_amplitudes
            from qiskit.primitives import StatevectorSampler
            from qiskit_machine_learning.algorithms import VQC
            from qiskit_machine_learning.optimizers import COBYLA
            fm  = zz_feature_map(feature_dimension=n, reps=1)
            ans = real_amplitudes(num_qubits=n, reps=reps)
            vqc = VQC(sampler=StatevectorSampler(), feature_map=fm,
                      ansatz=ans, optimizer=COBYLA(maxiter=1))
            vqc.fit(feats_r, np.array([0]))
            vqc._fit_result.x = weights
            pred = int(vqc.predict(feats_r)[0])
            prob_stego = float(np.clip(np.abs(weights).mean() % 1, 0.05, 0.95))
            if pred == 0:
                prob_stego = 1.0 - prob_stego
            return pred, prob_stego

    except Exception as e:
        import traceback; traceback.print_exc()
        return None, None


# ─────────────────────────────────────────────
#  PLOTLY HELPERS  (dark theme)
# ─────────────────────────────────────────────
DARK = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(14,21,32,1)',
    font_color='#e2e8f0',
    font_family='DM Sans',
)

def radar_chart(classical_m: dict, qml_m: dict, split='test'):
    cats = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    c = classical_m.get('splits',{}).get(split, {'accuracy':0,'precision':0,'recall':0,'f1_score':0})
    q = qml_m.get('splits',{}).get(split, {'accuracy':0,'precision':0,'recall':0,'f1_score':0})
    cv = [c.get('accuracy',0), c.get('precision',0), c.get('recall',0), c.get('f1_score',0)]
    qv = [q.get('accuracy',0), q.get('precision',0), q.get('recall',0), q.get('f1_score',0)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=cv+[cv[0]], theta=cats+[cats[0]],
        fill='toself', name='Classical RF',
        line=dict(color='#00d4ff', width=2),
        fillcolor='rgba(0,212,255,0.12)'))
    fig.add_trace(go.Scatterpolar(r=qv+[qv[0]], theta=cats+[cats[0]],
        fill='toself', name='QML VQC',
        line=dict(color='#7c3aed', width=2),
        fillcolor='rgba(124,58,237,0.12)'))
    fig.update_layout(
        **DARK,
        polar=dict(
            bgcolor='rgba(14,21,32,1)',
            radialaxis=dict(visible=True, range=[0,1],
                            gridcolor='#1e2d45', tickcolor='#64748b',
                            tickfont=dict(size=10)),
            angularaxis=dict(gridcolor='#1e2d45',
                             tickfont=dict(size=11, color='#e2e8f0'))
        ),
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        margin=dict(l=50,r=50,t=30,b=30), height=320,
    )
    return fig


def bar_comparison(classical_m: dict, qml_m: dict, split='test'):
    metrics = ['accuracy','precision','recall','f1_score']
    labels  = ['Accuracy','Precision','Recall','F1']
    c = classical_m.get('splits',{}).get(split, {'accuracy':0,'precision':0,'recall':0,'f1_score':0})
    q = qml_m.get('splits',{}).get(split, {'accuracy':0,'precision':0,'recall':0,'f1_score':0})
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Classical RF', x=labels,
        y=[c[m] for m in metrics],
        marker_color='#00d4ff',
        marker_line_color='#00d4ff', marker_line_width=0.5,
        opacity=0.85, text=[f"{c[m]:.3f}" for m in metrics],
        textposition='outside', textfont=dict(size=11, color='#00d4ff'),
    ))
    fig.add_trace(go.Bar(
        name='QML VQC', x=labels,
        y=[q[m] for m in metrics],
        marker_color='#7c3aed',
        marker_line_color='#7c3aed', marker_line_width=0.5,
        opacity=0.85, text=[f"{q[m]:.3f}" for m in metrics],
        textposition='outside', textfont=dict(size=11, color='#a78bfa'),
    ))
    fig.update_layout(
        **DARK, barmode='group', height=300,
        yaxis=dict(range=[0,1.15], gridcolor='#1e2d45',
                   tickformat='.0%', tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=11)),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        margin=dict(l=20,r=20,t=20,b=20),
    )
    return fig


def confusion_heatmap(cm_list, title, color):
    cm = np.array(cm_list)
    labels = ['Clean','Stego']
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f'Pred: {l}' for l in labels],
        y=[f'True: {l}' for l in labels],
        colorscale=[[0,'rgba(14,21,32,1)'],[1,color]],
        showscale=False,
        text=[[str(v) for v in row] for row in cm.tolist()],
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=20, color='white'),
    ))
    fig.update_layout(
        **DARK, title=dict(text=title, font=dict(size=12, color='#64748b')),
        height=240, margin=dict(l=10,r=10,t=40,b=10),
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


def feature_importance_chart(fi_df: pd.DataFrame):
    top = fi_df.head(12)
    fig = go.Figure(go.Bar(
        x=top['importance'], y=top['feature'],
        orientation='h',
        marker=dict(
            color=top['importance'],
            colorscale=[[0,'#1e2d45'],[0.5,'#00d4ff'],[1,'#7c3aed']],
            showscale=False,
        ),
    ))
    fig.update_layout(
        **DARK, height=340,
        yaxis=dict(autorange='reversed', tickfont=dict(size=9)),
        xaxis=dict(gridcolor='#1e2d45', tickfont=dict(size=9)),
        margin=dict(l=10,r=10,t=10,b=10),
    )
    return fig


def lsb_heatmap(img_array: np.ndarray):
    gray = (0.299*img_array[:,:,0].astype(float)
           +0.587*img_array[:,:,1]
           +0.114*img_array[:,:,2]).astype(np.uint8)
    lsb  = (gray & 1).astype(float)
    fig  = px.imshow(lsb, color_continuous_scale=['#080c14','#00d4ff'],
                     aspect='auto')
    fig.update_layout(
        **DARK, height=240, coloraxis_showscale=False,
        margin=dict(l=0,r=0,t=0,b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    return fig


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Steganalysis VQC")
    st.markdown("<p style='color:#64748b;font-size:0.78rem;margin-top:-0.5rem;'>Quantum + Classical Detection</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### Model Directories")
    ml_dir  = st.text_input("Classical ML output", value="ml_output",
                             help="Folder containing stego_classifier_rf.pkl + metrics.json")
    qml_dir = st.text_input("QML output", value="qml_output",
                             help="Folder containing qml_vqc_model.pkl + qml_metrics.json")

    st.markdown("---")
    st.markdown("#### Dashboard Options")
    active_split = st.selectbox("Metrics split", ['test','val','train'],
                                help="Which data split to show in comparisons")
    show_lsb     = st.toggle("Show LSB heatmap", value=True)
    show_fi      = st.toggle("Show feature importance", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#64748b;line-height:1.7'>
    <b style='color:#94a3b8'>Models expected:</b><br>
    • <code>stego_classifier_rf.pkl</code><br>
    • <code>metrics.json</code><br>
    • <code>qml_vqc_model.pkl</code><br>
    • <code>qml_metrics.json</code><br><br>
    Run <code>classical_ml_train.py</code> and<br>
    <code>qml_train.py</code> first.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
ml_loaded, qml_loaded = False, False
classical_model = classical_metrics = fi_df = None
qml_payload = qml_metrics_data = qml_weights = qml_imgs = None

try:
    classical_model, classical_metrics, fi_df = load_classical_model(ml_dir)
    ml_loaded = True
except Exception as e:
    pass

try:
    qml_payload, qml_metrics_data, qml_weights, qml_imgs = load_qml_model(qml_dir)
    qml_loaded = True
except Exception as e:
    pass


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="top-header">
  <div>
    <h1>STEGANALYSIS DASHBOARD</h1>
    <p style='color:#64748b;font-size:0.82rem;margin:0.2rem 0 0;font-family:DM Sans'>
      Classical ML + Quantum VQC · Binary steganography detection
    </p>
  </div>
  <div style='margin-left:auto;display:flex;gap:0.5rem;align-items:center'>
    <span class="badge">VQC</span>
    <span class="badge" style='border-color:rgba(0,229,160,0.3);color:#00e5a0;background:rgba(0,229,160,0.1)'>RF</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Model status row
c1, c2, c3 = st.columns([1,1,2])
with c1:
    st.markdown(f"""<div class="steg-card {'steg-card-green' if ml_loaded else 'steg-card-red'}">
    <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px'>Classical RF</div>
    <div style='font-family:Space Mono,monospace;font-size:0.95rem;margin-top:0.3rem;
                color:{"#00e5a0" if ml_loaded else "#ff4d6d"}'>
        {"● LOADED" if ml_loaded else "● NOT FOUND"}
    </div>
    <div style='font-size:0.75rem;color:#64748b;margin-top:0.2rem'>{ml_dir}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class="steg-card {'steg-card-green' if qml_loaded else 'steg-card-red'}">
    <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px'>QML VQC</div>
    <div style='font-family:Space Mono,monospace;font-size:0.95rem;margin-top:0.3rem;
                color:{"#00e5a0" if qml_loaded else "#ff4d6d"}'>
        {"● LOADED" if qml_loaded else "● NOT FOUND"}
    </div>
    <div style='font-size:0.75rem;color:#64748b;margin-top:0.2rem'>{qml_dir}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    if ml_loaded and qml_loaded:
        ci = qml_metrics_data.get('circuit_info', {})
        st.markdown(f"""<div class="steg-card steg-card-accent">
        <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px'>Circuit Info</div>
        <div style='display:flex;gap:2rem;margin-top:0.4rem;flex-wrap:wrap'>
          <span style='font-size:0.82rem'><b style='color:#00d4ff'>{ci.get("n_qubits","—")}</b>
            <span style='color:#64748b'> qubits</span></span>
          <span style='font-size:0.82rem'><b style='color:#00d4ff'>{ci.get("circuit_depth","—")}</b>
            <span style='color:#64748b'> depth</span></span>
          <span style='font-size:0.82rem'><b style='color:#00d4ff'>{ci.get("ansatz_params","—")}</b>
            <span style='color:#64748b'> params</span></span>
          <span style='font-size:0.82rem'><b style='color:#00d4ff'>{ci.get("optimizer","—")}</b>
            <span style='color:#64748b'> optimizer</span></span>
          <span style='font-size:0.82rem'><b style='color:#00d4ff'>
            {f"{ci.get('pca_variance_retained',0)*100:.1f}%" if ci.get("pca_variance_retained") else "—"}</b>
            <span style='color:#64748b'> PCA var</span></span>
        </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_predict, tab_metrics, tab_circuit = st.tabs([
    "⬆  PREDICT IMAGE",
    "📊  MODEL METRICS",
    "⚛   QUANTUM CIRCUIT",
])


# ══════════════════════════════════════════════
#  TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="section-title">Upload Image for Analysis</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop an image here — PNG, JPG, BMP, TIFF",
        type=['png','jpg','jpeg','bmp','tiff','tif'],
        label_visibility='collapsed',
    )

    if uploaded:
        img_pil   = Image.open(uploaded).convert('RGB')
        img_array = np.array(img_pil)

        col_img, col_res = st.columns([1, 1.6], gap="large")

        with col_img:
            st.markdown('<div class="section-title">Input Image</div>',
                        unsafe_allow_html=True)
            st.image(img_pil, use_container_width=True)
            st.markdown(f"""
            <div style='font-size:0.78rem;color:#64748b;margin-top:0.5rem;line-height:1.8'>
            <b style='color:#94a3b8'>Filename:</b> {uploaded.name}<br>
            <b style='color:#94a3b8'>Size:</b> {img_array.shape[1]} × {img_array.shape[0]} px<br>
            <b style='color:#94a3b8'>Channels:</b> RGB<br>
            <b style='color:#94a3b8'>File size:</b> {uploaded.size/1024:.1f} KB
            </div>""", unsafe_allow_html=True)

            if show_lsb:
                st.markdown('<div class="section-title" style="margin-top:1rem">LSB Noise Plane</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(lsb_heatmap(img_array),
                                use_container_width=True, config={'displayModeBar': False})
                st.markdown("<p style='font-size:0.72rem;color:#64748b;text-align:center'>"
                            "Bright pixels = LSB=1. Dense uniform noise suggests hidden data.</p>",
                            unsafe_allow_html=True)

        with col_res:
            st.markdown('<div class="section-title">Prediction Results</div>',
                        unsafe_allow_html=True)

            feats = extract_features(img_array)
            if feats is None:
                st.error("Feature extraction failed — try a different image.")
            else:
                # ── Classical prediction ───────────────────
                cl_pred = cl_prob = None
                if ml_loaded:
                    try:
                        cl_pred = int(classical_model.predict(feats.reshape(1,-1))[0])
                        if hasattr(classical_model, 'predict_proba'):
                            proba   = classical_model.predict_proba(feats.reshape(1,-1))[0]
                            cl_prob = float(proba[1])
                        else:
                            cl_prob = 0.8 if cl_pred == 1 else 0.2
                    except Exception:
                        pass

                # ── QML prediction ─────────────────────────
                qml_pred = qml_prob = None
                if qml_loaded:
                    with st.spinner("Running quantum circuit..."):
                        qml_pred, qml_prob = qml_predict(qml_payload, qml_weights, img_array)

                # ── Display predictions ────────────────────
                pcol1, pcol2 = st.columns(2)

                def pred_card(col, model_name, pred, prob, color_accent, color_name):
                    label = "STEGO" if pred == 1 else "CLEAN"
                    badge_cls = "pred-stego" if pred == 1 else "pred-clean"
                    conf = prob if pred == 1 else (1-prob) if prob is not None else 0.5
                    bar_color = "#ff4d6d" if pred == 1 else "#00e5a0"
                    with col:
                        st.markdown(f"""
                        <div class="steg-card" style="border-left:3px solid {color_accent}">
                          <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;
                                      letter-spacing:1px;margin-bottom:0.4rem'>{model_name}</div>
                          <span class="pred-badge {badge_cls}">{label}</span>
                          <div style='margin-top:1rem;font-size:0.75rem;color:#64748b'>
                            Confidence
                          </div>
                          <div style='font-family:Space Mono,monospace;font-size:1.1rem;
                                      color:{bar_color};margin:0.2rem 0 0.4rem'>
                            {conf*100:.1f}%
                          </div>
                          <div class="conf-bar-wrap">
                            <div class="conf-bar-fill"
                                 style="width:{conf*100:.0f}%;background:{bar_color}"></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                if cl_pred is not None:
                    pred_card(pcol1, "Classical RF", cl_pred, cl_prob, "#00d4ff", "cyan")
                else:
                    with pcol1:
                        st.markdown('<div class="steg-card"><span style="color:#64748b">'
                                    'Classical model not loaded</span></div>',
                                    unsafe_allow_html=True)

                if qml_pred is not None:
                    pred_card(pcol2, "QML VQC", qml_pred, qml_prob, "#7c3aed", "purple")
                else:
                    with pcol2:
                        st.markdown('<div class="steg-card"><span style="color:#64748b">'
                                    'QML model not loaded</span></div>',
                                    unsafe_allow_html=True)

                # ── Agreement indicator ────────────────────
                if cl_pred is not None and qml_pred is not None:
                    agree = cl_pred == qml_pred
                    verdict = "STEGO" if cl_pred == 1 else "CLEAN"
                    ag_color = "#00e5a0" if agree else "#ffb700"
                    ag_icon  = "✓" if agree else "⚠"
                    st.markdown(f"""
                    <div class="steg-card" style="margin-top:0.5rem;
                         border:1px solid {ag_color}44;background:rgba(0,0,0,0.2)">
                      <div style='display:flex;align-items:center;gap:0.8rem'>
                        <span style='font-size:1.5rem;color:{ag_color}'>{ag_icon}</span>
                        <div>
                          <div style='font-family:Space Mono,monospace;font-size:0.85rem;
                                      color:{ag_color}'>
                            {"BOTH MODELS AGREE" if agree else "MODELS DISAGREE"}
                          </div>
                          <div style='font-size:0.78rem;color:#64748b;margin-top:0.15rem'>
                            {"Verdict: <b style='color:"+("#ff4d6d" if cl_pred==1 else "#00e5a0")+"'>" + verdict + "</b>" if agree else "Review individual confidences above"}
                          </div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Feature breakdown ──────────────────────
                with st.expander("▸ Feature breakdown (48 extracted values)"):
                    feat_names = (
                        [f'{ch}_{n}' for ch in ['R','G','B']
                         for n in ['lsb_mean','lsb_std','even_odd','chi2','entropy',
                                   'dct_mean','dct_std','dct_p25','dct_p75','diff_h','diff_v']]
                        + ['corr_RG','corr_RB','corr_GB']
                        + [f'bitplane_{i}_std' for i in range(4)]
                        + ['global_mean','global_std','pct5','pct95',
                           'dynamic_range','median','RG_diff','GB_diff']
                    )
                    feat_df = pd.DataFrame({
                        'Feature': feat_names[:len(feats)],
                        'Value': feats
                    })
                    st.dataframe(
                        feat_df.style.background_gradient(
                            subset=['Value'], cmap='Blues'),
                        use_container_width=True, height=260
                    )

    else:
        st.markdown("""
        <div style='text-align:center;padding:4rem 2rem;color:#64748b'>
          <div style='font-size:3rem;margin-bottom:1rem'>⬆</div>
          <div style='font-family:Space Mono,monospace;font-size:0.9rem;
                      color:#94a3b8;margin-bottom:0.5rem'>
            Upload an image to begin analysis
          </div>
          <div style='font-size:0.8rem'>
            Supports PNG · JPG · BMP · TIFF
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 2 — METRICS DASHBOARD
# ══════════════════════════════════════════════
with tab_metrics:
    if not ml_loaded and not qml_loaded:
        st.warning("No models loaded. Check sidebar paths.")
    else:
        st.markdown('<div class="section-title">Performance Summary</div>',
                    unsafe_allow_html=True)

        # ── Key metric cards ──────────────────────────────
        m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
        def metric_card(col, label, val, delta=None, is_pct=True):
            fmt = f"{val*100:.1f}%" if is_pct else f"{val:.4f}"
            with col:
                st.metric(label, fmt,
                          delta=f"{delta:+.1%}" if delta is not None else None)

        rf_time_s  = classical_metrics.get('training_time_s', None)
        qml_time_s = None
        if qml_loaded:
            ci_t = qml_metrics_data.get('circuit_info', {})
            qml_time_s = ci_t.get('training_time_s', None)

        if ml_loaded:
            cm = classical_metrics['splits'][active_split]
            metric_card(m1, "RF Accuracy",  cm['accuracy'])
            metric_card(m2, "RF Precision", cm['precision'])
            metric_card(m3, "RF F1",        cm['f1_score'])
            with m4:
                rf_t = f"{rf_time_s:.0f}s" if rf_time_s else "—"
                st.metric("RF Train time", rf_t)
        if qml_loaded:
            qm = qml_metrics_data['splits'][active_split]
            delta_acc = (qm['accuracy'] - classical_metrics['splits'][active_split]['accuracy']
                         ) if ml_loaded else None
            metric_card(m5, "VQC Accuracy",  qm['accuracy'],  delta_acc)
            metric_card(m6, "VQC Precision", qm['precision'])
            metric_card(m7, "VQC F1",        qm['f1_score'])
            with m8:
                qml_t = f"{qml_time_s:.0f}s" if qml_time_s else "—"
                delta_t = None
                if rf_time_s and qml_time_s:
                    delta_t = f"+{qml_time_s - rf_time_s:.0f}s slower"
                st.metric("VQC Train time", qml_t, delta=delta_t)

        st.markdown('<div class="section-title">Comparative Analysis</div>',
                    unsafe_allow_html=True)

        row1_l, row1_r = st.columns(2, gap="large")

        with row1_l:
            st.markdown("##### Radar — metric profile")
            if ml_loaded and qml_loaded:
                st.plotly_chart(radar_chart(classical_metrics, qml_metrics_data, active_split),
                                use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Both models needed for comparison.")

        with row1_r:
            st.markdown("##### Bar — side-by-side metrics")
            if ml_loaded and qml_loaded:
                st.plotly_chart(bar_comparison(classical_metrics, qml_metrics_data, active_split),
                                use_container_width=True, config={'displayModeBar': False})

        # ── Confusion matrices ────────────────────────────
        st.markdown('<div class="section-title">Confusion Matrices</div>',
                    unsafe_allow_html=True)
        cm1, cm2 = st.columns(2, gap="large")

        with cm1:
            st.markdown("##### Classical RF")
            if ml_loaded:
                c_cm = classical_metrics['splits'][active_split]['confusion_matrix']
                st.plotly_chart(confusion_heatmap(c_cm, f"RF — {active_split}", "#00d4ff"),
                                use_container_width=True, config={'displayModeBar': False})

        with cm2:
            st.markdown("##### QML VQC")
            if qml_loaded:
                q_cm = qml_metrics_data['splits'][active_split]['confusion_matrix']
                st.plotly_chart(confusion_heatmap(q_cm, f"VQC — {active_split}", "#7c3aed"),
                                use_container_width=True, config={'displayModeBar': False})

        # ── All splits table ──────────────────────────────
        st.markdown('<div class="section-title">All Splits — Full Table</div>',
                    unsafe_allow_html=True)

        rows = []
        for split in ['train','val','test']:
            if ml_loaded and split in classical_metrics['splits']:
                cm = classical_metrics['splits'][split]
                t_rf = f"{classical_metrics.get('training_time_s',0):.1f}s" if split=='train' and classical_metrics.get('training_time_s') else '—'
                rows.append({'Model':'Classical RF','Split':split.capitalize(),
                             'Accuracy':cm['accuracy'],'Precision':cm['precision'],
                             'Recall':cm['recall'],'F1':cm['f1_score'],
                             'Samples':cm.get('n_samples','—'),
                             'Train time': t_rf})
            if qml_loaded and split in qml_metrics_data['splits']:
                qm = qml_metrics_data['splits'][split]
                ci_t2 = qml_metrics_data.get('circuit_info',{})
                t_vqc = f"{ci_t2.get('training_time_s',0):.1f}s" if split=='train' and ci_t2.get('training_time_s') else '—'
                rows.append({'Model':'QML VQC','Split':split.capitalize(),
                             'Accuracy':qm['accuracy'],'Precision':qm['precision'],
                             'Recall':qm['recall'],'F1':qm['f1_score'],
                             'Samples':qm.get('n_samples','—'),
                             'Train time': t_vqc})

        if rows:
            df_table = pd.DataFrame(rows)
            st.dataframe(
                df_table.style
                    .format({'Accuracy':'{:.2%}','Precision':'{:.2%}',
                             'Recall':'{:.2%}','F1':'{:.2%}'})
                    .background_gradient(subset=['Accuracy','F1'], cmap='Blues'),
                use_container_width=True, hide_index=True,
            )

        # ── Feature importance ────────────────────────────
        if show_fi and fi_df is not None:
            st.markdown('<div class="section-title">Feature Importance — Classical RF</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(feature_importance_chart(fi_df),
                            use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════
#  TAB 3 — QUANTUM CIRCUIT
# ══════════════════════════════════════════════
with tab_circuit:
    if not qml_loaded:
        st.warning("QML model not loaded. Check the qml_output path in the sidebar.")
    else:
        ci = qml_metrics_data.get('circuit_info', {})

        st.markdown('<div class="section-title">Circuit Architecture</div>',
                    unsafe_allow_html=True)

        # Stats row
        s1,s2,s3,s4,s5 = st.columns(5)
        backend_val = ci.get("backend","Simulator")
        if isinstance(backend_val, str) and len(backend_val) > 18:
            backend_val = backend_val[:15] + "..."
        t_qml = ci.get("training_time_s")
        t_qml_str = f"{t_qml:.0f}s" if t_qml else "—"
        stats = [
            (s1, "Qubits",      ci.get("n_qubits","—")),
            (s2, "Depth",       ci.get("circuit_depth","—")),
            (s3, "Gate count",  ci.get("circuit_gates","—")),
            (s4, "Parameters",  ci.get("ansatz_params","—")),
            (s5, "Train time",  t_qml_str),
        ]
        for col, label, val in stats:
            with col:
                st.markdown(f"""
                <div class="steg-card steg-card-purple" style="text-align:center">
                  <div style='font-size:0.65rem;color:#64748b;text-transform:uppercase;
                              letter-spacing:1px'>{label}</div>
                  <div style='font-family:Space Mono,monospace;font-size:1.1rem;
                              color:#a78bfa;margin-top:0.3rem'>{val}</div>
                </div>""", unsafe_allow_html=True)

        # Model parameters — adapts to QSVM vs VQC
        model_type = qml_payload.get('model_type', 'VQC')
        st.markdown('<div class="section-title">Model Parameters</div>',
                    unsafe_allow_html=True)

        if model_type == 'QSVM':
            try:
                sv = qml_payload.get('support_vectors', None)
                sl = qml_payload.get('support_labels',  None)
                if sv is not None and len(sv) > 1:
                    sv_df = pd.DataFrame({
                        'PC1'  : sv[:, 0],
                        'PC2'  : sv[:, 1],
                        'Class': ['Clean' if l==0 else 'Stego' for l in sl]
                    })
                    fig_sv = px.scatter(
                        sv_df, x='PC1', y='PC2', color='Class',
                        color_discrete_map={'Clean':'#00e5a0','Stego':'#ff4d6d'},
                        title='Support Vectors (PCA space)',
                        height=320,
                    )
                    fig_sv.update_layout(
                        **DARK,
                        legend=dict(bgcolor='rgba(0,0,0,0)'),
                        margin=dict(l=20,r=20,t=40,b=20),
                        xaxis=dict(gridcolor='#1e2d45', title='PC1'),
                        yaxis=dict(gridcolor='#1e2d45', title='PC2'),
                    )
                    fig_sv.update_traces(marker=dict(size=8, opacity=0.8))
                    st.plotly_chart(fig_sv, use_container_width=True,
                                    config={'displayModeBar': False})
                    n_clean = int((sl==0).sum())
                    n_stego = int((sl==1).sum())
                    st.markdown(
                        f"<p style='font-size:0.72rem;color:#64748b;text-align:center'>"
                        f"{len(sv)} support vectors · {n_clean} clean · "
                        f"{n_stego} stego · C={qml_payload.get('C','—')}</p>",
                        unsafe_allow_html=True)
            except Exception as e:
                st.info(f"Support vector plot unavailable: {e}")
        else:
            w = qml_weights
            if len(w) > 1:
                n_cols = min(16, len(w))
                w_grid = np.pad(w, (0, (-len(w)) % n_cols)).reshape(-1, n_cols)
                fig_w  = px.imshow(w_grid, color_continuous_scale='RdBu',
                                   zmin=-np.pi, zmax=np.pi, aspect='auto')
                fig_w.update_layout(**DARK, height=160,
                                    coloraxis_colorbar=dict(
                                        title='θ (rad)', tickfont=dict(size=9),
                                        len=0.8, thickness=10),
                                    margin=dict(l=0,r=60,t=0,b=0),
                                    xaxis=dict(showticklabels=False),
                                    yaxis=dict(showticklabels=False))
                st.plotly_chart(fig_w, use_container_width=True,
                                config={'displayModeBar': False})
                st.markdown(
                    f"<p style='font-size:0.72rem;color:#64748b;text-align:center'>"
                    f"{len(w)} trainable rotation angles · "
                    f"range [{w.min():.3f}, {w.max():.3f}] rad</p>",
                    unsafe_allow_html=True)
            else:
                st.info("QSVM uses kernel-based classification — no rotation angles.")

        # Visual outputs
        st.markdown('<div class="section-title">Visual Outputs</div>',
                    unsafe_allow_html=True)

        img_cols = [
            ('circuit_diagram',       'Circuit Diagram',          '1x'),
            ('qsphere',               'Q-Sphere',                 '1x'),
            ('measurement_histogram', 'Measurement Histogram',    '1x'),
        ]

        c1, c2, c3 = st.columns(3, gap="large")
        for col, (key, title, _) in zip([c1,c2,c3], img_cols):
            with col:
                st.markdown(f"##### {title}")
                if key in qml_imgs:
                    st.image(qml_imgs[key], use_container_width=True)
                else:
                    st.markdown("""<div class="steg-card" style="text-align:center;
                        color:#64748b;padding:2rem">Image not found</div>""",
                                unsafe_allow_html=True)

        # Training curve
        if 'training_curve' in qml_imgs:
            st.markdown('<div class="section-title">Training Curve</div>',
                        unsafe_allow_html=True)
            st.image(qml_imgs['training_curve'], use_container_width=True)

        # Raw weights expander
        with st.expander("▸ View raw trained weights"):
            w_df = pd.DataFrame({
                'Index': range(len(qml_weights)),
                'Weight (rad)': qml_weights,
                'Weight (°)': np.degrees(qml_weights),
            })
            st.dataframe(w_df.style.background_gradient(
                subset=['Weight (rad)'], cmap='RdBu'),
                use_container_width=True, height=280)

        # Circuit info JSON
        with st.expander("▸ Full circuit metadata"):
            st.json(ci)