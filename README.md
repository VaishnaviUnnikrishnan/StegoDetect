# Steganalysis VQC

A comparative study of **Classical Machine Learning** and **Quantum Machine Learning** for detecting steganography in images. This project implements a binary classifier to distinguish between clean images and images containing hidden data using LSB (Least Significant Bit) steganography.

## 🚀 Features

- **Feature Extraction**: 48 statistical features from image analysis (LSB statistics, DCT coefficients, pixel correlations, bit-plane noise)
- **Classical ML**: Random Forest classifier with feature importance analysis
- **Quantum ML**: Quantum Support Vector Machine (QSVM) using PegasosQSVC with ZZFeatureMap kernel
- **Interactive Dashboard**: Streamlit web app for real-time image analysis and model comparison
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrices across train/val/test splits

## 📊 Project Overview

### Dataset
- **Source**: Stego-pvd-dataset (custom dataset with PVD steganography)
- **Splits**: Train (balanced), Validation, Test
- **Classes**: Clean vs Stego (binary classification)
- **Format**: PNG images with LSB-embedded hidden data

### Feature Engineering
48 features extracted per image:
- **LSB Analysis**: Mean, std, even/odd ratios, chi-square uniformity
- **Histogram Entropy**: Per color channel
- **DCT Statistics**: Mean, std, percentiles from central block
- **Pixel Differences**: Horizontal/vertical adjacent pixel differences
- **Color Correlations**: RGB channel correlations
- **Bit-Plane Noise**: Standard deviation across 4 bit planes
- **Global Statistics**: Mean, std, percentiles, dynamic range, median

### Models

#### Classical ML (Random Forest)
- **Algorithm**: RandomForestClassifier (300 estimators)
- **Preprocessing**: None (RF handles raw features well)
- **Output**: `ml_output/stego_classifier_rf.pkl`, `metrics.json`, `feature_importance.csv`

#### Quantum ML (QSVM)
- **Algorithm**: PegasosQSVC with FidelityQuantumKernel
- **Feature Map**: ZZFeatureMap (reps=2)
- **Dimensionality Reduction**: PCA to 4 qubits (retains ~95% variance)
- **Backend**: Local AerSimulator (StatevectorSampler) or IBM Quantum
- **Output**: `qml_output/qml_vqc_model.pkl`, `qml_metrics.json`, circuit visualizations

## 🛠 Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/VaishnaviUnnikrishnan/StegoDetect.git
cd StegoDetect

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
streamlit
numpy
pandas
scikit-learn
qiskit
qiskit-machine-learning
matplotlib
seaborn
plotly
Pillow
scipy
joblib
```

##  Usage

### 1. Train Classical ML Model
```bash
python classical_ml_train.py \
    --train ./Stego-pvd-dataset/stego_train \
    --val ./Stego-pvd-dataset/stego_val \
    --test ./Stego-pvd-dataset/stego_test \
    --output ./ml_output
```

**Options:**
- `--model`: `rf` (default), `gb`, or `svm`
- Outputs feature arrays (`.npy`) for QML reuse

### 2. Train Quantum ML Model
```bash
python qml_train.py \
    --data ./ml_output \
    --output ./qml_output \
    --n-qubits 4 \
    --reps 2 \
    --C 100.0 \
    --num-steps 200 \
    --train-size 50
```

**Options:**
- `--ibm-token`: Use IBM Quantum backend (requires account)
- `--ibm-backend`: Backend name (default: `ibm_brisbane`)

### 3. Run Interactive Dashboard
```bash
streamlit run app.py
```

Upload images to get predictions from both models with confidence scores, feature breakdowns, and comparative metrics.

##  Results

### Performance Comparison (Test Set)
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.85 | 0.83 | 0.87 | 0.85 |
| QSVM (4 qubits) | 0.82 | 0.81 | 0.84 | 0.82 |

### Key Insights
- **Classical ML**: Robust baseline with interpretable feature importance
- **Quantum ML**: Competitive performance with quantum advantage potential on larger datasets
- **Feature Importance**: LSB statistics and DCT coefficients most predictive
- **Training Time**: RF (~30s), QSVM (~120s on local simulator)

##  Project Structure
```
Steganalysis_VQC/
├── app.py                    # Streamlit dashboard
├── classical_ml_train.py     # Classical ML training script
├── qml_train.py             # Quantum ML training script
├── repair_qml_metrics.py    # Metrics repair utility
├── stego_encoder.py         # Steganography encoding utility
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── ml_output/               # Classical ML outputs
│   ├── stego_classifier_rf.pkl
│   ├── metrics.json
│   ├── feature_importance.csv
│   └── *.npy (feature arrays)
├── qml_output/              # Quantum ML outputs
│   ├── qml_vqc_model.pkl
│   ├── qml_metrics.json
│   ├── vqc_weights.npy
│   └── *.png (visualizations)
├── models/                  # Saved models (ignored)
├── results/                 # Additional results (ignored)
└── Stego-pvd-dataset/       # Dataset (ignored - large files)
    ├── docs/                # Documentation
    ├── stego_train/         # Training data
    ├── stego_val/           # Validation data
    └── stego_test/          # Test data
```

## 🎯 Methodology

### Classical Pipeline
1. **Feature Extraction**: 48 statistical features from image analysis
2. **Training**: Random Forest with balanced class weights
3. **Evaluation**: Standard ML metrics + feature importance

### Quantum Pipeline
1. **Preprocessing**: MinMax scaling + PCA to 4D quantum space
2. **Kernel Construction**: ZZFeatureMap → FidelityQuantumKernel
3. **Training**: PegasosQSVC optimization
4. **Prediction**: Quantum kernel evaluation

### Dashboard Features
- **Image Upload**: Support for PNG, JPG, BMP, TIFF
- **Dual Prediction**: Classical + Quantum model predictions
- **Confidence Visualization**: Progress bars and agreement indicators
- **Feature Analysis**: Breakdown of 48 extracted features
- **Model Comparison**: Radar charts, bar plots, confusion matrices
- **Quantum Circuit**: Circuit diagrams, Q-sphere, training curves

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Outputs
<img width="1365" height="614" alt="image" src="https://github.com/user-attachments/assets/4f56360b-b4f7-4bac-978b-6e05854fd22f" />
<img width="1361" height="603" alt="image" src="https://github.com/user-attachments/assets/d6dea273-f7f2-4d51-8218-0cb3d8838569" />
<img width="1362" height="601" alt="image" src="https://github.com/user-attachments/assets/d71d62a6-8e9f-4d21-93ae-aad4b94f6280" />






Project Link: [https://github.com/VaishnaviUnnikrishnan/StegoDetect](https://github.com/VaishnaviUnnikrishnan/StegoDetect)
A comparative study between RF classifier and VQC classifier to detect LSB Steganographic Images.
