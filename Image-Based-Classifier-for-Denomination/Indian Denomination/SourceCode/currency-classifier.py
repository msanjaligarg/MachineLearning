# Standard library imports
import os
import zipfile
import time
import random

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Google Colab specific
from google.colab import files

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from tensorflow.keras.optimizers import Adam

def load_images(folder, label, size=(128, 128)):
    """Load and label images from a folder for traditional models"""
    images = []
    if not os.path.exists(folder):
        print(f"Folder '{folder}' not found. Skipping.")
        return []
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, size)
                    images.append((img, label))
                else:
                    print(f"Could not load image: {img_path}")
    return images

def extract_images_from_zip(zip_bytes, label, size=(128, 128)):
    """Extract and process images directly from zip file bytes"""
    images = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                with zip_ref.open(file_info) as file:
                    img_bytes = file.read()
                    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.resize(img, size)
                        images.append((img, label))
    return images

# Unified File Upload and Processing

print("Upload ZIP files for both ₹100 and ₹200 notes")
uploaded = files.upload()

# Process for both traditional and hybrid approaches
data_traditional = []
data_hybrid = []

for filename, file_bytes in uploaded.items():
    if filename.endswith('.zip'):
        label = 0 if '100' in filename.lower() else 1

        # Traditional approach: extract to folder
        folder = filename.replace('.zip', '')
        os.makedirs(folder, exist_ok=True)
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(folder)
        print(f"Extracted {filename} to folder: {folder}")
        data_traditional.extend(load_images(folder, label))

        # Hybrid approach: process directly from zip bytes
        data_hybrid.extend(extract_images_from_zip(file_bytes, label))

# Traditional ML Models Data Preparation
if not data_traditional:
    raise SystemExit("No images were loaded for traditional models.")
else:
    random.shuffle(data_traditional)
    X_trad = np.array([img for img, _ in data_traditional], dtype=np.float32) / 255.0
    y_trad = np.array([label for _, label in data_traditional])
    X_flat = X_trad.reshape(X_trad.shape[0], -1)

# Hybrid ML Models Data Preparation
if not data_hybrid:
    raise SystemExit("No images were loaded for hybrid models.")
else:
    random.shuffle(data_hybrid)
    images, labels = zip(*data_hybrid)
    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels)

    # Train/test split (common for both approaches)
    split = int(0.8 * len(images))
    X_train, X_test = images[:split], images[split:]
    y_train, y_test = labels[:split], labels[split:]

    print(f"\nDataset Summary:")
    print(f"Traditional: {len(data_traditional)} images")
    print(f"Hybrid: {len(data_hybrid)} images")
    print(f"Train set: {len(X_train)} | Test set: {len(X_test)}")

# Train/Val/Test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
# Corrected test_state to test_size
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Traditional ML Models
    results = {}

    def evaluate(name, model, X_test, y_test):
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"\n{name} Accuracy: {acc*100:.2f}%")
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred))
        return acc

    # 1) Logistic Regression
    lr = LogisticRegression(max_iter=1000).fit(X_train_flat, y_train)
    results["Logistic Regression"] = evaluate("Logistic Regression", lr, X_test_flat, y_test)

    # 2) Decision Tree
    dt = DecisionTreeClassifier().fit(X_train_flat, y_train)
    results["Decision Tree"] = evaluate("Decision Tree", dt, X_test_flat, y_test)

    # 3) Random Forest
    rf = RandomForestClassifier().fit(X_train_flat, y_train)
    results["Random Forest"] = evaluate("Random Forest", rf, X_test_flat, y_test)

    # 4) SVM
    svm = SVC().fit(X_train_flat, y_train)
    results["SVM"] = evaluate("SVM", svm, X_test_flat, y_test)

    # 5) KNN
    knn = KNeighborsClassifier().fit(X_train_flat, y_train)
    results["KNN"] = evaluate("KNN", knn, X_test_flat, y_test)

    # 6) KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_train_flat)
    kmeans_pred = kmeans.predict(X_test_flat)
    acc_kmeans = accuracy_score(y_test, kmeans_pred)
    print(f"\n KMeans Accuracy: {acc_kmeans*100:.2f}%")
    results["KMeans"] = acc_kmeans

    # 7) Agglomerative Clustering
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
    agg = AgglomerativeClustering(n_clusters=2)
    agg_pred = agg.fit_predict(X_test_flat)

    # Calculate metrics
    acc_agg = accuracy_score(y_test, agg_pred)
    ars_agg = adjusted_rand_score(y_test, agg_pred)
    nmi_agg = normalized_mutual_info_score(y_test, agg_pred)

    print(f"\n Hierarchical Clustering Performance:")
    print(f"  - Accuracy: {acc_agg*100:.2f}%")
    print(f"  - Adjusted Rand Score (ARS): {ars_agg:.4f}")
    print(f"  - Normalized Mutual Info (NMI): {nmi_agg:.4f}")

    # Store results in a dictionary
    results["Hierarchical"] = {
        "Accuracy": acc_agg,
        "Adjusted_Rand_Score": ars_agg,
        "NMI": nmi_agg
    }

    # 8) CNN Model
    if X_train.shape[1:] != (128, 128, 3):
         print(f"Warning: CNN input shape mismatch. Expected (128, 128, 3), got {X_train.shape[1:]}")
         print("Skipping CNN model training.")
    else:
        cnn = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        cnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        print("\nStarting CNN training...")
        try:
            cnn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
            acc_cnn = cnn.evaluate(X_test, y_test, verbose=0)[1]
            results["CNN"] = acc_cnn
            print(f"\n CNN Accuracy: {acc_cnn*100:.2f}%")
        except Exception as e:
            print(f" Error during CNN training: {e}")

    # 9) LSTM Model
    if X_train.shape[1] != 128 or X_train.shape[2] != 128 or X_train.shape[3] != 3:
         print(f" Warning: LSTM input shape mismatch. Expected (None, 128, 128, 3) before reshape, got {X_train.shape}")
         print("Skipping LSTM model training.")
    else:
        X_train_seq = X_train.reshape((X_train.shape[0], 128, -1))
        X_val_seq = X_val.reshape((X_val.shape[0], 128, -1))
        X_test_seq = X_test.reshape((X_test.shape[0], 128, -1))

        lstm = Sequential([
            LSTM(64, input_shape=(128, 128*3)),
            Dense(1, activation='sigmoid')
        ])
        lstm.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        print("\n Starting LSTM training...")
        try:
            lstm.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), epochs=10)
            acc_lstm = lstm.evaluate(X_test_seq, y_test, verbose=0)[1]
            results["LSTM"] = acc_lstm
            print(f"\n LSTM Accuracy: {acc_lstm*100:.2f}%")
        except Exception as e:
            print(f" Error during LSTM training: {e}")

    # Visualization
    if results:
        plt.figure(figsize=(12,6))
        sns.barplot(x=list(results.keys()), y=[v*100 for v in results.values()])
        plt.title("Accuracy Comparison of All Models")
        plt.ylabel("Accuracy (%)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.ylim(0, 100)
        plt.show()
    else:
        print("\n No model results to display.")

!pip install -q qiskit qiskit-aer scikit-image scikit-learn matplotlib

# Hybrid ML Models
import numpy as np, zipfile, io, time, warnings, math, random, hashlib
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay
)
from sklearn.preprocessing import StandardScaler
from scipy.signal import convolve2d
from skimage.color import rgb2gray
warnings.filterwarnings("ignore")

def get_backend_executor(shots=1024):
    try:
        from qiskit.primitives import Sampler
        sampler = Sampler()
        print("Using Sampler backend")
        def run(circ):
            qdist = sampler.run([circ]).result().quasi_dists[0]
            return {k: int(v * shots) for k, v in qdist.items()}
        return run
    except Exception as e_sampler:
        try:
            from qiskit import Aer, transpile
            backend = Aer.get_backend("qasm_simulator")
            print("Using Aer qasm_simulator")
            def run(circ):
                circ.measure_all()
                job = backend.run(transpile(circ, backend), shots=shots)
                return job.result().get_counts()
            return run
        except Exception as e_aer:
            print("No quantum backend found — using dummy simulator")
            random.seed(0)
            def run(_circ):
                # Deterministic pseudo‑counts (60 / 40 split)
                return {"0": int(0.6 * shots), "1": int(0.4 * shots)}
            return run

EXECUTE_CIRCUIT = get_backend_executor()
SHOTS = 1024  # global shots for all models

def extract_images_from_zip(zip_bytes, label, size=(64, 64)):
    imgs = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                with zf.open(name) as f:
                    img = Image.open(f).convert("RGB").resize(size)
                    imgs.append((np.array(img), label))
    return imgs

def compute_metrics(y_true, y_pred, y_proba=None):
    m = {
        "accuracy":               accuracy_score(y_true, y_pred),
        "balanced_accuracy":      balanced_accuracy_score(y_true, y_pred),
        "precision":              precision_score(y_true, y_pred),
        "recall":                 recall_score(y_true, y_pred),
        "f1":                     f1_score(y_true, y_pred),
        "mcc":                    matthews_corrcoef(y_true, y_pred),
        "kappa":                  cohen_kappa_score(y_true, y_pred),
        "conf_mat":               confusion_matrix(y_true, y_pred),
        "class_report":           classification_report(y_true, y_pred, output_dict=True)
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_proba)
        except:
            m["roc_auc"] = None
    else:
        m["roc_auc"] = None
    return m

def plot_results(metrics, title_prefix=""):
    cm = metrics["conf_mat"]
    names = ["accuracy","precision","recall","f1"]
    vals  = [metrics[k] for k in names]

    plt.figure(figsize=(10,4))

    # Confusion matrix
    plt.subplot(1,2,1)
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{title_prefix} Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j, i, v, ha="center", va="center", color="black")
    plt.colorbar()

    # Bar chart
    plt.subplot(1,2,2)
    plt.bar(names, vals)
    plt.ylim(0,1); plt.title(f"{title_prefix} Metrics")
    plt.show()

from qiskit import QuantumCircuit

class QuantumModelBase:
    def __init__(self, n_qubits=4):
        self.n_qubits = n_qubits
        self.scaler   = StandardScaler()

    def fit(self, X, y):
        return self
    def predict(self, X):
        feats = self._extract_features(X)
        preds, probas = [], []
        for v in feats:
            qc = self._build_circuit(v)
            counts = EXECUTE_CIRCUIT(qc)
            prob0 = counts.get("0", 0) / SHOTS
            preds.append(0 if prob0 > 0.5 else 1)
            probas.append(prob0)
        return np.array(preds), np.array(probas)

    def _extract_features(self, X): raise NotImplementedError
    def _build_circuit(self, feature_vec): raise NotImplementedError

# 1) Q-CNN
class QCNN(QuantumModelBase):
    def _extract_features(self, X):
        kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
        feats=[]
        for img in X:
            gray = rgb2gray(img)
            conv = convolve2d(gray, kernel, mode="valid")
            pooled = conv[::2,::2]
            feats.append(pooled.flatten()[:self.n_qubits])
        return self.scaler.fit_transform(feats)
    def _build_circuit(self, v):
        qc = QuantumCircuit(self.n_qubits)
        for i,x in enumerate(v):
            qc.ry(float(x)*np.pi, i)
        for i in range(self.n_qubits-1):
            qc.cx(i, i+1)
        qc.measure_all()
        return qc

# 2) Q-LSTM (tiny classical LSTM encoder + quantum readout)
class QLSTM(QuantumModelBase):
    def __init__(self, n_qubits=4):
        super().__init__(n_qubits)
        self.Wf = np.random.randn(n_qubits,n_qubits)*0.05
        self.Wi = np.random.randn(n_qubits,n_qubits)*0.05
        self.Wo = np.random.randn(n_qubits,n_qubits)*0.05
        self.Wc = np.random.randn(n_qubits,n_qubits)*0.05
    def _sigmoid(self,z): return 1/(1+np.exp(-z))
    def _lstm_seq(self, seq):
        h = c = np.zeros(self.n_qubits)
        for x in seq:
            f = self._sigmoid(self.Wf@(x+h))
            i = self._sigmoid(self.Wi@(x+h))
            o = self._sigmoid(self.Wo@(x+h))
            c = f*c + i*np.tanh(self.Wc@x)
            h = o*np.tanh(c)
        return h
    def _extract_features(self, X):
        feats=[]
        for img in X:
            gray = rgb2gray(img).flatten()
            seq = gray.reshape(-1,self.n_qubits)[:32]
            feats.append(self._lstm_seq(seq))
        return self.scaler.fit_transform(feats)
    def _build_circuit(self, v):
        qc = QuantumCircuit(self.n_qubits)
        for i,x in enumerate(v):
            qc.rx(float(x)*np.pi, i)
        for i in range(self.n_qubits-1):
            qc.cz(i, i+1)
        qc.measure_all()
        return qc

# 3) Q-SVM (simple quantum kernel 1‑NN)
class QSVM(QuantumModelBase):
    def fit(self, X, y):
        feats = self._extract_features(X)
        self.class0 = feats[y==0][0]
        self.class1 = feats[y==1][0]
        return self
    def _extract_features(self, X):
        return QCNN()._extract_features(X)
    def _kernel(self, v1, v2):
        qc = QuantumCircuit(self.n_qubits)
        for i,(a,b) in enumerate(zip(v1,v2)):
            qc.ry(a*np.pi, i)
            qc.ry(-b*np.pi, i)      # inverse embedding
        for i in range(self.n_qubits-1):
            qc.cz(i,i+1)
        qc.measure_all()
        counts = EXECUTE_CIRCUIT(qc)
        return counts.get("0",0)/SHOTS
    def predict(self, X):
        feats = self._extract_features(X)
        preds, probas = [], []
        for v in feats:
            s0 = self._kernel(v,self.class0)
            s1 = self._kernel(v,self.class1)
            prob0 = (s0+1e-6)/(s0+s1+1e-6)  # normalize
            preds.append(0 if prob0>0.5 else 1)
            probas.append(prob0)
        return np.array(preds), np.array(probas)

# Train / evaluate / visualize for each model
all_metrics = {} # Dictionary to store metrics for all models

for ModelClass, name in [(QCNN, "QCNN"), (QLSTM, "QLSTM"), (QSVM, "QSVM")]:
    print(f"\n Running {name} …")
    model = ModelClass()

    if name == "QSVM":
        model.fit(X_train, y_train)
    y_pred, y_proba = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    all_metrics[name] = metrics
    print(f"\n=== {name} Performance Report ===")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1 Score:           {metrics['f1']:.4f}")
    print(f"ROC AUC:            {metrics.get('roc_auc', 'N/A')}")
    print(f"MCC:                {metrics['mcc']:.4f}")
    print(f"Cohen's Kappa:      {metrics['kappa']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    plot_results(metrics, title_prefix=name)

# Summarize all results
print("\n==== Overall Performance Summary ====")
for model_name, metrics in all_metrics.items():
    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC AUC: {metrics.get('roc_auc', 'N/A')}")
