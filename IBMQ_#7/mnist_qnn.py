"""
手書き数字 0/1 分類 - 量子ニューラルネットワーク (QNN) デモ
==============================================================
・sklearn 内蔵の手書き数字データセット (8x8px) から「0」と「1」を抽出
  ※ MNIST (28x28) と同じ流れ：実際の MNIST でも fetch_openml を
     fetch_openml("mnist_784") に差し替えるだけで動作します
・PCA で 4 次元に圧縮 → [0, π] にスケーリング
・4 量子ビット QNN (ZZFeatureMap + RealAmplitudes) で学習
・学習はシミュレータ (StatevectorEstimator) 上で実行
"""

import io, os, sys

# Windows 文字化け対策 -------------------------------------------------------
if sys.platform == "win32" and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONUTF8"] = "1"

# ---------------------------------------------------------------------------
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.optimizers import COBYLA

import warnings
warnings.filterwarnings("ignore")

# ===========================================================================
# 1. データ準備（sklearn 内蔵 Digits データセット、8x8px）
# ===========================================================================
RANDOM_SEED = 42
N_QUBITS    = 4     # PCA 次元 = 量子ビット数
TEST_SIZE   = 0.3

print("=== [1] データ読み込み ===")
digits = load_digits()
X_all, y_all = digits.data.astype(float), digits.target.astype(int)

# 0 と 1 だけ抽出
mask01   = (y_all == 0) | (y_all == 1)
X01, y01 = X_all[mask01], y_all[mask01]
print(f"  抽出サンプル: {len(X01)}  (0: {(y01==0).sum()}, 1: {(y01==1).sum()})")

# ラベルを {0, 1} → {-1, +1} に変換（EstimatorQNN は ±1 形式）
y_pm = np.where(y01 == 0, -1.0, +1.0)

# ===========================================================================
# 2. PCA で 4 次元に圧縮
# ===========================================================================
print("=== [2] PCA で 4 次元に圧縮 ===")
pca      = PCA(n_components=N_QUBITS, random_state=RANDOM_SEED)
X_pca    = pca.fit_transform(X01)
cum_var  = pca.explained_variance_ratio_.cumsum()[-1] * 100
print(f"  累積寄与率: {cum_var:.1f}%")

# [0, π] にスケーリング（Rz / Ry の回転角として量子回路に入力）
scaler   = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_pca)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_pm,
    test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_pm
)
print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

# ===========================================================================
# 3. 量子回路の構築
# ===========================================================================
print("=== [3] 量子回路構築 ===")

# -- 特徴マップ: ZZFeatureMap --
#    |0⟩^⊗4 → 古典データを量子振幅にエンコード（回転角 + ZZ もつれ）
feature_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=1)

# -- 変分 Ansatz: RealAmplitudes --
#    学習可能なパラメータ θ を持つ変分回路
ansatz = RealAmplitudes(num_qubits=N_QUBITS, reps=2)

# -- 特徴マップ + Ansatz を結合 --
qc = QuantumCircuit(N_QUBITS)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

n_feat_params   = feature_map.num_parameters
n_weight_params = ansatz.num_parameters
print(f"  量子ビット数   : {N_QUBITS}")
print(f"  特徴量パラメータ: {n_feat_params}")
print(f"  重みパラメータ  : {n_weight_params}")

# -- 観測量: 0 番目量子ビットの Z 期待値 → 符号でクラス分類 --
observable = SparsePauliOp("IIIZ")   # Qiskit: LSB = qubit 0

# -- EstimatorQNN: 回路の期待値を順伝播として使う --
estimator = StatevectorEstimator()
qnn = EstimatorQNN(
    circuit=qc,
    observables=observable,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    estimator=estimator,
)

# ===========================================================================
# 4. 学習
# ===========================================================================
print("=== [4] 学習開始 (COBYLA, maxiter=200) ===")

loss_history = []

def callback(weights, obj_val):
    loss_history.append(float(obj_val))
    if len(loss_history) % 20 == 0:
        print(f"  iter {len(loss_history):4d}  loss = {obj_val:.4f}", flush=True)

np.random.seed(RANDOM_SEED)
initial_point = np.random.uniform(-np.pi, np.pi, n_weight_params)

clf = NeuralNetworkClassifier(
    neural_network=qnn,
    optimizer=COBYLA(maxiter=200),
    callback=callback,
    initial_point=initial_point,
)
clf.fit(X_train, y_train)

# ===========================================================================
# 5. 評価
# ===========================================================================
print("=== [5] 評価 ===")
y_pred_train = clf.predict(X_train).ravel()
y_pred_test  = clf.predict(X_test).ravel()

acc_train = accuracy_score(y_train.ravel(), y_pred_train)
acc_test  = accuracy_score(y_test.ravel(),  y_pred_test)
print(f"  Train Accuracy : {acc_train*100:.1f}%")
print(f"  Test  Accuracy : {acc_test*100:.1f}%")

# ===========================================================================
# 6. 可視化 ① — 学習曲線 + PCA 散布図 + 混同行列
# ===========================================================================
print("=== [6] 図を保存 ===")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) 損失曲線
axes[0].plot(range(1, len(loss_history)+1), loss_history, color="#3b82d4", linewidth=1.5)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Objective Value")
axes[0].set_title("(a) Training Loss Curve")
axes[0].grid(True, alpha=0.3)

# (b) PCA 散布図 (PC1 vs PC2)
X_test_pca  = scaler.inverse_transform(X_test)   # スケールを PCA 空間に戻す
y_test_flat = y_test.ravel()
for label, color, marker, name in [(-1, "#3b82d4", "o", "0"), (1, "#ef4444", "s", "1")]:
    sel = y_test_flat == label
    axes[1].scatter(X_test_pca[sel, 0], X_test_pca[sel, 1],
                    c=color, marker=marker, label=f"Digit {name}", alpha=0.7, s=40)
wrong = (y_pred_test != y_test_flat)
axes[1].scatter(X_test_pca[wrong, 0], X_test_pca[wrong, 1],
                c="black", marker="x", s=90, linewidths=1.8,
                label="Misclassified", zorder=5)
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("(b) PCA Space (Test set)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# (c) 混同行列
cm = confusion_matrix(y_test_flat, y_pred_test, labels=[-1, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Digit 0", "Digit 1"])
disp.plot(ax=axes[2], colorbar=False, cmap="Blues")
axes[2].set_title(f"(c) Confusion Matrix  (Test={acc_test*100:.1f}%)")

fig.suptitle("Quantum Neural Network — Handwritten 0/1 Classification", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("mnist_qnn_result.png", dpi=150, bbox_inches="tight")
print("  -> mnist_qnn_result.png")

# ===========================================================================
# 7. 可視化 ② — サンプル画像（正解・誤分類）
# ===========================================================================
fig2, axes2 = plt.subplots(2, 8, figsize=(14, 4))
img_size = int(np.sqrt(X01.shape[1]))  # 8 (8x8 px)

correct_idx   = np.where(~wrong)[0][:8]
incorrect_idx = np.where( wrong)[0]

# テストデータを元の画像空間に逆変換
X_test_orig = pca.inverse_transform(scaler.inverse_transform(X_test))

for col in range(8):
    if col < len(correct_idx):
        ci = correct_idx[col]
        axes2[0, col].imshow(X_test_orig[ci].reshape(img_size, img_size), cmap="gray")
        t = "0" if y_test_flat[ci] == -1 else "1"
        p = "0" if y_pred_test[ci] == -1 else "1"
        axes2[0, col].set_title(f"T:{t}/P:{p}", fontsize=7)
    axes2[0, col].axis("off")

for col in range(8):
    if col < len(incorrect_idx):
        ci = incorrect_idx[col]
        axes2[1, col].imshow(X_test_orig[ci].reshape(img_size, img_size), cmap="gray")
        t = "0" if y_test_flat[ci] == -1 else "1"
        p = "0" if y_pred_test[ci] == -1 else "1"
        axes2[1, col].set_title(f"T:{t}/P:{p}", fontsize=7, color="red")
    axes2[1, col].axis("off")

axes2[0, 0].set_ylabel("Correct", fontsize=9)
axes2[1, 0].set_ylabel("Wrong",   fontsize=9)
fig2.suptitle("Sample Images — Correct (top) / Misclassified (bottom)", fontsize=11)
plt.tight_layout()
fig2.savefig("mnist_qnn_samples.png", dpi=150, bbox_inches="tight")
print("  -> mnist_qnn_samples.png")

print("\n===== 最終結果 =====")
print(f"  Train Accuracy : {acc_train*100:.1f}%")
print(f"  Test  Accuracy : {acc_test*100:.1f}%")
print("====================")
