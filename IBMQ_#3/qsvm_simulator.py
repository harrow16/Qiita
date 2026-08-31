"""
qsvm_simulator.py
シミュレーターを使った QSVM（量子カーネル SVM）の学習・評価スクリプト。

構成:
  zz_feature_map  ->  FidelityQuantumKernel (ComputeUncompute + StatevectorSampler)
  ->  SVC(kernel='precomputed')  ->  accuracy 評価
"""
import os

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from data_prep import get_iris_data

RESULTS_DIR = "results"


def run_qsvm_simulator() -> dict:
    """シミュレーターで QSVM を実行し、結果を返す。"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- データ準備 ---
    X_train, X_test, y_train, y_test = get_iris_data()
    print(f"訓練: {X_train.shape}, テスト: {X_test.shape}")

    # --- 量子特徴マップ ---
    # reps=1, alpha=1.0 がこのデータセットで最良のカーネル分離を示す
    feature_map = zz_feature_map(feature_dimension=2, reps=1, alpha=1.0)

    # 量子回路図を保存
    fig = feature_map.decompose().draw(output="mpl", fold=-1)
    circuit_path = os.path.join(RESULTS_DIR, "circuit_simulator.png")
    fig.savefig(circuit_path, bbox_inches="tight")
    print(f"量子回路図を保存: {circuit_path}")

    # --- 量子カーネル構築 ---
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

    # --- カーネル行列計算 ---
    print("訓練カーネル行列を計算中...")
    K_train = qkernel.evaluate(X_train)
    print("テストカーネル行列を計算中...")
    K_test = qkernel.evaluate(X_test, X_train)

    # カーネル行列を保存（可視化スクリプトで再利用）
    np.save(os.path.join(RESULTS_DIR, "kernel_train_sim.npy"), K_train)
    np.save(os.path.join(RESULTS_DIR, "kernel_test_sim.npy"), K_test)
    np.save(os.path.join(RESULTS_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(RESULTS_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(RESULTS_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(RESULTS_DIR, "y_test.npy"), y_test)

    # --- C パラメータを選択してから SVM 学習・予測 ---
    best_C, best_cv = 1.0, 0.0
    for C in [0.1, 1.0, 10.0]:
        svc_cv = SVC(kernel="precomputed", C=C)
        scores = cross_val_score(svc_cv, K_train, y_train, cv=5)
        if scores.mean() > best_cv:
            best_cv, best_C = scores.mean(), C
    print(f"最適 C={best_C} (CV accuracy={best_cv:.4f})")

    svc = SVC(kernel="precomputed", C=best_C)
    svc.fit(K_train, y_train)
    y_pred = svc.predict(K_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n[シミュレーター] Accuracy: {acc:.4f} ({int(acc * len(y_test))}/{len(y_test)})")

    return {
        "accuracy": acc,
        "y_test": y_test,
        "y_pred": y_pred,
        "K_train": K_train,
        "K_test": K_test,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "svc": svc,
    }


if __name__ == "__main__":
    run_qsvm_simulator()
