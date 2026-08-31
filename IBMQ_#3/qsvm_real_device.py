"""
qsvm_real_device.py
IBM Quantum 実機を使った QSVM の評価スクリプト。

- QiskitRuntimeService.least_busy() で最小待ち行列のバックエンドを自動選択
- SamplerV2 Primitive でカーネル行列を計算
- シミュレーター結果と比較して出力

使い方:
    python qsvm_real_device.py <IBM_QUANTUM_API_KEY>
"""
import os
import sys

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from data_prep import get_iris_data

RESULTS_DIR = "results"


def run_qsvm_real_device(api_key: str, sim_accuracy: float | None = None) -> dict:
    """
    IBM Quantum 実機で QSVM を実行し、結果を返す。

    Parameters
    ----------
    api_key : str
        IBM Quantum Platform の API キー
    sim_accuracy : float | None
        比較用のシミュレーター accuracy（表示のみ）
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- データ準備（シミュレーターと同じ分割を再利用）---
    sim_data_path = os.path.join(RESULTS_DIR, "X_train.npy")
    if os.path.exists(sim_data_path):
        X_train = np.load(os.path.join(RESULTS_DIR, "X_train.npy"))
        X_test  = np.load(os.path.join(RESULTS_DIR, "X_test.npy"))
        y_train = np.load(os.path.join(RESULTS_DIR, "y_train.npy"))
        y_test  = np.load(os.path.join(RESULTS_DIR, "y_test.npy"))
        print("シミュレーターと同じデータ分割を使用")
    else:
        X_train, X_test, y_train, y_test = get_iris_data()

    print(f"訓練: {X_train.shape}, テスト: {X_test.shape}")

    # --- IBM Quantum バックエンド選択 ---
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
    backend = service.least_busy(
        simulator=False,
        operational=True,
        min_num_qubits=2,
    )
    print(f"使用バックエンド: {backend.name} (qubits={backend.num_qubits}, "
          f"待ち行列={backend.status().pending_jobs})")

    # --- 量子特徴マップ（シミュレーターと同じパラメータ）---
    feature_map = zz_feature_map(feature_dimension=2, reps=1, alpha=1.0)

    # バックエンド向けにトランスパイル
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled_fm = pm.run(feature_map)
    print(f"トランスパイル後の深さ: {transpiled_fm.depth()}")

    # --- 量子カーネル構築（SamplerV2 を使用）---
    # shots=1024 に抑えてノイズ耐性を保ちつつショット数上限を回避
    # max_circuits_per_job で1ジョブあたりの回路数を制限（上限 10,000,000 shots 対策）
    #   1ジョブの shots = max_circuits_per_job × shots_per_circuit × 2（ComputeUncompute は回路を2倍使用）
    #   200 × 1024 × 2 = 409,600 shots/job → 上限内
    sampler = SamplerV2(mode=backend)
    fidelity = ComputeUncompute(sampler=sampler, pass_manager=pm)
    qkernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity,
        max_circuits_per_job=200,   # 1ジョブあたりの回路ペア数を制限
    )

    # --- カーネル行列計算 ---
    print("訓練カーネル行列を計算中（実機）... ※複数ジョブに分割して送信")
    K_train = qkernel.evaluate(X_train)
    print("テストカーネル行列を計算中（実機）... ※複数ジョブに分割して送信")
    K_test = qkernel.evaluate(X_test, X_train)

    # 実機カーネル行列を保存
    np.save(os.path.join(RESULTS_DIR, "kernel_train_real.npy"), K_train)
    np.save(os.path.join(RESULTS_DIR, "kernel_test_real.npy"), K_test)

    # --- SVM 学習・予測（C=10 はシミュレーターの最適値を流用）---
    svc = SVC(kernel="precomputed", C=10.0)
    svc.fit(K_train, y_train)
    y_pred = svc.predict(K_test)
    acc = accuracy_score(y_test, y_pred)

    # --- 結果比較 ---
    print("\n" + "=" * 45)
    print(f"  バックエンド      : {backend.name}")
    print(f"  [実機]  Accuracy : {acc:.4f} ({int(acc * len(y_test))}/{len(y_test)})")
    if sim_accuracy is not None:
        print(f"  [シミュレーター] : {sim_accuracy:.4f}")
        diff = acc - sim_accuracy
        print(f"  差分             : {diff:+.4f} （ノイズによる劣化）")
    print("=" * 45)

    return {
        "accuracy": acc,
        "backend": backend.name,
        "y_test": y_test,
        "y_pred": y_pred,
        "K_train": K_train,
        "K_test": K_test,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python qsvm_real_device.py <IBM_QUANTUM_API_KEY>")
        sys.exit(1)

    api_key = sys.argv[1]

    # シミュレーター結果が保存済みであれば読み込んで比較
    sim_acc = None
    sim_kernel_path = os.path.join(RESULTS_DIR, "kernel_train_sim.npy")
    if os.path.exists(sim_kernel_path):
        y_test      = np.load(os.path.join(RESULTS_DIR, "y_test.npy"))
        K_train_sim = np.load(os.path.join(RESULTS_DIR, "kernel_train_sim.npy"))
        K_test_sim  = np.load(os.path.join(RESULTS_DIR, "kernel_test_sim.npy"))
        y_train     = np.load(os.path.join(RESULTS_DIR, "y_train.npy"))
        svc_sim = SVC(kernel="precomputed", C=10.0)
        svc_sim.fit(K_train_sim, y_train)
        sim_acc = accuracy_score(y_test, svc_sim.predict(K_test_sim))

    run_qsvm_real_device(api_key=api_key, sim_accuracy=sim_acc)
