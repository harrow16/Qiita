"""
setup_ibmq.py
IBM Quantum Platform への接続確認スクリプト。

使い方:
    python setup_ibmq.py <IBM_QUANTUM_API_KEY>
"""
import sys

from qiskit_ibm_runtime import QiskitRuntimeService


def list_backends(api_key: str) -> None:
    """利用可能なバックエンド一覧と待ち行列数を表示する。"""
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
    backends = service.backends(simulator=False, operational=True, min_num_qubits=2)

    print(f"\n利用可能な実機バックエンド ({len(backends)} 件):")
    print(f"{'名前':<30} {'Qubits':>6} {'待ち行列':>8}")
    print("-" * 48)
    for b in sorted(backends, key=lambda b: b.status().pending_jobs):
        status = b.status()
        print(f"{b.name:<30} {b.num_qubits:>6} {status.pending_jobs:>8}")

    least = service.least_busy(simulator=False, operational=True, min_num_qubits=2)
    print(f"\n最小待ち行列バックエンド: {least.name} (qubits={least.num_qubits})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python setup_ibmq.py <IBM_QUANTUM_API_KEY>")
        sys.exit(1)

    list_backends(sys.argv[1])
