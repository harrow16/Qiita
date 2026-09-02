"""
IBM Quantum Platform 認証セットアップスクリプト

初回のみ実行してください。
API Key は https://quantum.ibm.com/ で取得できます。
"""

from qiskit_ibm_runtime import QiskitRuntimeService


def setup_account(api_key: str) -> None:
    """
    IBM Quantum の API Key を保存する。
    一度実行すれば ~/.qiskit/qiskit-ibm.json に保存され、
    以降は QiskitRuntimeService() だけで認証される。

    Args:
        api_key: IBM Quantum Platform で発行した API Key
    """
    QiskitRuntimeService.save_account(
        channel="ibm_quantum",
        token=api_key,
        overwrite=True,
    )
    print("API Key の保存が完了しました。")
    print("以降は QiskitRuntimeService() で自動接続されます。")


def list_backends() -> None:
    """
    利用可能なバックエンド（実機・シミュレーター）を一覧表示する。
    """
    service = QiskitRuntimeService()
    backends = service.backends()
    print(f"\n利用可能なバックエンド ({len(backends)} 件):")
    for b in backends:
        status = b.status()
        print(f"  - {b.name:30s}  qubits={b.num_qubits:3d}  operational={status.operational}")


def check_connection() -> bool:
    """
    IBM Quantum への接続確認を行う。

    Returns:
        接続成功なら True
    """
    try:
        service = QiskitRuntimeService()
        _ = service.backends()
        print("接続成功: IBM Quantum Platform に接続できました。")
        return True
    except Exception as e:
        print(f"接続失敗: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使い方: python ibmq_setup.py <YOUR_API_KEY>")
        print("         python ibmq_setup.py check")
        print("         python ibmq_setup.py list")
        sys.exit(1)

    command = sys.argv[1]

    if command == "check":
        check_connection()
    elif command == "list":
        list_backends()
    else:
        # API Key として扱う
        setup_account(command)
        check_connection()
        list_backends()
