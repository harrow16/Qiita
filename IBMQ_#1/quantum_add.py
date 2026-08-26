"""
quantum_add.py
量子回路を使って 1 + 1 = 2 を計算するサンプルプログラム

【回路の設計】
  2ビット加算: A=1, B=1 を加算すると S=0, Carry=1 → つまり "10" (2進) = 2 (10進)

  量子ビットの割り当て:
    q[0] : A (入力 1)
    q[1] : B (入力 1)
    q[2] : Sum (和ビット)
    q[3] : Carry (桁上がりビット)

  加算の論理:
    Sum   = A XOR B       → CNOT(A→Sum), CNOT(B→Sum)
    Carry = A AND B       → Toffoli(A, B → Carry)

【実行モード】
  --mode sim   : ローカルシミュレーター（デフォルト）
  --mode real  : IBM Quantum 実機（API トークン必要）
"""

import orjson_patch  # orjson DLL ブロック回避（AppLocker 環境用）
import argparse
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def build_adder_circuit(a: int, b: int) -> QuantumCircuit:
    """
    1ビット加算回路を構築する。
    a, b: 0 または 1
    """
    qr = QuantumRegister(4, name='q')   # q[0]=A, q[1]=B, q[2]=Sum, q[3]=Carry
    cr = ClassicalRegister(2, name='c') # c[0]=Sum, c[1]=Carry
    qc = QuantumCircuit(qr, cr)

    # --- 入力の準備 ---
    if a == 1:
        qc.x(qr[0])   # A = 1
    if b == 1:
        qc.x(qr[1])   # B = 1

    qc.barrier()

    # --- 加算ロジック ---
    # Sum = A XOR B
    qc.cx(qr[0], qr[2])   # CNOT: A → Sum
    qc.cx(qr[1], qr[2])   # CNOT: B → Sum

    # Carry = A AND B
    qc.ccx(qr[0], qr[1], qr[3])  # Toffoli: A, B → Carry

    qc.barrier()

    # --- 測定 ---
    qc.measure(qr[2], cr[0])  # Sum   → c[0]
    qc.measure(qr[3], cr[1])  # Carry → c[1]

    return qc


def run_simulation(qc: QuantumCircuit, shots: int = 1024) -> dict:
    """ローカル Aer シミュレーターで実行する。"""
    from qiskit_aer import AerSimulator
    simulator = AerSimulator()
    job = simulator.run(qc, shots=shots)
    result = job.result()
    return result.get_counts(qc)


def run_on_real_device(qc: QuantumCircuit, api_token: str, shots: int = 1024) -> dict:
    """IBM Quantum 実機で実行する。"""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    # IBM Quantum に接続
    print("IBM Quantum に接続中...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_token)

    # 利用可能な実機の中で最もキュー待ちが少ないバックエンドを自動選択
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=4)
    print(f"使用するバックエンド: {backend.name}")

    # 実機向けにトランスパイル（回路を実機のゲートセットに変換）
    print("回路をトランスパイル中...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)

    print(f"トランスパイル後の回路深さ: {isa_circuit.depth()}")

    # 実機で実行
    print(f"実機に送信中... (shots={shots})")
    sampler = Sampler(backend)
    job = sampler.run([isa_circuit], shots=shots)

    print(f"ジョブID: {job.job_id()}")
    print("結果を待機中...（実機はキュー待ちがあるため数分かかる場合があります）")

    result = job.result()

    # SamplerV2 の結果からカウントを取得
    counts_raw = result[0].data.c.get_counts()
    return counts_raw


def interpret_result(counts: dict, a: int, b: int):
    """測定結果を表示・解釈する。"""
    print(f"\n【測定結果】 {counts}")
    total = sum(counts.values())
    print(f"{'ビット列':^10} {'回数':^8} {'確率':^8}  解釈")
    print("-" * 45)
    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
        carry = int(bitstring[0])   # c[1] = Carry (上位ビット)
        s     = int(bitstring[1])   # c[0] = Sum   (下位ビット)
        value = carry * 2 + s
        prob  = count / total * 100
        mark  = " ← 正解" if value == a + b else ""
        print(f"  {bitstring:^10} {count:^8} {prob:>6.1f}%   {carry}{s}(2進) = {value}(10進){mark}")
    print()


def main():
    parser = argparse.ArgumentParser(description="量子加算回路: 1 + 1")
    parser.add_argument("--mode", choices=["sim", "real"], default="sim",
                        help="sim=シミュレーター / real=IBM Quantum 実機 (デフォルト: sim)")
    parser.add_argument("--token", type=str, default=None,
                        help="IBM Quantum API トークン (--mode real 時に必要)")
    parser.add_argument("--shots", type=int, default=1024,
                        help="測定回数 (デフォルト: 1024)")
    args = parser.parse_args()

    a, b = 1, 1
    print(f"=== 量子加算回路: {a} + {b} ===\n")

    qc = build_adder_circuit(a, b)

    # 回路図を表示
    print("【量子回路】")
    print(str(qc.draw(output='text', fold=-1)))

    if args.mode == "sim":
        print("\n--- ローカルシミュレーターで実行 ---")
        counts = run_simulation(qc, shots=args.shots)
    else:
        if not args.token:
            print("エラー: --mode real の場合は --token <API_TOKEN> を指定してください。")
            return
        print("\n--- IBM Quantum 実機で実行 ---")
        counts = run_on_real_device(qc, api_token=args.token, shots=args.shots)

    interpret_result(counts, a, b)
    print(f"結論: {a} + {b} = 2 ✓")


if __name__ == "__main__":
    main()
