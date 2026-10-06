"""
h2_realdevice.py
================
Qiskit Nature を使った水素分子（H2）の基底エネルギー計算 ── IBM Quantum 実機版

概要:
    1. .env から IBM Quantum API キーを読み込む（ファイルに直書きしない）
    2. 空いている実機を自動選択（最少待ちキュー）
    3. Estimator Primitive (Runtime) で VQE を実行
    4. 平衡距離付近（0.7 Å）の1点を計算して結果を保存

実行:
    python src/h2_realdevice.py

前提:
    プロジェクトルートに .env を用意してください（.env.example 参照）
    IBMQ_API_KEY=your_api_key_here
"""

import io
import os
import sys
import time
import warnings
from pathlib import Path

# Windows コンソールで日本語・特殊文字を正しく表示するための設定
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONUTF8"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# .env ファイルを読み込む（python-dotenv がない場合は手動パース）
def _load_dotenv(dotenv_path: Path):
    """軽量な .env パーサー（python-dotenv 不要）"""
    if not dotenv_path.exists():
        return
    with open(dotenv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            # 引用符の除去
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), value)


# プロジェクトルートの .env を読み込む
_PROJECT_ROOT = Path(__file__).parent.parent
_load_dotenv(_PROJECT_ROOT / ".env")

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────
# 出力先ディレクトリ
# ──────────────────────────────────────────
RESULTS_DIR = _PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────
# IBM Quantum 接続
# ──────────────────────────────────────────
def get_ibm_service():
    """
    環境変数 IBMQ_API_KEY から API キーを取得し、
    QiskitRuntimeService を返す。
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    api_key = os.environ.get("IBMQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "環境変数 IBMQ_API_KEY が設定されていません。\n"
            "プロジェクトルートの .env に IBMQ_API_KEY=<your_key> を記入してください。"
        )

    # qiskit-ibm-runtime 0.20以降は channel="ibm_quantum_platform"
    try:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
    except Exception:
        # フォールバック（旧バージョン向け）
        service = QiskitRuntimeService(channel="ibm_cloud", token=api_key)
    return service


def select_best_backend(service, min_qubits: int = 2):
    """
    利用可能な実機の中からキュー待ちが最も少ないものを自動選択する。

    Parameters
    ----------
    service : QiskitRuntimeService
    min_qubits : int
        最低必要量子ビット数（H2 UCCSD + ParityMapper 2量子ビット削減 → 2量子ビット）

    Returns
    -------
    backend : IBMBackend
    """
    print("利用可能な実機を検索中...")

    # シミュレータを除いた実機のみ取得
    backends = service.backends(
        simulator=False,
        operational=True,
        min_num_qubits=min_qubits,
    )

    if not backends:
        raise RuntimeError(
            f"利用可能な実機が見つかりません（最低 {min_qubits} 量子ビット必要）。"
        )

    # キュー待ち数でソートして最小を選択
    def _pending_jobs(b):
        try:
            status = b.status()
            return status.pending_jobs
        except Exception:
            return 9999

    backends_with_queue = [(b, _pending_jobs(b)) for b in backends]
    backends_with_queue.sort(key=lambda x: x[1])

    best, pending = backends_with_queue[0]
    print(f"\n利用可能な実機一覧（キュー順）:")
    for b, p in backends_with_queue[:5]:
        marker = " ← 選択" if b.name == best.name else ""
        print(f"  {b.name:30s}  待ちジョブ数: {p:4d}{marker}")

    print(f"\n選択した実機: {best.name}  (待ちジョブ数: {pending})")
    return best


# ──────────────────────────────────────────
# H2 ハミルトニアン係数（STO-3G 解析値）
# シミュレーション版と同じ実装（PySCF 不要・Windows 対応）
# ──────────────────────────────────────────
_R_TABLE  = [0.50, 0.60, 0.70, 0.735, 0.75, 0.80, 0.90, 1.00, 1.20, 1.40, 1.60, 2.00, 2.50]
_G0_VALS  = [-0.8121,-0.8953,-0.9432,-0.9551,-0.9597,-0.9786,-1.0178,-1.0581,-1.1279,-1.1603,-1.1743,-1.1867,-1.1875]
_G1_VALS  = [ 0.1720, 0.1695, 0.1811, 0.1843, 0.1854, 0.1899, 0.1979, 0.2009, 0.2085, 0.2103, 0.2088, 0.2014, 0.1897]
_G2_VALS  = [-0.2237,-0.2269,-0.2455,-0.2498,-0.2514,-0.2578,-0.2697,-0.2772,-0.2895,-0.2941,-0.2928,-0.2820,-0.2647]
_G3_VALS  = [ 0.1685, 0.1749, 0.1835, 0.1857, 0.1867, 0.1901, 0.1963, 0.2015, 0.2099, 0.2131, 0.2126, 0.2072, 0.1993]
_G4_VALS  = [ 0.0454, 0.0598, 0.0697, 0.0719, 0.0726, 0.0749, 0.0784, 0.0805, 0.0833, 0.0834, 0.0816, 0.0764, 0.0700]
_G5_VALS  = [ 0.0454, 0.0598, 0.0697, 0.0719, 0.0726, 0.0749, 0.0784, 0.0805, 0.0833, 0.0834, 0.0816, 0.0764, 0.0700]


def get_h2_coefficients(distance_angstrom: float) -> dict:
    """核間距離 (Å) に対する STO-3G ハミルトニアン係数を補間して返す。"""
    r = np.array(_R_TABLE)
    def interp(vals):
        return float(np.interp(distance_angstrom, r, np.array(vals)))
    return {
        "g0": interp(_G0_VALS), "g1": interp(_G1_VALS), "g2": interp(_G2_VALS),
        "g3": interp(_G3_VALS), "g4": interp(_G4_VALS), "g5": interp(_G5_VALS),
    }


def build_hamiltonian(coeff: dict):
    """係数辞書から SparsePauliOp ハミルトニアンを構築して返す。"""
    from qiskit.quantum_info import SparsePauliOp
    return SparsePauliOp.from_list([
        ("II", coeff["g0"]), ("ZI", coeff["g1"]), ("IZ", coeff["g2"]),
        ("ZZ", coeff["g3"]), ("XX", coeff["g4"]), ("YY", coeff["g5"]),
    ])


def run_exact(coeff: dict) -> float:
    """厳密対角化で H2 基底エネルギーを返す。"""
    I2 = np.eye(2, dtype=complex)
    X  = np.array([[0,1],[1,0]], dtype=complex)
    Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z  = np.array([[1,0],[0,-1]], dtype=complex)
    g0,g1,g2,g3,g4,g5 = (coeff["g0"],coeff["g1"],coeff["g2"],
                          coeff["g3"],coeff["g4"],coeff["g5"])
    H = (g0*np.kron(I2,I2) + g1*np.kron(Z,I2) + g2*np.kron(I2,Z)
       + g3*np.kron(Z,Z)   + g4*np.kron(X,X)  + g5*np.kron(Y,Y))
    return float(np.linalg.eigvalsh(H)[0])


# ──────────────────────────────────────────
# VQE 実行（実機 Runtime Estimator 使用）
# ──────────────────────────────────────────
def run_vqe_on_real_device(backend, coeff: dict) -> float:
    """
    IBM Quantum 実機上で VQE を実行してエネルギーを返す。

    - アンザッツ: Ry-CNOT 型（2量子ビット・パラメータ5個）
    - Estimator: EstimatorV2（Open プラン対応・Session なし）
    - 最適化: COBYLA（ノイズ環境に強い勾配不要法）

    Note: IBM Quantum Open プランでは Session が使えないため、
          EstimatorV2 を backend 直接渡しで使用します。

    Returns
    -------
    energy : float  (Hartree)
    """
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2 as Estimator, EstimatorOptions
    from scipy.optimize import minimize

    hamiltonian = build_hamiltonian(coeff)

    # ── アンザッツ回路（Ry-CNOT 型）
    theta = ParameterVector("θ", 5)
    qc = QuantumCircuit(2)
    qc.x(0)                  # HF 初期状態（|01⟩）
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)
    qc.rz(theta[2], 0)
    qc.ry(theta[3], 1)
    qc.cx(0, 1)
    qc.ry(theta[4], 0)

    print(f"\nアンザッツ量子ビット数: {qc.num_qubits}")
    print(f"パラメータ数: {qc.num_parameters}")

    # ── バックエンド向けにトランスパイル
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    qc_t = pm.run(qc)
    ham_t = hamiltonian.apply_layout(qc_t.layout)

    # ── Estimator オプション（エラー緩和）
    # Open プランは resilience_level=1 まで対応
    options = EstimatorOptions()
    options.resilience_level = 1

    # Open プランは Session 不可 → backend を直接渡す
    estimator = Estimator(mode=backend, options=options)

    energies = []   # 収束ログ用

    def energy_fn(params):
        pub = (qc_t, ham_t, [params])
        result = estimator.run([pub]).result()
        e = float(result[0].data.evs[0])
        energies.append(e)
        if len(energies) % 5 == 0:
            print(f"  iter {len(energies):3d}  E = {e:.6f} Ha")
        return e

    print("VQE 実行中（実機）... キュー待ちを含むため時間がかかります")
    t0 = time.time()

    # COBYLA: ノイズに強い勾配不要の最適化法
    x0 = np.random.default_rng(0).uniform(-np.pi, np.pi, 5)
    opt = minimize(energy_fn, x0, method="COBYLA",
                   options={"maxiter": 50, "rhobeg": 0.5})

    elapsed = time.time() - t0
    energy = float(opt.fun)
    print(f"\n完了！  エネルギー = {energy:.6f} Hartree  (所要時間: {elapsed:.1f}s)")
    return energy


# ──────────────────────────────────────────
# 結果の保存
# ──────────────────────────────────────────
def save_results(distance: float, e_real: float, e_exact: float, backend_name: str):
    """テキスト・グラフで結果を保存する。"""
    # テキスト
    txt_path = RESULTS_DIR / "realdevice_result.txt"
    content = (
        f"=== H2 基底エネルギー計算（実機） ===\n"
        f"使用バックエンド : {backend_name}\n"
        f"核間距離         : {distance:.3f} Å\n"
        f"実機 VQE エネルギー : {e_real:.6f} Hartree\n"
        f"厳密対角化 (参照) : {e_exact:.6f} Hartree\n"
        f"差分              : {abs(e_real - e_exact):.2e} Hartree\n"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✓ テキスト保存: {txt_path}")

    # グラフ（棒グラフ比較）
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"実機 VQE\n({backend_name})", "厳密対角化\n(参照)"]
    values = [e_real, e_exact]
    colors = ["#1f77b4", "#ff7f0e"]
    bars = ax.bar(labels, values, color=colors, alpha=0.8, width=0.4)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.002,
            f"{v:.4f} Ha",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("エネルギー (Hartree)", fontsize=12)
    ax.set_title(f"H₂ 基底エネルギー比較 (核間距離={distance} Å)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path = RESULTS_DIR / "realdevice_result.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"✓ グラフ保存: {plot_path}")


# ──────────────────────────────────────────
# メイン
# ──────────────────────────────────────────
def main():
    print("=" * 60)
    print("  H2 基底エネルギー計算 ── IBM Quantum 実機版")
    print("  Ry-CNOT ansatz + COBYLA + EstimatorV2")
    print("  ハミルトニアン: STO-3G 解析係数（PySCF 不要）")
    print("=" * 60)

    # H2 の平衡核間距離（0.735 Å）
    DISTANCE = 0.735

    # IBM Quantum に接続
    service = get_ibm_service()

    # 空いている実機を自動選択
    backend = select_best_backend(service, min_qubits=2)

    # ハミルトニアン係数を取得
    print(f"\n核間距離 {DISTANCE} A でハミルトニアンを構築中...")
    coeff = get_h2_coefficients(DISTANCE)

    # 厳密解（ローカルで計算）
    print("厳密対角化で参照エネルギーを計算中...")
    e_exact = run_exact(coeff)
    print(f"厳密値: {e_exact:.6f} Hartree")

    # 実機で VQE
    e_real = run_vqe_on_real_device(backend, coeff)

    # 結果保存
    save_results(DISTANCE, e_real, e_exact, backend.name)

    print("\n完了!")


if __name__ == "__main__":
    main()
