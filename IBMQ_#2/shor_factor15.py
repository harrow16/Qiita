"""
shor_factor15.py
================
IBM Q を使って 15 を素因数分解する — Shor のアルゴリズム実装
対象: N=15, a=7 (互いに素かつ適切な周期を持つ底)

量子位相推定 (QPE) + 量子フーリエ逆変換 (IQFT) を用いて
f(x) = 7^x mod 15 の周期 r を求め、素因数を導出する。

依存ライブラリ:
    pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib pylatexenc
"""

from __future__ import annotations

import math
import sys
import warnings
from fractions import Fraction

# Windows での日本語出力を UTF-8 に統一
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
import matplotlib
matplotlib.use("Agg")          # GUI なし環境でも動作
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. ユーティリティ
# ─────────────────────────────────────────────────────────────

def gcd(a: int, b: int) -> int:
    """ユークリッドの互除法"""
    while b:
        a, b = b, a % b
    return a


def mod_exp(base: int, exp: int, mod: int) -> int:
    """高速べき乗剰余"""
    return pow(base, exp, mod)


def continued_fraction_period(phase: float, n_qubits: int, N: int) -> int | None:
    """
    連分数展開で位相 phase から周期 r を推定する。
    r < N を満たす最初の分母を返す。
    """
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator
    return r if r < N else None


# ─────────────────────────────────────────────────────────────
# 2. 制御 U ゲート: |y⟩ → |a*y mod N⟩
#    N=15, a=7 の場合の専用実装（4 量子ビット演算レジスタ）
# ─────────────────────────────────────────────────────────────

def c_amod15(a: int, power: int) -> QuantumCircuit:
    """
    制御ユニタリ U^(2^power): |y⟩ → |(a^(2^power)) * y mod 15⟩
    4 量子ビット (target) + 1 制御ビット の計 5 量子ビット回路を返す。

    a=7, N=15 の乗算順列:
        7^1  mod 15 = 7   置換: 1→7→4→13→1, 2→14→8→11→2
        7^2  mod 15 = 4   置換: 1→4→1, 2→8→2, 7→13→7
        7^4  mod 15 = 1   恒等
    各ステップは SWAP と X の組合せで表現可能。
    """
    # a mod 15 を power 乗
    U_val = mod_exp(a, power, 15)

    # 置換マッピング (0–14, ただし作業レジスタは 0–15 を表現)
    # |0⟩ は |0*U mod 15⟩=|0⟩ のまま
    perm = {i: (U_val * i) % 15 for i in range(16)}
    # perm[15] は不定だが通常到達しない (|15⟩ は使わない)
    perm[15] = 15

    # --- 4 qubit target + 1 control の QuantumCircuit ---
    qc = QuantumCircuit(5)   # q[0]=control, q[1..4]=target (LSB=q[1])
    ctrl = 0

    # 置換を実装: ビット反転と SWAP の組合せで表現
    # U_val ∈ {1,2,4,7,8,11,13,14} のみ有効
    # 下記は各 U_val に対して予め手計算したゲート列

    def apply_perm(qc: QuantumCircuit, u: int) -> None:
        """target: q[1..4], |y⟩ → |u*y mod 15⟩ を CSWAP/CX で実装"""
        t = [1, 2, 3, 4]   # target indices in qc

        if u == 1:
            pass  # 恒等

        elif u == 2:
            # 1→2→4→8→1  (2-cycle群の直積 + 3→6→12→9→3)
            # 実用上 |1⟩-|2⟩-|4⟩-|8⟩ のサイクル: LSB から表現
            # |0001⟩→|0010⟩→|0100⟩→|1000⟩→|0001⟩
            qc.cswap(ctrl, t[0], t[3])  # swap bit0,bit3
            qc.cswap(ctrl, t[0], t[2])  # swap bit0,bit2
            qc.cswap(ctrl, t[0], t[1])  # swap bit0,bit1

        elif u == 4:
            # 1→4→1, 2→8→2, 3→12→3 など
            qc.cswap(ctrl, t[0], t[2])
            qc.cswap(ctrl, t[1], t[3])

        elif u == 7:
            # 7の置換サイクル: 1→7→4→13→1, 2→14→8→11→2
            qc.cswap(ctrl, t[1], t[3])
            qc.cswap(ctrl, t[0], t[2])
            qc.cswap(ctrl, t[0], t[3])
            qc.cx(ctrl, t[0])
            qc.cx(ctrl, t[1])
            qc.cx(ctrl, t[2])
            qc.cx(ctrl, t[3])

        elif u == 8:
            qc.cswap(ctrl, t[0], t[3])
            qc.cswap(ctrl, t[1], t[3])
            qc.cswap(ctrl, t[2], t[3])

        elif u == 11:
            qc.cx(ctrl, t[0])
            qc.cx(ctrl, t[1])
            qc.cx(ctrl, t[2])
            qc.cx(ctrl, t[3])
            qc.cswap(ctrl, t[0], t[3])
            qc.cswap(ctrl, t[1], t[3])
            qc.cswap(ctrl, t[2], t[3])

        elif u == 13:
            qc.cswap(ctrl, t[0], t[3])
            qc.cx(ctrl, t[0])
            qc.cx(ctrl, t[1])
            qc.cx(ctrl, t[2])
            qc.cx(ctrl, t[3])

        elif u == 14:
            qc.cx(ctrl, t[0])
            qc.cx(ctrl, t[1])
            qc.cx(ctrl, t[2])
            qc.cx(ctrl, t[3])

    apply_perm(qc, U_val)
    return qc


# ─────────────────────────────────────────────────────────────
# 3. 量子位相推定 (QPE) 回路の構築
# ─────────────────────────────────────────────────────────────

def build_qpe_circuit(n_count: int = 8, a: int = 7) -> QuantumCircuit:
    """
    量子位相推定回路を構築する。

    Parameters
    ----------
    n_count : int
        測定用レジスタのビット数 (位相精度に影響)。デフォルト 8。
    a : int
        底 (gcd(a, 15) = 1 を満たす整数)。デフォルト 7。

    Returns
    -------
    QuantumCircuit
        QPE 回路 (測定付き)
    """
    # レジスタ定義
    qr_count  = QuantumRegister(n_count, name="count")   # 位相推定レジスタ
    qr_target = QuantumRegister(4,       name="target")  # 作業レジスタ (|1⟩ に初期化)
    cr        = ClassicalRegister(n_count, name="meas")

    qc = QuantumCircuit(qr_count, qr_target, cr)

    # --- 初期化 ---
    # 位相推定レジスタ: 全ビットをアダマールで重ね合わせ
    qc.h(qr_count)
    # 作業レジスタ: |0001⟩ = |1⟩ (固有ベクトルの近似)
    qc.x(qr_target[0])

    qc.barrier()

    # --- 制御 U^(2^k) の適用 ---
    for k in range(n_count):
        power = 2 ** k
        # c_amod15 は 5 qubit 回路 (ctrl=q[0], target=q[1..4])
        U_gate = c_amod15(a, power).to_gate(label=f"U^{power}")
        # 制御ビットを count[k] に割り当て
        ctrl_qubit  = [qr_count[k]]
        target_bits = list(qr_target)
        cu = U_gate.control(0)          # すでに ctrl が組み込まれているので
        # 注: c_amod15 の q[0] が制御ビット
        # append でカスタムゲートを挿入
        qc.append(U_gate, [qr_count[k]] + list(qr_target))

    qc.barrier()

    # --- 量子フーリエ逆変換 (IQFT) ---
    iqft = QFT(n_count, inverse=True, do_swaps=True).decompose()
    qc.append(iqft, qr_count)

    qc.barrier()

    # --- 測定 ---
    qc.measure(qr_count, cr)

    return qc


# ─────────────────────────────────────────────────────────────
# 4. 測定結果から素因数を導出
# ─────────────────────────────────────────────────────────────

def extract_factors(counts: dict, n_count: int, a: int = 7, N: int = 15) -> list[tuple[int, int]]:
    """
    測定カウントから周期 r を推定し、素因数を返す。

    Parameters
    ----------
    counts   : 測定結果の辞書 {"bitstring": count}
    n_count  : 位相推定レジスタのビット数
    a, N     : アルゴリズムのパラメータ

    Returns
    -------
    list of (p, q) : 素因数のペアのリスト
    """
    results = []
    total   = sum(counts.values())

    for bitstring, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        # 測定値を整数に変換
        measured_int = int(bitstring, 2)
        # 位相 φ = measured_int / 2^n_count
        phase = measured_int / (2 ** n_count)

        # 連分数展開で周期 r を推定
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator

        # r が偶数かつ a^(r/2) ≢ -1 (mod N) の場合のみ有効
        if r % 2 == 0:
            x = mod_exp(a, r // 2, N)
            if x != N - 1:
                p = gcd(x - 1, N)
                q = gcd(x + 1, N)
                if 1 < p < N and 1 < q < N:
                    results.append((p, q))
                    print(f"  測定値={measured_int:4d}  位相={phase:.4f}  r={r}  "
                          f"gcd({a}^{r//2}-1,{N})={p}  gcd({a}^{r//2}+1,{N})={q}  "
                          f"-> {N} = {p} x {q}  (確率 {cnt/total:.2%})")

    return list(set(results))  # 重複除去


# ─────────────────────────────────────────────────────────────
# 5. シミュレーター実行
# ─────────────────────────────────────────────────────────────

def run_simulator(n_count: int = 8, shots: int = 2048, a: int = 7) -> dict:
    """
    Aer シミュレーターで QPE 回路を実行する。

    Parameters
    ----------
    n_count : 位相推定レジスタのビット数
    shots   : 試行回数
    a       : 底

    Returns
    -------
    counts : 測定結果辞書
    """
    print("\n" + "=" * 60)
    print("  [シミュレーター] N=15, a={}, n_count={}".format(a, n_count))
    print("=" * 60)

    qc = build_qpe_circuit(n_count=n_count, a=a)

    print(f"\n回路サイズ: {qc.num_qubits} 量子ビット, "
          f"深さ={qc.depth()}, ゲート数={qc.size()}")

    backend  = AerSimulator()
    t_qc     = transpile(qc, backend, optimization_level=1)
    job      = backend.run(t_qc, shots=shots)
    result   = job.result()
    counts   = result.get_counts()

    print(f"\n測定回数: {shots} shots")
    print("\n--- 素因数導出 ---")
    factors = extract_factors(counts, n_count, a=a)

    if factors:
        p, q = factors[0]
        print(f"\n[OK] 素因数分解成功: {15} = {p} x {q}")
    else:
        print("\n[NG] 素因数が見つかりませんでした。shots を増やして再試行してください。")

    # ヒストグラム保存
    _save_histogram(counts, filename="sim_histogram.png", title="Simulator Result (N=15, a=7)")

    return counts


# ─────────────────────────────────────────────────────────────
# 6. 実機 (IBM Quantum) 実行
# ─────────────────────────────────────────────────────────────

def run_real_device(
    ibm_token: str,
    backend_name: str | None = None,
    n_count: int = 4,
    shots: int = 1024,
    a: int = 7,
) -> dict | None:
    """
    IBM Quantum 実機で QPE 回路を実行する。

    Parameters
    ----------
    ibm_token    : IBM Quantum API トークン (https://quantum.ibm.com/ で取得)
    backend_name : 使用するバックエンド名。None の場合は最小待ち時間のものを自動選択。
    n_count      : 位相推定レジスタのビット数 (実機では 4 程度が現実的)
    shots        : 試行回数
    a            : 底

    Returns
    -------
    counts または None (エラー時)
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
        from qiskit_ibm_runtime import SamplerOptions
    except ImportError:
        print("[ERROR] qiskit-ibm-runtime が見つかりません。"
              "  pip install qiskit-ibm-runtime  を実行してください。")
        return None

    print("\n" + "=" * 60)
    print("  [実機] N=15, a={}, n_count={}".format(a, n_count))
    print("=" * 60)

    # IBM Quantum サービスに接続
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=ibm_token)

    # バックエンド選択 (qiskit-ibm-runtime 0.20+ の新 API)
    if backend_name is None:
        # n_count + 4 量子ビット以上を持つ実機バックエンドを選択
        backends = service.backends(
            simulator=False,
            operational=True,
            min_num_qubits=n_count + 4,
        )
        if not backends:
            print("[ERROR] 利用可能な実機バックエンドが見つかりません。")
            return None
        # 保留ジョブが最少のバックエンドを選択
        backend = min(backends, key=lambda b: b.status().pending_jobs)
    else:
        backend = service.backend(backend_name)

    print(f"\n使用バックエンド: {backend.name}")
    print(f"量子ビット数  : {backend.num_qubits}")
    print(f"待機ジョブ数  : {backend.status().pending_jobs}")

    qc = build_qpe_circuit(n_count=n_count, a=a)

    # トランスパイル (実機のネイティブゲートに変換)
    t_qc = transpile(
        qc,
        backend=backend,
        optimization_level=3,
        seed_transpiler=42,
    )
    print(f"\nトランスパイル後: 深さ={t_qc.depth()}, ゲート数={t_qc.size()}")

    # SamplerV2 で実行
    sampler = Sampler(backend)
    job     = sampler.run([t_qc], shots=shots)

    print(f"\nジョブ送信完了: job_id = {job.job_id()}")
    print("結果を待機中... (数分かかる場合があります)")

    result = job.result()
    # SamplerV2 の結果形式
    pub_result = result[0]
    counts_raw = pub_result.data.meas.get_counts()

    print(f"\n測定回数: {shots} shots")
    print("\n--- 素因数導出 ---")
    factors = extract_factors(counts_raw, n_count, a=a)

    if factors:
        p, q = factors[0]
        print(f"\n[OK] 素因数分解成功: {15} = {p} x {q}")
    else:
        print("\n[NG] 素因数が見つかりませんでした。")

    _save_histogram(counts_raw, filename="real_histogram.png", title=f"Real Device Result ({backend.name})")

    return counts_raw


# ─────────────────────────────────────────────────────────────
# 7. 補助: ヒストグラム保存
# ─────────────────────────────────────────────────────────────

def _save_histogram(counts: dict, filename: str, title: str) -> None:
    """上位 20 ビット列のヒストグラムを PNG に保存する"""
    import matplotlib.font_manager as fm

    # 日本語フォントを自動選択 (なければ英語フォールバック)
    jp_candidates = ["MS Gothic", "Yu Gothic", "Meiryo", "IPAGothic",
                     "Noto Sans CJK JP", "TakaoGothic"]
    jp_font = None
    for fname in jp_candidates:
        try:
            fp = fm.findfont(fm.FontProperties(family=fname), fallback_to_default=False)
            if fname.lower() in fp.lower():
                jp_font = fm.FontProperties(family=fname)
                break
        except Exception:
            pass

    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:20]
    labels  = [f"{int(k,2):03d}\n({k})" for k, _ in sorted_counts]
    heights = [v for _, v in sorted_counts]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(labels)), heights, color="#3b82d4", edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Measured Value (decimal / binary)",
                  fontproperties=jp_font)
    ax.set_ylabel("Count", fontproperties=jp_font)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\nHistogram saved: {filename}")


# ─────────────────────────────────────────────────────────────
# 8. 回路図の保存
# ─────────────────────────────────────────────────────────────

def save_circuit_diagram(n_count: int = 4, a: int = 7) -> None:
    """回路図を PNG に保存する (pylatexenc 必須)"""
    qc = build_qpe_circuit(n_count=n_count, a=a)
    try:
        fig = qc.draw("mpl", fold=-1, scale=0.6)
        fig.savefig("circuit_diagram.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("回路図保存: circuit_diagram.png")
    except Exception as e:
        print(f"回路図の保存に失敗しました: {e}")


# ─────────────────────────────────────────────────────────────
# 9. エントリポイント
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="IBM Q で 15 を素因数分解する (Shor のアルゴリズム)"
    )
    parser.add_argument("--mode",    choices=["sim", "real", "both"], default="sim",
                        help="実行モード: sim=シミュレーター, real=実機, both=両方 (デフォルト: sim)")
    parser.add_argument("--token",   type=str, default=None,
                        help="IBM Quantum API トークン (--mode real/both の場合必須)")
    parser.add_argument("--backend", type=str, default=None,
                        help="使用するバックエンド名 (省略時は自動選択)")
    parser.add_argument("--n_count", type=int, default=8,
                        help="位相推定レジスタのビット数 (デフォルト: 8)")
    parser.add_argument("--shots",   type=int, default=2048,
                        help="試行回数 (デフォルト: 2048)")
    parser.add_argument("--a",       type=int, default=7,
                        help="底 a: gcd(a, 15)=1 を満たす整数 (デフォルト: 7)")
    parser.add_argument("--diagram", action="store_true",
                        help="回路図を PNG に保存する")
    args = parser.parse_args()

    # 入力検証
    assert 1 < args.a < 15, "a は 2 ≤ a ≤ 14 の範囲で指定してください"
    assert gcd(args.a, 15) == 1, f"gcd({args.a}, 15) ≠ 1 です。別の底を選んでください"

    if args.diagram:
        save_circuit_diagram(n_count=min(args.n_count, 4), a=args.a)

    if args.mode in ("sim", "both"):
        run_simulator(n_count=args.n_count, shots=args.shots, a=args.a)

    if args.mode in ("real", "both"):
        if args.token is None:
            print("[ERROR] 実機モードには --token が必要です。")
        else:
            # 実機では n_count を 4 に制限 (デコヒーレンス対策)
            run_real_device(
                ibm_token=args.token,
                backend_name=args.backend,
                n_count=min(args.n_count, 4),
                shots=args.shots,
                a=args.a,
            )
