"""
巡回セールスマン問題 (TSP) を QAOA で解く
IBM Quantum Platform 対応版

対象都市数: 3〜4都市（デモ用）
アルゴリズム: QAOA (Quantum Approximate Optimization Algorithm)

使い方:
  # ローカルシミュレーターで実行 (デフォルト)
  python tsp_qaoa.py

  # IBM Quantum 実機で実行 (待ちが最少のバックエンドを自動選択)
  python tsp_qaoa.py --mode ibmq --api-key <YOUR_API_KEY>

  # オプション指定例
  python tsp_qaoa.py --mode ibmq --api-key <YOUR_API_KEY> --reps 2 --shots 2048 --penalty 15
"""

import argparse
import numpy as np
import json
import os
from datetime import datetime

# --- Qiskit imports ---
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# IBM Quantum (Runtime)
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


# ============================================================
# 1. 問題データ定義
# ============================================================

def make_distance_matrix(cities: list[tuple[float, float]]) -> np.ndarray:
    """
    都市座標リストからユークリッド距離行列を生成する。

    Args:
        cities: [(x0,y0), (x1,y1), ...] 形式の座標リスト

    Returns:
        n×n の距離行列 (numpy.ndarray)
    """
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = cities[i][0] - cities[j][0]
                dy = cities[i][1] - cities[j][1]
                dist[i][j] = np.sqrt(dx ** 2 + dy ** 2)
    return dist


# デモ用: 4都市の座標 (x, y)
DEMO_CITIES = [
    (0.0, 0.0),   # 都市0: 東京
    (1.0, 0.0),   # 都市1: 名古屋
    (1.0, 1.0),   # 都市2: 大阪
    (0.0, 1.0),   # 都市3: 京都
]
CITY_NAMES = ["東京", "名古屋", "大阪", "京都"]


# ============================================================
# 2. QUBO 定式化
# ============================================================

def build_tsp_qubo(dist_matrix: np.ndarray, penalty: float = 10.0) -> QuadraticProgram:
    """
    TSP を QUBO (二値最適化問題) として定式化する。

    変数: x[i][p] = 都市 i を訪問順序 p 番目に訪れるとき 1, それ以外 0
    制約:
      - 各都市をちょうど1回訪れる
      - 各訪問順序にちょうど1都市だけ割り当てる

    Args:
        dist_matrix: n×n 距離行列
        penalty: 制約違反に対するペナルティ係数

    Returns:
        QuadraticProgram オブジェクト
    """
    n = len(dist_matrix)
    qp = QuadraticProgram(name="TSP")

    # --- 変数定義 ---
    for i in range(n):
        for p in range(n):
            qp.binary_var(name=f"x_{i}_{p}")

    # --- 目的関数: 総移動距離の最小化 ---
    linear = {}
    quadratic = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for p in range(n):
                q = (p + 1) % n
                key = (f"x_{i}_{p}", f"x_{j}_{q}")
                quadratic[key] = quadratic.get(key, 0.0) + dist_matrix[i][j]

    qp.minimize(linear=linear, quadratic=quadratic)

    # --- 制約1: 各都市をちょうど1回訪れる ---
    for i in range(n):
        constraint = {f"x_{i}_{p}": 1 for p in range(n)}
        qp.linear_constraint(
            linear=constraint,
            sense="==",
            rhs=1,
            name=f"city_{i}_once",
        )

    # --- 制約2: 各順序にちょうど1都市 ---
    for p in range(n):
        constraint = {f"x_{i}_{p}": 1 for i in range(n)}
        qp.linear_constraint(
            linear=constraint,
            sense="==",
            rhs=1,
            name=f"position_{p}_once",
        )

    return qp


# ============================================================
# 3. 古典ソルバー (ベースライン)
# ============================================================

def solve_classical(qp: QuadraticProgram) -> dict:
    """
    NumPy 固有値ソルバーで QUBO の厳密解を求める (ベースライン用・小規模向け)。

    QUBO ハミルトニアンを密行列として展開し、全固有値・固有ベクトルを NumPy で
    計算して最小固有値に対応する解ビット列を返す。総当たりに相当するため
    都市数が増えると指数的に遅くなるが、4都市程度では一瞬で厳密解が得られる。
    QAOA の解が正しいかどうかを検証するためのベースラインとして使用する。

    Args:
        qp: 解く QuadraticProgram (QUBO 変換前の問題)

    Returns:
        結果辞書 {"method", "fval", "x", "status"}
    """
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)

    solver = NumPyMinimumEigensolver()
    optimizer = MinimumEigenOptimizer(solver)
    result = optimizer.solve(qubo)

    return {
        "method": "Classical (NumPy)",
        "fval": result.fval,
        "x": result.x.tolist(),
        "status": str(result.status),
    }


# ============================================================
# 4. QAOA ソルバー (シミュレーター)
# ============================================================

def solve_qaoa_simulator(qp: QuadraticProgram, reps: int = 2) -> dict:
    """
    Qiskit の StatevectorSimulator を使った QAOA 実行。
    IBM Quantum 実機への接続前にローカルで動作確認するために使う。

    Args:
        qp: 解く QuadraticProgram
        reps: QAOA の層数 (p パラメータ)

    Returns:
        結果辞書
    """
    from qiskit_algorithms.minimum_eigensolvers import QAOA as QAOASolver
    from qiskit.primitives import Sampler as LocalSampler

    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)

    sampler = LocalSampler()
    optimizer = COBYLA(maxiter=300)
    qaoa = QAOASolver(sampler=sampler, optimizer=optimizer, reps=reps)

    eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = eigen_optimizer.solve(qubo)

    return {
        "method": f"QAOA Simulator (reps={reps})",
        "fval": result.fval,
        "x": result.x.tolist(),
        "status": str(result.status),
    }


# ============================================================
# 5. IBM Quantum バックエンド選択
# ============================================================

def select_least_busy_backend(min_qubits: int = 16, api_key: str | None = None) -> str:
    """
    IBM Quantum に接続し、稼働中かつ待ちジョブ数が最も少ない
    実機バックエンドを選んで名前を返す。

    Args:
        min_qubits: 必要な最低量子ビット数 (TSP 4都市 = 16ビット)
        api_key: IBM Quantum API Key。指定した場合は保存済み認証より優先される。

    Returns:
        選択されたバックエンド名
    """
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key) if api_key else QiskitRuntimeService()

    candidates = [
        b for b in service.backends(simulator=False, operational=True)
        if b.num_qubits >= min_qubits
    ]
    if not candidates:
        raise RuntimeError(
            f"稼働中の実機バックエンドが見つかりませんでした "
            f"(min_qubits={min_qubits})。"
        )

    # pending_jobs が最小のものを選ぶ
    best = min(candidates, key=lambda b: b.status().pending_jobs)

    print(f"\n利用可能な実機バックエンド ({len(candidates)} 件):")
    for b in sorted(candidates, key=lambda b: b.status().pending_jobs):
        status = b.status()
        print(
            f"  {'★' if b.name == best.name else ' '} "
            f"{b.name:30s}  qubits={b.num_qubits:3d}  "
            f"pending_jobs={status.pending_jobs:4d}"
        )
    print(f"\n→ 選択されたバックエンド: {best.name}")
    return best.name


# ============================================================
# 6. IBM Quantum 実機ソルバー
# ============================================================

class _TranspilingSampler:
    """
    qiskit_algorithms の QAOA が要求する Sampler インターフェースに準拠しつつ、
    実機投入前に generate_preset_pass_manager でトランスパイルを行うラッパー。
    """

    def __init__(self, backend, shots: int):
        from qiskit_ibm_runtime import SamplerOptions
        options = SamplerOptions()
        options.default_shots = shots
        self._inner = Sampler(backend, options=options)
        self._backend = backend

    def run(self, circuits, **kwargs):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pm = generate_preset_pass_manager(optimization_level=1, backend=self._backend)
        # circuits は QuantumCircuit のリスト、または (circuit, params, ...) の PUB タプルのリスト
        transpiled = []
        for item in circuits:
            if isinstance(item, tuple):
                # PUB 形式: 先頭要素だけトランスパイルして残りはそのまま
                transpiled.append((pm.run(item[0]),) + item[1:])
            else:
                transpiled.append(pm.run(item))
        return self._inner.run(transpiled, **kwargs)


def solve_qaoa_ibmq(
    qp: QuadraticProgram,
    reps: int = 1,
    shots: int = 1024,
    min_qubits: int = 16,
    api_key: str | None = None,
) -> dict:
    """
    IBM Quantum 実機で QAOA を実行する。
    待ちジョブ数が最少のバックエンドを自動選択する。

    Args:
        qp: 解く QuadraticProgram
        reps: QAOA の層数
        shots: 測定ショット数
        min_qubits: 必要な最低量子ビット数
        api_key: IBM Quantum API Key。指定した場合は保存済み認証より優先される。

    Returns:
        結果辞書
    """
    from qiskit_algorithms.minimum_eigensolvers import QAOA as QAOASolver

    backend_name = select_least_busy_backend(min_qubits=min_qubits, api_key=api_key)
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key) if api_key else QiskitRuntimeService()
    backend = service.backend(backend_name)

    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)

    sampler = _TranspilingSampler(backend, shots=shots)
    optimizer = COBYLA(maxiter=100)
    qaoa = QAOASolver(sampler=sampler, optimizer=optimizer, reps=reps)

    eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = eigen_optimizer.solve(qubo)

    return {
        "method": f"QAOA IBM Quantum ({backend_name}, reps={reps})",
        "fval": result.fval,
        "x": result.x.tolist(),
        "status": str(result.status),
        "backend": backend_name,
    }


# ============================================================
# 6. 結果の解釈
# ============================================================

def decode_result(x: list, n: int, city_names: list[str]) -> list[str]:
    """
    QUBO の解ビット列からルート順序を復元する。

    Args:
        x: バイナリ変数の解リスト
        n: 都市数
        city_names: 都市名リスト

    Returns:
        訪問順の都市名リスト
    """
    route_indices = [-1] * n
    for i in range(n):
        for p in range(n):
            idx = i * n + p
            if idx < len(x) and round(x[idx]) == 1:
                route_indices[p] = i

    route = []
    for p in range(n):
        if route_indices[p] >= 0:
            route.append(city_names[route_indices[p]])
        else:
            route.append("?")
    route.append(route[0])  # 出発地に戻る
    return route


def calc_total_distance(route_indices: list[int], dist_matrix: np.ndarray) -> float:
    """ルートの総移動距離を計算する。"""
    total = 0.0
    n = len(route_indices)
    for k in range(n):
        i = route_indices[k]
        j = route_indices[(k + 1) % n]
        total += dist_matrix[i][j]
    return total


# ============================================================
# 7. 引数パーサー
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="巡回セールスマン問題を QAOA で解く (IBM Quantum 対応版)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["simulator", "ibmq"],
        default="simulator",
        help="実行モード: simulator=ローカルシミュレーター / ibmq=IBM Quantum 実機",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="QAOA の層数 (p)。大きいほど精度が上がるが実行時間・ノイズも増える",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="量子ビット測定のショット数 (ibmq モード時のみ有効)",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=10.0,
        help="QUBO 制約違反に対するペナルティ係数",
    )
    parser.add_argument(
        "--min-qubits",
        type=int,
        default=16,
        dest="min_qubits",
        help="実機選択時に必要な最低量子ビット数",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        dest="api_key",
        help="IBM Quantum API Key (ibmq モード時に指定。省略時は保存済み認証を使用)",
    )
    return parser.parse_args()


# ============================================================
# 8. メイン実行
# ============================================================

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  巡回セールスマン問題 × QAOA (IBM Quantum)")
    print(f"  モード: {args.mode.upper()}")
    print("=" * 60)

    cities = DEMO_CITIES
    names = CITY_NAMES
    n = len(cities)

    dist = make_distance_matrix(cities)
    print(f"\n都市数: {n}")
    print(f"都市名: {names}")
    print(f"\n距離行列:\n{np.round(dist, 3)}")

    qp = build_tsp_qubo(dist, penalty=args.penalty)
    print(f"\nQUBO 変数数: {qp.get_num_vars()}")

    output: dict = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "mode": args.mode,
        "reps": args.reps,
        "shots": args.shots,
        "penalty": args.penalty,
        "cities": {names[i]: list(cities[i]) for i in range(n)},
    }

    # --- [1] 古典ソルバー (ベースライン) ---
    print("\n[1] 古典ソルバー (ベースライン) で計算中...")
    classical_result = solve_classical(qp)
    classical_route = decode_result(classical_result["x"], n, names)
    print(f"  最適ルート : {' → '.join(classical_route)}")
    print(f"  目的関数値 : {classical_result['fval']:.4f}")
    output["classical"] = {**classical_result, "route": classical_route}

    # --- [2] QAOA (モードに応じて切り替え) ---
    if args.mode == "simulator":
        print(f"\n[2] QAOA シミュレーター (reps={args.reps}) で計算中...")
        qaoa_result = solve_qaoa_simulator(qp, reps=args.reps)
        label = "qaoa_simulator"
    else:
        print(f"\n[2] QAOA IBM Quantum 実機 (reps={args.reps}, shots={args.shots}) で計算中...")
        qaoa_result = solve_qaoa_ibmq(qp, reps=args.reps, shots=args.shots, min_qubits=args.min_qubits, api_key=args.api_key)
        label = "qaoa_ibmq"

    qaoa_route = decode_result(qaoa_result["x"], n, names)
    print(f"  最適ルート : {' → '.join(qaoa_route)}")
    print(f"  目的関数値 : {qaoa_result['fval']:.4f}")
    output[label] = {**qaoa_result, "route": qaoa_route}

    # --- 結果を JSON 保存 ---
    os.makedirs("results", exist_ok=True)
    out_path = f"results/tsp_result_{output['timestamp']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存しました: {out_path}")


if __name__ == "__main__":
    main()
