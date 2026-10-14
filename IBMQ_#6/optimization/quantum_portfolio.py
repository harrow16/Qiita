"""
量子コンピューターを使ったポートフォリオ最適化モジュール

IBM CPLEXの代わりに量子アルゴリズム（QAOA / VQE）を使用して
ポートフォリオ最適化問題を解く。

問題定式化:
  - 目的: 期待収益（手数料控除後）を最大化
  - 制約:
      1. 予算: 100万円以内（手数料込み）
      2. 銘柄数: 最大 MAX_STOCKS 銘柄
      3. 業界分散: 同一セクターから最大1銘柄
      4. 購入単位: 100株単位（各銘柄 1単元のみ選ぶ簡略版）

量子版では扱いやすさのため「各銘柄を買うか買わないか（0/1）」の
バイナリ選択問題に簡略化し、QUBOへ変換して量子回路で解く。
"""

import io
import os
import sys

# Windows 文字化け対策（直接実行時のみ再ラップする）
if sys.platform == "win32" and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONUTF8"] = "1"

import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from datetime import datetime

# Qiskit Optimization
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

# Qiskit Algorithms
from qiskit_algorithms import QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

# Qiskit Primitives（シミュレータ）
from qiskit.primitives import StatevectorSampler as Sampler

# Qiskit Optimization Algorithms
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# ─────────────────────────── 定数 ─────────────────────────── #
BUDGET = 1_000_000        # 予算 100万円
COMMISSION_RATE = 0.001   # 手数料率 0.1%
MAX_STOCKS = 5            # 最大銘柄数
LOT_SIZE = 100            # 購入単位（100株）
TOP_N = 12                # 量子ビット数を抑えるため上位 N 銘柄に絞る
RANDOM_SEED = 42


# ─────────────────────────── ユーティリティ ─────────────────────────── #

# 株データを辞書のリストとして扱う（pandasなし）
StockList = list  # List[dict]


def load_predictions(csv_path: str) -> StockList:
    """予測結果 CSV を読み込む（標準 csv モジュール使用）"""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "Ticker": row["Ticker"],
                "Current_Price": float(row["Current_Price"]),
                "Predicted_Return": float(row["Predicted_Return"]),
                "Sector": row["Sector"],
                "Name": row["Name"],
            })

    required_cols = {"Ticker", "Current_Price", "Predicted_Return", "Sector", "Name"}
    if rows and not required_cols.issubset(rows[0].keys()):
        raise ValueError(f"CSVに必要なカラムが不足しています")

    # 収益率降順でソート
    rows.sort(key=lambda r: r["Predicted_Return"], reverse=True)
    return rows


def filter_candidates(stocks: StockList, top_n: int = TOP_N) -> StockList:
    """
    量子ビット数を抑えるため候補銘柄を絞り込む。
    上位 top_n 銘柄かつ予算内（1単元購入可能）なものに限定。
    """
    affordable = [
        s for s in stocks
        if s["Current_Price"] * LOT_SIZE * (1 + COMMISSION_RATE) <= BUDGET
    ]
    result = affordable[:top_n]
    print(f"候補銘柄: {len(result)} 銘柄（上位 {top_n} 銘柄から予算内に絞り込み）")
    return result


# ─────────────────────────── QUBO 構築 ─────────────────────────── #

def build_quadratic_program(stocks: StockList) -> QuadraticProgram:
    """
    ポートフォリオ選択問題を QuadraticProgram として定式化する。

    各銘柄 i に対してバイナリ変数 x_i ∈ {0, 1} を定義:
      x_i = 1: 銘柄 i を 1単元（100株）購入する
      x_i = 0: 購入しない

    目的関数（最大化）:
      sum_i [ price_i * return_i * LOT_SIZE - price_i * LOT_SIZE * commission ] * x_i

    ※ qiskit-optimization は最小化問題として扱うため符号を反転。

    注意: QUBO変換のため不等式制約は使わず、違反時のペナルティを
          目的関数に組み込むアプローチを取る（QuadraticProgramToQubo
          は float係数の不等式制約を扱えないため）。

    制約はすべてペナルティ項として目的関数に追加:
      1. 予算制約（超過額のペナルティ）
      2. 銘柄数制約（超過数のペナルティ）
      3. セクター制約（同セクター複数選択のペナルティ）
    """
    SCALE = 10_000          # 円 → 万円スケール（係数を整数に丸める）
    PENALTY_BUDGET = 100.0  # 予算違反 1万円あたりのペナルティ
    PENALTY_SECTOR = 1000.0 # セクター制約違反のペナルティ
    PENALTY_NSTOCK = 500.0  # 銘柄数超過のペナルティ

    n = len(stocks)
    qp = QuadraticProgram(name="PortfolioOptimization")

    # ── バイナリ変数の宣言 ──
    for i in range(n):
        qp.binary_var(name=f"x{i}")

    # ── 目的関数の係数（最小化のため符号反転）──
    linear_obj: dict = {}
    quadratic_obj: dict = {}

    for i, s in enumerate(stocks):
        cost_i = s["Current_Price"] * LOT_SIZE
        net_return_i = cost_i * s["Predicted_Return"] - cost_i * COMMISSION_RATE
        linear_obj[f"x{i}"] = -net_return_i  # 最小化なので符号反転

    # ── ペナルティ1: 予算制約 ──
    # penalty = PENALTY_BUDGET * max(0, TotalCost - BUDGET)^2
    # ≈ PENALTY_BUDGET * [sum_i cost_i * x_i - BUDGET]^2  if > 0
    # 2次ペナルティ展開:
    #   (sum_i a_i x_i)^2 = sum_i a_i^2 x_i^2 + 2*sum_{i<j} a_i a_j x_i x_j
    #   x_i^2 = x_i  (binary)
    costs = [s["Current_Price"] * LOT_SIZE * (1 + COMMISSION_RATE) / SCALE for s in stocks]
    budget_scaled = BUDGET / SCALE

    for i in range(n):
        linear_obj[f"x{i}"] = linear_obj.get(f"x{i}", 0.0) + \
            PENALTY_BUDGET * (costs[i] ** 2 - 2 * budget_scaled * costs[i])
        for j in range(i + 1, n):
            key = (f"x{i}", f"x{j}")
            quadratic_obj[key] = quadratic_obj.get(key, 0.0) + \
                2 * PENALTY_BUDGET * costs[i] * costs[j]

    # ── ペナルティ2: 銘柄数制約（最大 MAX_STOCKS 銘柄）──
    # penalty = PENALTY_NSTOCK * max(0, sum_i x_i - MAX_STOCKS)^2
    # 2次展開: (sum x_i)^2 - 2*MAX_STOCKS * sum x_i
    for i in range(n):
        linear_obj[f"x{i}"] = linear_obj.get(f"x{i}", 0.0) + \
            PENALTY_NSTOCK * (1 - 2 * MAX_STOCKS)
        for j in range(i + 1, n):
            key = (f"x{i}", f"x{j}")
            quadratic_obj[key] = quadratic_obj.get(key, 0.0) + 2 * PENALTY_NSTOCK

    # ── ペナルティ3: セクター制約（各セクター最大1銘柄）──
    seen_sectors: dict[str, list] = {}
    for i, s in enumerate(stocks):
        seen_sectors.setdefault(s["Sector"], []).append(i)

    for sector, indices in seen_sectors.items():
        if len(indices) <= 1:
            continue
        # penalty = PENALTY_SECTOR * (sum_{i in sector} x_i - 1)^2 if > 1
        # 2次展開: sum x_i^2 - 2*sum x_i + 2*sum_{i<j} x_i x_j
        for i in indices:
            linear_obj[f"x{i}"] = linear_obj.get(f"x{i}", 0.0) + \
                PENALTY_SECTOR * (1 - 2)
        for idx_a, i in enumerate(indices):
            for j in indices[idx_a + 1:]:
                key = (f"x{i}", f"x{j}")
                quadratic_obj[key] = quadratic_obj.get(key, 0.0) + 2 * PENALTY_SECTOR

    qp.minimize(linear=linear_obj, quadratic=quadratic_obj)

    return qp


# ─────────────────────────── 量子最適化 ─────────────────────────── #

def run_qaoa(qp: QuadraticProgram, reps: int = 2) -> dict:
    """
    QAOA（Quantum Approximate Optimization Algorithm）で最適化を実行。

    Args:
        qp  : QuadraticProgram（ペナルティ込みの純粋な QUBO）
        reps: QAOA の繰り返し層数 p

    Returns:
        result dict with keys: x, fval, variables
    """
    algorithm_globals.random_seed = RANDOM_SEED

    n_qubits = qp.get_num_vars()
    print(f"量子ビット数: {n_qubits}")

    # Qiskit Primitives の Sampler を使用（シミュレータ）
    sampler = Sampler()

    optimizer = COBYLA(maxiter=500, rhobeg=1.0)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)

    # MinimumEigenOptimizer でラップして解く
    eigen_optimizer = MinimumEigenOptimizer(qaoa)

    print(f"QAOA を実行中（reps={reps}）... しばらくお待ちください")
    result = eigen_optimizer.solve(qp)

    return _parse_result(result, qp)


def run_vqe(qp: QuadraticProgram) -> dict:
    """
    VQE（Variational Quantum Eigensolver）+ SamplingVQE で最適化を実行。

    Args:
        qp: QuadraticProgram（ペナルティ込みの純粋な QUBO）

    Returns:
        result dict
    """
    from qiskit.circuit.library import RealAmplitudes

    algorithm_globals.random_seed = RANDOM_SEED

    n_qubits = qp.get_num_vars()
    print(f"量子ビット数: {n_qubits}")

    ansatz = RealAmplitudes(n_qubits, reps=2)
    optimizer = COBYLA(maxiter=500)

    sampler = Sampler()
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=optimizer)

    eigen_optimizer = MinimumEigenOptimizer(vqe)

    print("VQE を実行中... しばらくお待ちください")
    result = eigen_optimizer.solve(qp)

    return _parse_result(result, qp)


def _parse_result(result, qp: QuadraticProgram) -> dict:
    """OptimizationResult を辞書形式にパース"""
    variables = qp.variables
    x_values = result.x
    return {
        "x": x_values,
        "fval": result.fval,
        "status": str(result.status),
        "variables": {v.name: int(round(x_values[i])) for i, v in enumerate(variables)},
    }


# ─────────────────────────── 結果整形 ─────────────────────────── #

def format_results(opt_result: dict, stocks: StockList, algorithm: str = "QAOA") -> str:
    """最適化結果を読みやすい文字列に整形（元の CPLEX 出力に合わせた形式）"""

    selected = [
        stocks[int(k[1:])]
        for k, v in opt_result["variables"].items()
        if v == 1 and int(k[1:]) < len(stocks)
    ]

    total_investment = 0.0
    total_expected_profit = 0.0
    total_shares = 0
    lines = []

    lines.append("")
    lines.append("=" * 72)
    lines.append("     Portfolio Optimization Result - Quantum Computing Version     ")
    lines.append("=" * 72)
    lines.append("")
    lines.append("[ Algorithm ]")
    lines.append(f"  {algorithm.upper()} (Quantum Approximate Optimization Algorithm)")
    lines.append("  Simulator: Qiskit StatevectorSampler")
    lines.append("")
    lines.append("[ Selected Stocks ]")
    lines.append("-" * 72)
    lines.append("")

    for rank, row in enumerate(selected, start=1):
        cost_i = row["Current_Price"] * LOT_SIZE
        expected_profit = cost_i * row["Predicted_Return"]
        total_investment += cost_i
        total_expected_profit += expected_profit
        total_shares += LOT_SIZE

        lines.append(f"[ Stock #{rank} ]")
        lines.append(f"  Company Name      : {row['Name']}")
        lines.append(f"  Ticker Code       : {row['Ticker']}")
        lines.append(f"  Sector            : {row['Sector']}")
        lines.append(f"  Current Price     : {row['Current_Price']:,.0f} JPY")
        lines.append(f"  Units to Buy      : 1 unit")
        lines.append(f"  Shares to Buy     : {LOT_SIZE} shares")
        lines.append(f"  Investment Amount : {cost_i:,.0f} JPY")
        lines.append(f"  Expected Return   : {row['Predicted_Return']*100:.2f} %")
        lines.append(f"  Expected Profit   : {expected_profit:,.0f} JPY")
        lines.append("")

    commission = total_investment * COMMISSION_RATE
    net_return = total_expected_profit - commission

    lines.append("-" * 72)
    lines.append("")
    lines.append("[ Portfolio Summary ]")
    lines.append("-" * 72)
    lines.append(f"  Selected Stocks         : {len(selected)} stocks")
    lines.append(f"  Total Shares            : {total_shares} shares")
    lines.append("")
    lines.append("[ Investment Breakdown ]")
    lines.append(f"  Principal Investment    : {total_investment:>12,.0f} JPY")
    lines.append(f"  Trading Fee (0.1%)      : {commission:>12,.0f} JPY")
    lines.append("  " + "-" * 50)
    lines.append(f"  Total Cost              : {total_investment + commission:>12,.0f} JPY")
    lines.append(f"  Budget Remaining        : {BUDGET - total_investment - commission:>12,.0f} JPY")
    lines.append("")
    lines.append("[ Expected Returns (5 Trading Days) ]")
    lines.append(f"  Expected Total Profit   : {total_expected_profit:>12,.0f} JPY")
    lines.append(f"  Net Profit (after fee)  : {net_return:>12,.0f} JPY")
    if total_investment > 0:
        lines.append(f"  Expected Return Rate    : {net_return/total_investment*100:.2f} %")
    lines.append("")
    lines.append("[ Constraint Satisfaction ]")
    budget_ok = (total_investment + commission) <= BUDGET
    stocks_ok = len(selected) <= MAX_STOCKS
    sectors_used = [s["Sector"] for s in selected]
    sector_ok = len(sectors_used) == len(set(sectors_used))

    lines.append(f"  {'[OK]' if budget_ok else '[NG]'} Budget constraint (within {BUDGET:,} JPY)")
    lines.append(f"  {'[OK]' if stocks_ok else '[NG]'} Stock limit (max {MAX_STOCKS} stocks, selected {len(selected)})")
    lines.append(f"  {'[OK]' if sector_ok else '[NG]'} Sector diversity (each stock from different sector)")
    lines.append("")
    lines.append("[ Disclaimer ]")
    lines.append("  * This result is based on quantum computing + machine learning predictions")
    lines.append("  * Investment decisions should be made at your own risk")
    lines.append("  * Past performance does not guarantee future results")
    lines.append("")
    lines.append("=" * 72)
    lines.append("                 Quantum Optimization Completed                    ")
    lines.append("=" * 72)
    lines.append("")

    return "\n".join(lines)


# ─────────────────────────── メイン ─────────────────────────── #

class QuantumPortfolioOptimizer:
    """量子ポートフォリオ最適化クラス"""

    def __init__(
        self,
        predictions_csv: str,
        algorithm: str = "qaoa",
        top_n: int = TOP_N,
        qaoa_reps: int = 2,
        use_real_device: bool = False,
        api_key: str = None,
    ):
        """
        Args:
            predictions_csv : 株価予測結果 CSV のパス
            algorithm       : "qaoa" または "vqe"
            top_n           : 候補銘柄数（量子ビット削減のため）
            qaoa_reps       : QAOA の層数 p
            use_real_device : True の場合 IBM 実機を使用（Open プランでは Session 不可）
            api_key         : IBM Quantum API キー（実機使用時）
        """
        self.predictions_csv = predictions_csv
        self.algorithm = algorithm.lower()
        self.top_n = top_n
        self.qaoa_reps = qaoa_reps
        self.use_real_device = use_real_device
        self.api_key = api_key

    def run(self) -> str:
        """最適化パイプライン全体を実行し、結果文字列を返す"""
        print("=" * 60)
        print("量子ポートフォリオ最適化 開始")
        print(f"アルゴリズム: {self.algorithm.upper()}")
        print("=" * 60)

        # 1. 予測データ読み込み
        print("\n[1] 予測データを読み込み中...")
        stocks = load_predictions(self.predictions_csv)
        print(f"    全銘柄数: {len(stocks)}")

        # 2. 候補銘柄の絞り込み
        print("\n[2] 候補銘柄を絞り込み中...")
        candidates = filter_candidates(stocks, top_n=self.top_n)
        header = f"  {'#':>3}  {'Ticker':<10}  {'Sector':<25}  {'Return':>7}  Name"
        print(f"\n    候補銘柄一覧:")
        print(f"    {header}")
        print(f"    {'-'*75}")
        for i, s in enumerate(candidates):
            print(f"    {i:>3}  {s['Ticker']:<10}  {s['Sector']:<25}  {s['Predicted_Return']*100:>6.2f}%  {s['Name']}")

        # 3. QuadraticProgram の構築
        print("\n[3] 最適化問題（QUBO）を構築中...")
        qp = build_quadratic_program(candidates)
        print(f"    変数数（量子ビット数）: {qp.get_num_vars()}")
        print(f"    目的関数の項数（線形）: {len(qp.objective.linear.to_dict())}")
        print(f"    目的関数の項数（2次）: {len(qp.objective.quadratic.to_dict())}")

        # 4. 量子アルゴリズムで解く
        print(f"\n[4] 量子アルゴリズム ({self.algorithm.upper()}) を実行中...")
        if self.algorithm == "qaoa":
            opt_result = run_qaoa(qp, reps=self.qaoa_reps)
        elif self.algorithm == "vqe":
            opt_result = run_vqe(qp)
        else:
            raise ValueError(f"不明なアルゴリズム: {self.algorithm}. 'qaoa' または 'vqe' を指定してください")

        print(f"    最適化ステータス: {opt_result['status']}")
        print(f"    目的関数値: {opt_result['fval']:.2f}")

        # 5. 結果整形
        print("\n[5] 結果を整形中...")
        report = format_results(opt_result, candidates, algorithm=self.algorithm)

        print("=" * 60)
        print("量子ポートフォリオ最適化 完了")
        print("=" * 60)

        return report


def main():
    """単体テスト用エントリポイント"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "data", "prediction_results.csv")

    optimizer = QuantumPortfolioOptimizer(
        predictions_csv=csv_path,
        algorithm="qaoa",
        top_n=8,
        qaoa_reps=1,
    )
    report = optimizer.run()
    print(report)


if __name__ == "__main__":
    main()

# Made with Bob
