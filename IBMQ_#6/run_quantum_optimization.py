"""
株価予測 + 量子コンピューターによるポートフォリオ最適化
自動実行スクリプト

このスクリプトは以下を自動的に実行します：
  1. 株価予測モデルの実行（LightGBM）
  2. 予測結果の保存
  3. 量子アルゴリズム（QAOA / VQE）によるポートフォリオ最適化
  4. 結果の表示と保存

CPLEX の代替として Qiskit Optimization + Qiskit Aer シミュレータを使用。
IBM 実機への切り替えは --device オプションで可能。
"""

import io
import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Windows 文字化け対策
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONUTF8"] = "1"


# ─────────────────────────── ユーティリティ ─────────────────────────── #

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
OPTIMIZATION_DIR = BASE_DIR / "optimization"
RESULTS_DIR = BASE_DIR / "results"

STOCK_PREDICTION_SCRIPT = MODELS_DIR / "stock_prediction.py"
PREDICTION_CSV = DATA_DIR / "prediction_results.csv"
RESULTS_DIR.mkdir(exist_ok=True)


def print_section(title: str):
    """セクション区切りを表示"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


# ─────────────────────────── Step 1: 株価予測 ─────────────────────────── #

def run_stock_prediction() -> bool:
    """株価予測モデル（LightGBM）を実行する"""
    print_section("ステップ1: 株価予測モデルの実行（LightGBM）")

    if not STOCK_PREDICTION_SCRIPT.exists():
        # 元の 20260420 プロジェクトのスクリプトを利用
        fallback = Path("../20260420/models/stock_prediction.py")
        if fallback.exists():
            print(f"  → 元プロジェクトのスクリプトを使用: {fallback}")
            script_path = fallback.resolve()
            cwd = script_path.parent
        else:
            print(f"エラー: {STOCK_PREDICTION_SCRIPT} が見つかりません")
            print("  hint: models/stock_prediction.py を配置してください")
            return False
    else:
        script_path = STOCK_PREDICTION_SCRIPT
        cwd = MODELS_DIR

    # すでに予測結果がある場合はスキップ可能
    if PREDICTION_CSV.exists():
        print(f"  既存の予測結果を使用: {PREDICTION_CSV}")
        print("  （再実行する場合は --repredict オプションを指定）")
        return True

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(cwd),
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            print("エラー: 株価予測モデルの実行に失敗しました")
            return False

        print("[OK] 株価予測モデルの実行が完了しました")
        return True

    except Exception as exc:
        print(f"エラー: {exc}")
        return False


# ─────────────────────────── Step 2: 量子最適化 ─────────────────────────── #

def run_quantum_optimization(
    algorithm: str = "qaoa",
    top_n: int = 12,
    qaoa_reps: int = 2,
    use_real_device: bool = False,
    api_key: str = None,
) -> bool:
    """量子ポートフォリオ最適化を実行する"""
    print_section("ステップ2: 量子ポートフォリオ最適化の実行")

    if not PREDICTION_CSV.exists():
        print(f"エラー: 予測結果 CSV が見つかりません: {PREDICTION_CSV}")
        return False

    # quantum_portfolio モジュールをインポート
    sys.path.insert(0, str(OPTIMIZATION_DIR))
    try:
        from quantum_portfolio import QuantumPortfolioOptimizer
    except ImportError as exc:
        print(f"エラー: quantum_portfolio のインポートに失敗しました: {exc}")
        print("  hint: pip install qiskit qiskit-optimization qiskit-algorithms qiskit-aer")
        return False

    optimizer = QuantumPortfolioOptimizer(
        predictions_csv=str(PREDICTION_CSV),
        algorithm=algorithm,
        top_n=top_n,
        qaoa_reps=qaoa_reps,
        use_real_device=use_real_device,
        api_key=api_key,
    )

    try:
        report = optimizer.run()
        print(report)
        _save_results(report, algorithm)
        return True
    except Exception as exc:
        import traceback
        print(f"エラー: 量子最適化の実行に失敗しました")
        traceback.print_exc()
        return False


def _save_results(report: str, algorithm: str):
    """最適化結果をファイルに保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"quantum_result_{algorithm}_{timestamp}.txt"
    try:
        result_file.write_text(report, encoding="utf-8")
        print(f"\n結果を保存しました: {result_file}")
    except Exception as exc:
        print(f"警告: 結果の保存に失敗しました: {exc}")


# ─────────────────────────── CLI ─────────────────────────── #

def parse_args():
    parser = argparse.ArgumentParser(
        description="量子コンピューター（QAOA/VQE）を使ったポートフォリオ最適化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト（QAOA、シミュレータ、上位12銘柄）
  python run_quantum_optimization.py

  # VQE を使用
  python run_quantum_optimization.py --algorithm vqe

  # QAOA の層数を増やす（精度向上、時間増加）
  python run_quantum_optimization.py --reps 3

  # 候補銘柄数を減らして高速化
  python run_quantum_optimization.py --top-n 8

  # 予測から再実行
  python run_quantum_optimization.py --repredict

  # IBM 実機を使用（API キーが必要）
  python run_quantum_optimization.py --device --api-key YOUR_KEY
""",
    )
    parser.add_argument(
        "--algorithm", choices=["qaoa", "vqe"], default="qaoa",
        help="量子アルゴリズム（デフォルト: qaoa）"
    )
    parser.add_argument(
        "--reps", type=int, default=2, metavar="N",
        help="QAOA の層数 p（デフォルト: 2）"
    )
    parser.add_argument(
        "--top-n", type=int, default=12, metavar="N",
        help="候補銘柄数（量子ビット削減用、デフォルト: 12）"
    )
    parser.add_argument(
        "--repredict", action="store_true",
        help="既存の予測結果を無視して株価予測を再実行"
    )
    parser.add_argument(
        "--device", action="store_true",
        help="IBM 実機を使用（デフォルト: シミュレータ）"
    )
    parser.add_argument(
        "--api-key", metavar="KEY",
        help="IBM Quantum API キー（--device 使用時に必要）"
    )
    return parser.parse_args()


# ─────────────────────────── メイン ─────────────────────────── #

def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("  株式ポートフォリオ最適化システム（量子コンピューター版）")
    print("=" * 70)
    print(f"\n実行開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"アルゴリズム : {args.algorithm.upper()}")
    print(f"候補銘柄数  : {args.top_n} 銘柄")
    print(f"QAOA reps   : {args.reps}")
    print(f"実行環境    : {'IBM 実機' if args.device else 'シミュレータ（Qiskit Aer）'}")

    # --repredict が指定されている場合は既存結果を削除
    if args.repredict and PREDICTION_CSV.exists():
        PREDICTION_CSV.unlink()
        print("\n既存の予測結果を削除しました（再予測モード）")

    # Step 1: 株価予測
    if not run_stock_prediction():
        print("\n[ERROR] 株価予測モデルの実行に失敗しました")
        return 1

    # Step 2: 量子最適化
    if not run_quantum_optimization(
        algorithm=args.algorithm,
        top_n=args.top_n,
        qaoa_reps=args.reps,
        use_real_device=args.device,
        api_key=args.api_key,
    ):
        print("\n[ERROR] 量子最適化の実行に失敗しました")
        return 1

    # 完了
    print_section("すべての処理が完了しました")
    print("[OK] 株価予測モデルの実行（LightGBM）")
    print(f"[OK] 量子ポートフォリオ最適化（{args.algorithm.upper()}）")
    print("[OK] 結果の保存")
    print(f"\n実行完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
