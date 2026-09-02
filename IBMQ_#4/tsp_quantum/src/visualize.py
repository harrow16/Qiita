"""
量子回路の可視化スクリプト

QAOA 回路の構造を画像として出力し、
Qiita 記事や GitHub README に使用できるようにする。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 不要
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import to_ising
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from tsp_qaoa import make_distance_matrix, build_tsp_qubo, DEMO_CITIES, CITY_NAMES


def visualize_qaoa_circuit(reps: int = 1, output_path: str = "results/qaoa_circuit.png") -> None:
    """
    QAOA 回路の構造を PNG 画像として保存する。

    Args:
        reps: QAOA の層数 (p)
        output_path: 出力ファイルパス
    """
    # 小さい例 (3都市) で回路を可視化
    cities_3 = DEMO_CITIES[:3]
    names_3 = CITY_NAMES[:3]
    dist = make_distance_matrix(cities_3)
    qp = build_tsp_qubo(dist, penalty=10.0)

    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)
    ising_op, offset = to_ising(qubo)

    from qiskit.circuit.library import QAOAAnsatz
    circuit = QAOAAnsatz(cost_operator=ising_op, reps=reps)
    circuit = circuit.decompose()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig = circuit.draw(output="mpl", fold=-1, style={"backgroundcolor": "#FFFFFF"})
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"量子回路の画像を保存しました: {output_path}")


def visualize_route(route_names: list[str], cities: list[tuple], output_path: str = "results/tsp_route.png") -> None:
    """
    TSP の最適ルートを地図風に可視化する。

    Args:
        route_names: 訪問順の都市名リスト (最後に出発地が再登場)
        cities: 都市の座標リスト
        output_path: 出力ファイルパス
    """
    names = CITY_NAMES
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#f0f4ff")
    fig.patch.set_facecolor("#ffffff")

    # 全都市をプロット
    for i, (x, y) in enumerate(cities):
        ax.scatter(x, y, s=200, zorder=5, color="#3b82d4")
        ax.text(x + 0.03, y + 0.03, names[i], fontsize=12, fontweight="bold")

    # ルートを矢印で描画
    route_coords = []
    for name in route_names:
        if name in names:
            idx = names.index(name)
            route_coords.append(cities[idx])

    for k in range(len(route_coords) - 1):
        x0, y0 = route_coords[k]
        x1, y1 = route_coords[k + 1]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="#e05c3a", lw=2),
        )

    ax.set_xlim(-0.3, 1.4)
    ax.set_ylim(-0.3, 1.4)
    ax.set_title("TSP 最適ルート (QAOA)", fontsize=14, pad=15)
    ax.set_xlabel("X 座標")
    ax.set_ylabel("Y 座標")
    ax.grid(True, alpha=0.4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"ルート図を保存しました: {output_path}")


if __name__ == "__main__":
    print("量子回路の可視化...")
    visualize_qaoa_circuit(reps=1, output_path="results/qaoa_circuit.png")

    print("\n最適ルートの可視化...")
    demo_route = [CITY_NAMES[0], CITY_NAMES[1], CITY_NAMES[2], CITY_NAMES[3], CITY_NAMES[0]]
    visualize_route(demo_route, DEMO_CITIES, output_path="results/tsp_route.png")
