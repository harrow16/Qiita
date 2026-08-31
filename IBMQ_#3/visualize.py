"""
visualize.py
QSVM の結果を可視化するスクリプト。

生成物:
  results/kernel_heatmap.png    - シミュレーター vs 実機 のカーネル行列ヒートマップ
  results/decision_boundary.png - 2次元特徴空間上の決定境界
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI なし環境向け
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC

RESULTS_DIR = "results"


def _load_npy(filename: str) -> np.ndarray | None:
    path = os.path.join(RESULTS_DIR, filename)
    return np.load(path) if os.path.exists(path) else None


def plot_kernel_heatmap() -> None:
    """シミュレーター（と実機）のカーネル行列ヒートマップを描画・保存する。"""
    K_sim  = _load_npy("kernel_train_sim.npy")
    K_real = _load_npy("kernel_train_real.npy")
    y_train = _load_npy("y_train.npy")

    if K_sim is None:
        print("kernel_train_sim.npy が見つかりません。qsvm_simulator.py を先に実行してください。")
        return

    has_real = K_real is not None
    ncols = 2 if has_real else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # y_train でソートして可視化しやすくする
    if y_train is not None:
        order = np.argsort(y_train)
        K_sim = K_sim[np.ix_(order, order)]
        if has_real:
            K_real = K_real[np.ix_(order, order)]

    for ax, K, title in zip(
        axes,
        [K_sim] + ([K_real] if has_real else []),
        ["シミュレーター"] + (["実機"] if has_real else []),
    ):
        im = ax.imshow(K, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"量子カーネル行列 ({title})", fontsize=12)
        ax.set_xlabel("サンプルインデックス")
        ax.set_ylabel("サンプルインデックス")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "kernel_heatmap.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"カーネルヒートマップを保存: {out}")


def plot_decision_boundary() -> None:
    """2次元特徴空間上にデータ点と決定境界を描画・保存する。"""
    X_train = _load_npy("X_train.npy")
    X_test  = _load_npy("X_test.npy")
    y_train = _load_npy("y_train.npy")
    y_test  = _load_npy("y_test.npy")
    K_train = _load_npy("kernel_train_sim.npy")
    K_test  = _load_npy("kernel_test_sim.npy")

    if any(v is None for v in [X_train, X_test, y_train, y_test, K_train, K_test]):
        print("必要な .npy ファイルが見つかりません。qsvm_simulator.py を先に実行してください。")
        return

    # SVC を再学習（カーネル行列から）
    svc = SVC(kernel="precomputed", C=10.0)
    svc.fit(K_train, y_train)

    # メッシュグリッドで決定境界を計算
    # ※ 決定境界描画には全サンプルとのカーネルが必要なため、
    #   ここでは訓練・テストデータ点をプロットするにとどめ、
    #   SVC の decision_function を meshgrid 上で近似計算する
    x_min, x_max = X_train[:, 0].min() - 0.3, X_train[:, 0].max() + 0.3
    y_min, y_max = X_train[:, 1].min() - 0.3, X_train[:, 1].max() + 0.3
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # グリッド点と訓練データのカーネルをユークリッド距離の RBF で近似
    # （量子カーネルの近似表示 — 実際の量子回路は使わない）
    sigma = 1.0
    K_grid = np.exp(
        -np.sum((grid_points[:, None, :] - X_train[None, :, :]) ** 2, axis=2) / (2 * sigma ** 2)
    )
    Z = svc.decision_function(K_grid).reshape(xx.shape)

    cmap_light = ListedColormap(["#aec6e8", "#f7c6a3"])
    cmap_bold  = ListedColormap(["#2166ac", "#d6604d"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], alpha=0.35, colors=["#aec6e8", "#f7c6a3"])
    ax.contour(xx, yy, Z, levels=[0], colors="black", linewidths=1.2, linestyles="--")

    # 訓練データ
    scatter_train = ax.scatter(
        X_train[:, 0], X_train[:, 1],
        c=y_train, cmap=cmap_bold,
        edgecolors="k", s=60, alpha=0.8, label="訓練データ"
    )
    # テストデータ
    ax.scatter(
        X_test[:, 0], X_test[:, 1],
        c=y_test, cmap=cmap_bold,
        edgecolors="white", linewidths=1.5, s=120, marker="*", label="テストデータ"
    )

    ax.set_xlabel("特徴量 1（PCA 第1主成分）", fontsize=11)
    ax.set_ylabel("特徴量 2（PCA 第2主成分）", fontsize=11)
    ax.set_title("QSVM 決定境界（シミュレーター）\niris: setosa vs versicolor", fontsize=12)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "decision_boundary.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"決定境界を保存: {out}")


if __name__ == "__main__":
    # 日本語フォント設定（Windows）
    plt.rcParams["font.family"] = "MS Gothic"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_kernel_heatmap()
    plot_decision_boundary()
    print("可視化完了。results/ ディレクトリを確認してください。")
