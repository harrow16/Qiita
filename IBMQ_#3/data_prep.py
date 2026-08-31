"""
data_prep.py
iris データセットを QSVM 用に前処理するモジュール。
- クラス 0（setosa）と 1（versicolor）の 2クラスのみ使用
- PCA で 2次元に削減（量子回路のビット数削減）
- MinMaxScaler で [-π, π] にスケーリング
- 80/20 でtrain/test 分割
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_iris_data(
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    前処理済みの iris データを返す。

    Returns
    -------
    X_train, X_test, y_train, y_test
        X は shape (n_samples, 2)、値域は [-π, π]
        y はラベル 0 または 1
    """
    iris = load_iris()
    # クラス 0・1 のみ抽出
    mask = iris.target < 2
    X, y = iris.data[mask], iris.target[mask]

    # PCA で 2次元に削減
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    # [-π, π] にスケーリング
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_scaled = scaler.fit_transform(X_pca)

    # 80/20 で分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_iris_data()
    print(f"訓練データ: {X_train.shape}, テストデータ: {X_test.shape}")
    print(f"特徴量の範囲: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"ラベル分布 (train): {np.bincount(y_train)}")
    print(f"ラベル分布 (test) : {np.bincount(y_test)}")
