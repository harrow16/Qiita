# IBM Bob + 量子コンピュータ #7 — QNN で手書き数字 0/1 分類

IBM Bob + 量子コンピュータシリーズ **第7弾**のサンプルコードです。  
量子ニューラルネットワーク（QNN）を使って、手書き数字データセットの「0」と「1」を分類します。

> 📄 関連 Qiita 記事：*(公開後に URL を追記)*

---

## 概要

| 項目 | 内容 |
|---|---|
| データセット | sklearn 内蔵 Digits（8×8px）※ MNIST と同じ処理の流れ |
| 前処理 | PCA で 4 次元に圧縮 → [0, π] にスケーリング |
| 量子回路 | ZZFeatureMap (reps=1) + RealAmplitudes (reps=2)、4 量子ビット |
| 観測量 | SparsePauliOp("IIIZ")（qubit 0 の Z 期待値） |
| 最適化 | COBYLA (maxiter=200)、シミュレータ上で実行 |
| 実行環境 | Python 3.11 / Windows 11 |
| テスト精度 | **66.7%**（重みパラメータ 12 個の極小モデル） |

---

## ファイル構成

```
IBMQ_#7/
├── mnist_qnn.py            # メインプログラム
├── mnist_qnn_result.png    # 出力グラフ（損失曲線・PCA散布図・混同行列）
├── mnist_qnn_samples.png   # 出力グラフ（正解・誤分類サンプル画像）
└── README.md               # このファイル
```

---

## 実行結果

### mnist_qnn_result.png
| グラフ | 内容 |
|---|---|
| (a) Training Loss Curve | 学習損失の推移。iter 40 付近で収束 |
| (b) PCA Space | テストデータの PCA 散布図。青 = Digit 0、赤 = Digit 1、× = 誤分類 |
| (c) Confusion Matrix | 混同行列。Test Accuracy = 66.7% |

### mnist_qnn_samples.png
- **上段**：正解サンプル（T:正解ラベル / P:予測ラベル）
- **下段**：誤分類サンプル（赤文字で表示）

---

## 環境構築

### 必要パッケージのインストール

```bash
pip install qiskit qiskit-aer qiskit-algorithms qiskit-machine-learning scikit-learn matplotlib
```

### 動作確認済みバージョン

| パッケージ | バージョン |
|---|---|
| Python | 3.11 |
| qiskit | 2.5.2 |
| qiskit-algorithms | 0.4.0 |
| qiskit-machine-learning | 0.9.1 |
| qiskit-aer | 0.17.2 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |

---

## 実行方法

```bash
python mnist_qnn.py
```

実行時間の目安：**1〜3 分程度**（maxiter=200 ですが通常 80 反復前後で早期収束します）

実行後、以下のファイルが生成されます：
- `mnist_qnn_result.png`
- `mnist_qnn_samples.png`

> **Windows の場合**：文字化けが起きる場合は以下を先に実行してください。
> ```powershell
> $env:PYTHONUTF8 = "1"
> python mnist_qnn.py
> ```

> **Google Colab の場合**：`matplotlib.use("Agg")` の行をコメントアウトするとグラフがセル内にインライン表示されます。

---

## 実機（IBM Quantum）で動かす場合

[IBM Quantum Platform](https://quantum.cloud.ibm.com/) の無料アカウント（Open プラン）で実機実行できます。

```python
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

service = QiskitRuntimeService(channel="ibm_quantum_platform", token="YOUR_API_KEY")
backend = service.least_busy(simulator=False, operational=True)
estimator = Estimator(mode=backend)  # コード中の StatevectorEstimator をこれに差し替えるだけ
```

---

## シリーズ一覧

| 弾 | テーマ |
|---|---|
| #1〜#6 | *(別リポジトリ参照)* |
| **#7（本リポジトリ）** | **機械学習モデル編 Part2 — QNN で手書き数字分類** |

---

## 参考

- [Qiskit Machine Learning ドキュメント](https://qiskit-community.github.io/qiskit-machine-learning/)
- [IBM Quantum Platform](https://quantum.cloud.ibm.com/)
- [IBM Bob](https://bob.ibm.com/)
- Havlíček et al., "Supervised learning with quantum-enhanced feature spaces", *Nature* 567, 209–212 (2019)
- Schuld & Petruccione, *Machine Learning with Quantum Computers*, Springer (2021)
