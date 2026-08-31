# QSVM with IBM Quantum — iris 2クラス分類

Python + Qiskit で **量子カーネル SVM（QSVM）** を実装し、IBM Quantum のシミュレーターと実機の両方で動作させるサンプルコードです。

## 概要

| 項目 | 内容 |
|------|------|
| モデル | QSVM（量子カーネル SVM） |
| 量子特徴マップ | ZZFeatureMap（2 qubit, reps=1, alpha=1.0） |
| カーネル計算 | FidelityQuantumKernel + ComputeUncompute |
| データセット | scikit-learn iris（setosa vs versicolor、2特徴量） |
| 実行環境 | シミュレーター（StatevectorSampler）＋ IBM Quantum 実機 |
| シミュレーター精度 | **Accuracy 0.90（18/20）** |

## ファイル構成

```
.
├── requirements.txt        # 依存パッケージ
├── setup_ibmq.py           # IBM Quantum 接続確認（バックエンド一覧表示）
├── data_prep.py            # iris データ前処理
├── qsvm_simulator.py       # シミュレーターで QSVM 実行
├── qsvm_real_device.py     # IBM Quantum 実機で QSVM 実行
├── visualize.py            # 結果の可視化
└── results/                # 実行後に生成されるファイル
    ├── circuit_simulator.png   # 量子回路図
    ├── kernel_heatmap.png      # カーネル行列ヒートマップ
    └── decision_boundary.png   # 決定境界
```

## セットアップ

### 必要環境
- Python 3.11+
- IBM Quantum Platform アカウント（無料）: https://quantum.ibm.com
- `qiskit-ibm-runtime` 0.49 以降では `channel="ibm_quantum_platform"` を使用（旧 `"ibm_quantum"` は廃止）

### インストール

```bash
pip install -r requirements.txt
```

## 実行手順

### 1. 接続確認

```bash
python setup_ibmq.py YOUR_API_KEY
```

利用可能な実機バックエンドと待ち行列数が表示されます。

### 2. シミュレーターで実行

```bash
python qsvm_simulator.py
```

`results/` に量子回路図・カーネル行列・データが保存されます。

### 3. 実機で実行

```bash
python qsvm_real_device.py YOUR_API_KEY
```

`least_busy()` で待ち行列が最小のバックエンドを自動選択します。
シミュレーターより正確度が低下することがあります（量子ノイズの影響）。

> **注意（Open Plan 無料枠をお使いの場合）**
> Open Plan には月間の量子コンピュータ利用時間に上限があります。上限に達した場合は翌月初にリセットされます。
> また、実機では 1ジョブあたりのショット数に上限（10,000,000）があるため、`max_circuits_per_job=200` を設定してジョブを自動分割しています。

### 4. 結果の可視化

```bash
python visualize.py
```

`results/kernel_heatmap.png`（カーネル行列）と `results/decision_boundary.png`（決定境界）が生成されます。

## 依存パッケージ

```
qiskit >= 2.0
qiskit-machine-learning >= 0.9
qiskit-ibm-runtime >= 0.49
qiskit-aer >= 0.17
scikit-learn >= 1.9
numpy >= 2.0
matplotlib >= 3.11
scipy >= 1.4
```

## ライセンス

MIT
