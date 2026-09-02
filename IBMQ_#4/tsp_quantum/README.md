# 🔬 量子コンピュータで巡回セールスマン問題を解く (QAOA × IBM Quantum)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-purple)](https://qiskit.org/)
[![IBM Quantum](https://img.shields.io/badge/IBM%20Quantum-Platform-black)](https://quantum.ibm.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**巡回セールスマン問題 (TSP)** を量子近似最適化アルゴリズム **(QAOA)** で解き、IBM Quantum Platform の実機で実行するサンプルプロジェクトです。

---

## 📁 ディレクトリ構成

```
tsp_quantum/
├── src/
│   ├── tsp_qaoa.py      # メインプログラム (QUBO定式化 + QAOA実行)
│   ├── ibmq_setup.py    # IBM Quantum 認証・バックエンド確認
│   └── visualize.py     # 量子回路・ルートの可視化
├── results/             # 実行結果の JSON・画像
├── requirements.txt
└── README.md
```

---

## 🚀 セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. IBM Quantum API Key の取得

[IBM Quantum Platform](https://quantum.ibm.com/) でアカウントを作成し、API Key を取得してください。

実行時に `--api-key` 引数で直接渡します（ファイルへの保存は不要）。

---

## ▶️ 実行方法

### シミュレーターで実行 (IBM Quantum 不要・デフォルト)

```bash
cd tsp_quantum
python src/tsp_qaoa.py
# または明示的に
python src/tsp_qaoa.py --mode simulator
```

### IBM Quantum 実機で実行

待ちジョブ数が最少のバックエンドを**自動選択**します。`--api-key` に IBM Quantum の API Key を渡してください。

```bash
# 実機で実行 (デフォルト設定: reps=2, shots=1024)
python src/tsp_qaoa.py --mode ibmq --api-key <YOUR_API_KEY>

# オプションをカスタマイズして実行
python src/tsp_qaoa.py --mode ibmq --api-key <YOUR_API_KEY> --reps 1 --shots 2048 --penalty 15
```

### コマンドライン引数一覧

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--mode` | `simulator` | `simulator` または `ibmq` |
| `--api-key` | `None` | IBM Quantum API Key (ibmq モード時に指定) |
| `--reps` | `2` | QAOA の層数 (大きいほど精度↑、実行時間↑) |
| `--shots` | `1024` | 測定ショット数 (ibmq モード時のみ有効) |
| `--penalty` | `10.0` | QUBO 制約違反のペナルティ係数 |
| `--min-qubits` | `16` | 実機選択時に必要な最低量子ビット数 |

---

## 🧠 アルゴリズムの概要

| ステップ | 内容 |
|---------|------|
| ① | 都市間の距離行列を生成 |
| ② | TSP を **QUBO** (0-1 二値最適化) に変換 |
| ③ | QUBO をイジング模型 (ZZ 相互作用) に変換 |
| ④ | **QAOA** 回路をパラメータ付き量子ゲートで構築 |
| ⑤ | COBYLA オプティマイザでパラメータを最適化 |
| ⑥ | 測定結果からルートを復元 |

---

## 📊 実行結果の例 (4都市・シミュレーター)

```
都市名: ['東京', '名古屋', '大阪', '京都']

[1] 古典ソルバー (ベースライン)
  最適ルート : 東京 → 名古屋 → 大阪 → 京都 → 東京
  目的関数値 : 4.0000

[2] QAOA シミュレーター (reps=2)
  最適ルート : 東京 → 名古屋 → 大阪 → 京都 → 東京
  目的関数値 : 4.0000
```

---

## 📜 ライセンス

MIT License — 自由に改変・再配布できます。
