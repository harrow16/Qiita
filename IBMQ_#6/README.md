# 株式ポートフォリオ最適化 — 量子コンピューター版（QAOA / VQE）

LightGBM で予測した株価収益率をもとに、**QAOA / VQE（量子アルゴリズム）** で最適なポートフォリオを選出します。
データサイエンティスト協会主催のセミナー（元リポジトリ）で実施した **IBM Bob + IBM CPLEX（MIP）によるポートフォリオ最適化**を、**Qiskit + IBM Quantum に置き換えた**実装例です。

| | リンク |
|---|---|
| 🔗 元リポジトリ（CPLEX版）| [stock-portfolio-optimization](https://github.com/harrow16/stock-portfolio-optimization) |
| 📝 技術解説記事（Qiita）| [株価予測 × 量子コンピューター（QAOA）でポートフォリオ最適化してみた]((https://github.com/harrow16/Qiita/tree/main/IBMQ_%236) |

---

## 概要

### スコープ

本プロジェクトは「**株価予測（LightGBM）が完了した後の、最適化部分の量子化**」を扱います。
予測結果 CSV は `data/` に同梱済みのため、クローン後すぐに量子最適化を実行できます。

| フェーズ | 担当 | 状態 |
|---|---|---|
| 株価時系列データ収集 | 別プロジェクト (`20260420`) | 実施済み |
| LightGBM による収益率予測 | 別プロジェクト (`20260420`) | 実施済み・CSV 同梱 |
| **QUBO 変換** | **本プロジェクト** | ← ここから |
| **QAOA / VQE で最適化** | **本プロジェクト** | |
| **最適ポートフォリオ出力** | **本プロジェクト** | |

### CPLEX との違い

| 比較項目 | 旧版（CPLEX） | 本プロジェクト（量子）|
|---|---|---|
| 最適化ソルバー | IBM CPLEX OPL | Qiskit QAOA / VQE |
| 問題形式 | 混合整数計画（MIP）| QUBO（二次制約なし二値最適化）|
| 制約の扱い | `subject to { ... }` で陽に記述 | ペナルティ項として目的関数に組み込む |
| 購入単位 | 複数単元（整数変数）| 1単元のみ（バイナリ変数）|
| 必要ライブラリ | IBM CPLEX Studio（有料）| Qiskit（OSS）|

---

## ファイル構成

```
.
├── run_quantum_optimization.py   # メイン実行スクリプト（エントリポイント）
├── optimization/
│   └── quantum_portfolio.py      # QUBO 構築 + QAOA/VQE 最適化モジュール
├── data/
│   ├── prediction_results.csv    # 株価予測結果（LightGBMの出力）
│   └── README.md                 # データ仕様
├── results/                      # 最適化結果の出力先（自動生成）
├── requirements.txt              # 依存パッケージ
└── README.md                     # このファイル
```

---

## セットアップ

### 動作確認済み環境

- Python 3.11
- Windows 10 / 11（macOS / Linux でも動作可）

### インストール

```bash
pip install -r requirements.txt
```

または個別に：

```bash
pip install qiskit qiskit-optimization qiskit-algorithms qiskit-aer
```

---

## 使い方

### ステップ 1: リポジトリをクローンする

```bash
git clone https://github.com/YOUR_USERNAME/IBMQ_6.git
cd IBMQ_6
pip install -r requirements.txt
```

`data/prediction_results.csv`（LightGBM の予測結果）はリポジトリに同梱済みです。
追加のデータ準備は不要で、すぐに量子最適化を実行できます。

> 予測結果を更新したい場合は `data/README.md` を参照してください。

### ステップ 2: 量子最適化を実行する

```bash
# デフォルト（QAOA・シミュレータ・上位12銘柄）
python run_quantum_optimization.py

# VQE を使用
python run_quantum_optimization.py --algorithm vqe

# 候補銘柄を絞って高速化（量子ビット削減）
python run_quantum_optimization.py --top-n 8 --reps 1

# QAOA の層数を増やして精度向上
python run_quantum_optimization.py --reps 3

# 予測から再実行
python run_quantum_optimization.py --repredict

# IBM 実機で実行（API キーが必要）
python run_quantum_optimization.py --device --api-key YOUR_IBM_QUANTUM_API_KEY
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--algorithm` | `qaoa` | `qaoa` または `vqe` |
| `--reps` | `2` | QAOA の層数 $p$（多いほど精度向上・時間増加）|
| `--top-n` | `12` | 候補銘柄数（量子ビット数に直結）|
| `--repredict` | — | 既存の予測結果を破棄して再予測 |
| `--device` | — | IBM 実機を使用（省略時はシミュレータ）|
| `--api-key` | — | IBM Quantum API キー（`--device` 時に必要）|

### 量子ビット数と実行時間の目安

| `--top-n` | 量子ビット数 | `--reps 1` | `--reps 2` |
|---|---|---|---|
| 8 | 8 qubits | 約 10〜30 秒 | 約 30〜60 秒 |
| 12 | 12 qubits | 約 30〜120 秒 | 約 2〜5 分 |
| 16 | 16 qubits | 数分〜 | 十数分〜 |

---

## 実行結果例

```
========================================================================
     Portfolio Optimization Result - Quantum Computing Version
========================================================================

[ Algorithm ]
  QAOA (Quantum Approximate Optimization Algorithm)
  Simulator: Qiskit StatevectorSampler

[ Selected Stocks ]
  #1  Taiyo Yuden Co., Ltd.          (6976.T)  Technology      2.99%  416,000 JPY
  #2  Inpex Corporation              (1605.T)  Energy          2.77%  468,700 JPY
  #3  Tokyo Electric Power HD        (9501.T)  Utilities       2.73%   62,340 JPY
  #4  Sumitomo Chemical Co., Ltd.    (4005.T)  Basic Materials 2.71%   50,740 JPY

[ Investment Breakdown ]
  Total Cost              :      998,778 JPY  (Budget: 1,000,000 JPY)
  Budget Remaining        :        1,222 JPY

[ Expected Returns (5 Trading Days) ]
  Net Profit (after fee)  :       27,488 JPY
  Expected Return Rate    : 2.75 %

[ Constraint Satisfaction ]
  [OK] Budget constraint
  [OK] Stock limit (4 / 5 stocks)
  [OK] Sector diversity (all different sectors)
========================================================================
```

結果は `results/quantum_result_<algorithm>_<timestamp>.txt` に自動保存されます。

---

## 技術詳細

### QUBO 変換

各銘柄 $i$ に対してバイナリ変数 $x_i \in \{0, 1\}$ を定義し、以下の目的関数を最小化します：

$$H = -\sum_i \text{NetReturn}_i \cdot x_i + \lambda_1 (\text{予算超過})^2 + \lambda_2 (\text{銘柄数超過})^2 + \lambda_3 (\text{セクター重複})^2$$

制約をペナルティ項として展開する際、バイナリ変数の性質 $x_i^2 = x_i$ を利用して線形項・2次項に分解しています。

### ペナルティ係数

| 制約 | 係数（λ） | 説明 |
|---|---|---|
| 予算制約 | 100 | 金額をスケーリング（÷10,000）して調整 |
| 銘柄数制約 | 500 | 中程度のペナルティ |
| セクター制約 | 1000 | 厳守させるため最大 |

### 使用ライブラリ

| ライブラリ | 用途 |
|---|---|
| `qiskit` | 量子回路の基盤 |
| `qiskit-optimization` | QuadraticProgram・MinimumEigenOptimizer |
| `qiskit-algorithms` | QAOA・SamplingVQE・COBYLA |
| `qiskit-aer` | シミュレータ（StatevectorSampler）|

---

## 注意事項

- このプロジェクトは**教育・研究目的**のみです
- 量子アルゴリズムは**近似解**を返す場合があります
- 予測は過去データに基づくものであり、将来のパフォーマンスを保証しません
- **投資判断はご自身の責任で行ってください**

---

## ライセンス

MIT License — 詳細は [LICENSE](LICENSE) を参照してください。

---

**Made with Bob** 🤖
