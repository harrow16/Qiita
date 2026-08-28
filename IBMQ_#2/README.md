# shor-factor15-ibmq

IBM Q を使って **15 を素因数分解**する Shor のアルゴリズム実装です。  
Qiskit 1.x 対応。**シミュレーター**と **IBM Quantum 実機**の両方で実行できます。

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-6929c4)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 概要

**Shor のアルゴリズム**は、量子コンピュータを使って素因数分解を多項式時間で解くアルゴリズムです（P. W. Shor, 1994）。  
本実装では `N = 15`、底 `a = 7` を例に取り、以下のステップを実行します。

```
f(x) = 7^x mod 15 の周期 r を量子位相推定 (QPE) で発見
        ↓
r = 4 が判明
        ↓
gcd(7^2 - 1, 15) = 3,  gcd(7^2 + 1, 15) = 5
        ↓
15 = 3 × 5  ✓
```

### 量子回路の構成

```
count[0] ── H ──────────────────── ■ ──── IQFT ── 測定
count[1] ── H ────────────── ■ ─── │ ──── IQFT ── 測定
count[2] ── H ──────── ■ ─── │ ─── │ ──── IQFT ── 測定
count[3] ── H ── ■ ─── │ ─── │ ─── │ ──── IQFT ── 測定
                 │     │     │     │
target[0..3] ── U¹ ── U² ── U⁴ ── U⁸ ──────────── (捨てる)
```

---

## 必要環境

| ソフトウェア | バージョン |
|-------------|-----------|
| Python      | 3.9 以上   |
| Qiskit      | 1.0 以上   |
| qiskit-aer  | 0.14 以上  |
| qiskit-ibm-runtime | 0.20 以上 |

---

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/your-username/shor-factor15-ibmq.git
cd shor-factor15-ibmq

# 仮想環境の作成（推奨）
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 依存ライブラリのインストール
pip install -r requirements.txt
```

---

## 使い方

### シミュレーターで実行（推奨・無料）

```bash
# デフォルト設定で実行 (n_count=8, shots=2048, a=7)
python shor_factor15.py --mode sim

# shots を増やして精度向上
python shor_factor15.py --mode sim --shots 8192

# 回路図を circuit_diagram.png に保存
python shor_factor15.py --mode sim --diagram
```

**期待される出力例:**

```
============================================================
  [シミュレーター] N=15, a=7, n_count=8
============================================================

回路サイズ: 12 量子ビット, 深さ=142, ゲート数=316

測定回数: 2048 shots

--- 素因数導出 ---
  測定値=  64  位相=0.2500  r=4  gcd(7^2-1,15)=3  gcd(7^2+1,15)=5  → 15 = 3 × 5  (確率 23.8%)
  測定値= 192  位相=0.7500  r=4  gcd(7^2-1,15)=3  gcd(7^2+1,15)=5  → 15 = 3 × 5  (確率 22.9%)

✓ 素因数分解成功: 15 = 3 × 5

ヒストグラム保存: sim_histogram.png
```

### IBM Quantum 実機で実行

#### 1. API トークンの取得

1. [IBM Quantum](https://quantum.ibm.com/) にサインイン
2. ダッシュボード右上の **"Copy token"** をクリック

#### 2. 実行

```bash
# 最小待ち時間のバックエンドを自動選択
python shor_factor15.py --mode real --token "YOUR_IBM_QUANTUM_TOKEN"

# バックエンドを手動指定
python shor_factor15.py --mode real --token "TOKEN" --backend ibm_kyoto

# シミュレーターと実機を両方実行
python shor_factor15.py --mode both --token "TOKEN"
```

> **ヒント**: 実機では `n_count` が自動的に 4 に制限されます。  
> デコヒーレンスの影響を最小化するため、なるべく浅い回路を使用します。

#### 3. バックエンド一覧の確認

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum_platform", token="YOUR_TOKEN")
backends = service.backends(simulator=False, operational=True)
for b in backends:
    s = b.status()
    print(f"{b.name:30s}  qubits={b.num_qubits:3d}  pending={s.pending_jobs}")
```

---

## コマンドライン引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--mode` | `sim` | `sim` / `real` / `both` |
| `--token` | なし | IBM Quantum API トークン |
| `--backend` | 自動 | 使用するバックエンド名 |
| `--n_count` | `8` | 位相推定レジスタのビット数 |
| `--shots` | `2048` | 測定の試行回数 |
| `--a` | `7` | 底（`gcd(a, 15) = 1` を満たす整数） |
| `--diagram` | off | 回路図を `circuit_diagram.png` に保存 |

---

## 底 `a` の選択肢

`a` には `gcd(a, 15) = 1` を満たす整数を使用できます。

| a  | 周期 r | 因数導出 |
|----|--------|---------|
| 2  | 4      | gcd(2²-1,15)=3, gcd(2²+1,15)=5 ✓ |
| 4  | 2      | gcd(4¹-1,15)=3, gcd(4¹+1,15)=5 ✓ |
| 7  | 4      | gcd(7²-1,15)=3, gcd(7²+1,15)=5 ✓ |
| 8  | 4      | gcd(8²-1,15)=3, gcd(8²+1,15)=5 ✓ |
| 11 | 2      | gcd(11¹-1,15)=2, gcd(11¹+1,15)=3 → 5 ✓ |
| 13 | 4      | gcd(13²-1,15)=3, gcd(13²+1,15)=5 ✓ |

```bash
# a=2 で実行する例
python shor_factor15.py --mode sim --a 2
```

---

## ファイル構成

```
shor-factor15-ibmq/
├── shor_factor15.py      # メインプログラム
├── requirements.txt      # 依存ライブラリ
└── README.md             # このファイル
```

---

## アルゴリズムの参考文献

- P. W. Shor, "Algorithms for quantum computation: discrete logarithms and factoring," *Proceedings 35th Annual Symposium on Foundations of Computer Science*, pp. 124–134, 1994.
- M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press, 2010.
- [Qiskit Textbook — Shor's Algorithm](https://learn.qiskit.org/course/ch-algorithms/shors-algorithm)

---

## ライセンス

MIT License — 詳細は [LICENSE](LICENSE) を参照してください。
