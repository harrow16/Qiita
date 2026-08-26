# 量子コンピューターで 1 + 1 を計算してみた（IBM Quantum / Qiskit）

> Qiskit を使って、最もシンプルな量子加算回路を実装するサンプルです。
> ローカルシミュレーターと **IBM Quantum 実機** の両方で動作します。

---

## 概要

「量子コンピューターって何ができるの？」という疑問に答えるため、まずは最もシンプルな例として **1 + 1 = 2** を量子回路で計算してみます。

1ビット加算回路を CNOT ゲートと Toffoli ゲートで実装し、シミュレーターおよび IBM Quantum の実機で実行します。

---

## 環境

| ツール | バージョン |
|---|---|
| Python | 3.11 以上 |
| qiskit | 1.0 以上 |
| qiskit-aer | 0.14 以上 |
| qiskit-ibm-runtime | 0.49 以上 |

---

## セットアップ

```bash
pip install -r requirements.txt
```

---

## 量子回路の設計

### 1ビット加算の論理

| A | B | Sum (A XOR B) | Carry (A AND B) |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 |
| 1 | 0 | 1 | 0 |
| **1** | **1** | **0** | **1** |

今回は **A=1, B=1** なので → `Sum=0, Carry=1` → 2進数 `10` = **10進数 2**

### 量子ビットの割り当て

```
q[0] : A     (入力 1)
q[1] : B     (入力 1)
q[2] : Sum   (和ビット)       → 測定
q[3] : Carry (桁上がりビット)  → 測定
```

### 使用ゲート

| ゲート | 役割 | 古典論理との対応 |
|---|---|---|
| **X（NOT）** | 入力ビットを 1 にセット | - |
| **CNOT（CX）** | 制御ビットが1のときターゲットを反転 | XOR |
| **Toffoli（CCX）** | 制御2つが両方1のときターゲットを反転 | AND |

---

## 実行方法

### シミュレーター（ローカル・ノイズなし）

```bash
python quantum_add.py --mode sim
```

### IBM Quantum 実機

```bash
python quantum_add.py --mode real --token <あなたのAPIトークン>
```

API トークンは [IBM Quantum ダッシュボード](https://quantum.ibm.com/) で取得できます。

### 出力例（シミュレーター）

```
=== 量子加算回路: 1 + 1 ===

【量子回路】
     ┌───┐ ░                 ░
q_0: ┤ X ├─░───■─────────■───░───────
     ├───┤ ░   │         │   ░
q_1: ┤ X ├─░───┼────■────■───░───────
     └───┘ ░ ┌─┴─┐┌─┴─┐  │   ░ ┌─┐
q_2: ──────░─┤ X ├┤ X ├──┼───░─┤M├───
           ░ └───┘└───┘┌─┴─┐ ░ └╥┘┌─┐
q_3: ──────░───────────┤ X ├─░──╫─┤M├
           ░           └───┘ ░  ║ └╥┘
c: 2/═══════════════════════════╩══╩═
                                0  1

--- ローカルシミュレーターで実行 ---

【測定結果】 {'10': 1024}
   ビット列       回数       確率     解釈
---------------------------------------------
      10       1024    100.0%   10(2進) = 2(10進) ← 正解

結論: 1 + 1 = 2 ✓
```

---

## ハマりポイント（Windows 環境）

### orjson の DLL ブロック

Windows の AppLocker / WDAC ポリシーが有効な環境では、`qiskit-ibm-runtime` の依存ライブラリ `orjson`（Rust製 `.pyd`）がブロックされることがあります。

```
ImportError: DLL load failed while importing orjson:
アプリケーション制御ポリシーによってこのファイルがブロックされました。
```

同梱の [`orjson_patch.py`](orjson_patch.py) が標準ライブラリの `json` で `orjson` を差し替えます。
`quantum_add.py` の先頭で自動的に読み込まれるため、追加の設定は不要です。

---

## ファイル構成

```
.
├── quantum_add.py    # メインプログラム（シミュレーター／実機 切替対応）
├── orjson_patch.py   # Windows AppLocker 環境用 orjson 代替パッチ
├── requirements.txt  # 依存パッケージ
└── README.md         # この文書
```

---

## 参考

- [Qiskit 公式ドキュメント](https://docs.quantum.ibm.com/)
- [IBM Quantum](https://quantum.ibm.com/)
- [IBM Quantum Composer](https://quantum.ibm.com/composer)
