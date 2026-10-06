# H₂ 基底エネルギー計算 — Qiskit Nature + VQE

Qiskit Nature を使って水素分子（H₂）の基底エネルギーを量子コンピュータで計算するプロジェクトです。

- **シミュレーション版**: 状態ベクトルシミュレータで高精度計算
- **実機版**: IBM Quantum 実機を自動選択してエネルギー計算

---

## ディレクトリ構成

```
IBMQ_#5/
├── src/
│   ├── h2_simulation.py   # シミュレーション版（PEC 計算）
│   └── h2_realdevice.py   # IBM Quantum 実機版
├── results/               # 計算結果（CSV・PNG）出力先
├── .env                   # API キー（★ GitHub に上げないこと）
├── .env.example           # API キーのテンプレート
├── .gitignore
├── requirements.txt
└── README.md
```

---

## セットアップ

### 1. Python 環境

```bash
pip install -r requirements.txt
```

PySCF（量子化学バックエンド）は Linux / macOS 向けです。  
Windows 環境では WSL の利用を推奨します。

### 2. API キーの設定

`.env.example` をコピーして `.env` を作成し、IBM Quantum の API キーを入力します。

```bash
cp .env.example .env
```

`.env`:
```
IBMQ_API_KEY=your_ibm_quantum_api_key_here
```

> ⚠️ `.env` は `.gitignore` に登録済みです。**絶対に GitHub に公開しないでください。**

---

## 実行方法

### シミュレーション版

```bash
python src/h2_simulation.py
```

- 核間距離 0.5 Å ～ 2.5 Å を 17 点計算
- VQE（UCCSD アンザッツ）と厳密対角化を比較
- `results/pec_simulation.png` と `results/pec_simulation.csv` を出力

### 実機版

```bash
python src/h2_realdevice.py
```

- 利用可能な IBM Quantum 実機を自動選択（最少キュー優先）
- 核間距離 0.735 Å（平衡点付近）のエネルギーを計算
- `results/realdevice_result.png` と `results/realdevice_result.txt` を出力

---

## 計算手法の概要

| 項目 | 内容 |
|---|---|
| 分子 | H₂（水素分子）|
| 基底関数 | STO-3G（最小基底）|
| 活性空間 | 電子数 2、空間軌道数 2 |
| マッパー | ParityMapper（2量子ビット削減）|
| アンザッツ | UCCSD |
| 最適化（シミュレータ）| SLSQP |
| 最適化（実機）| COBYLA |
| エラー緩和（実機）| Resilience Level 1（読み出しエラー緩和）|

---

## 実行環境

| パッケージ | バージョン |
|---|---|
| Python | 3.11 |
| qiskit | 2.5.2 |
| qiskit-nature | 0.8.0 |
| qiskit-ibm-runtime | 0.49.0 |
| qiskit-aer | 0.17.2 |

---

## 参考文献

- [Qiskit Nature ドキュメント](https://qiskit-community.github.io/qiskit-nature/)
- [IBM Quantum](https://quantum.ibm.com/)
- Peruzzo et al., "A variational eigenvalue solver on a quantum processor", Nature Communications (2014)
