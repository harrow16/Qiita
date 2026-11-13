# iris-spss-nn-python

SPSS Modeler でエクスポートした iris_class.xml（PMML）のニューラルネットワークモデルを Python で再実装したプログラムです。

## 概要

アヤメ（iris）の計測データから品種を予測する SPSS Modeler のニューラルネットワークモデルを、Python の標準ライブラリのみを使って完全に再現します。
SPSS Modeler が使えない環境でも、同じ予測精度でモデルを動かすことができます。

## モデル構成

```
入力層（4ノード）
  ├─ petal length（花弁の長さ）
  ├─ petal width（花弁の幅）
  ├─ sepal length（がく片の長さ）
  └─ sepal width（がく片の幅）
      ↓ LinearNorm で正規化
隠れ層（3ノード）  活性化関数: tanh
      ↓
出力層（3ノード）  活性化関数: identity → softmax
  ├─ Iris-setosa の確率
  ├─ Iris-versicolor の確率
  └─ Iris-virginica の確率
      ↓
最大確率のクラスを予測結果として出力
```

| 要素 | 内容 |
|---|---|
| アルゴリズム | MLP（多層パーセプトロン） |
| 入力層 | 4 ノード |
| 隠れ層 | 3 ノード（活性化関数: tanh） |
| 出力層 | 3 ノード（活性化関数: identity + softmax） |
| 予測クラス | Iris-setosa / Iris-versicolor / Iris-virginica |
| 学習ツール | IBM SPSS Modeler |
| モデルエクスポート形式 | PMML 4.3（XML） |

## ファイル構成

```
.
├── iris_nn_predict.py   # メインプログラム
├── iris.csv             # 入力データ（アヤメのデータセット 150件）
├── iris_class.xml       # SPSS Modeler からエクスポートした PMML ファイル
├── result.csv           # 予測結果（実行後に生成）
└── README.md            # このファイル
```

## 動作環境

- Python 3.8 以上
- 標準ライブラリのみ使用（追加インストール不要）

## 使い方

### 1. リポジトリをクローン

```bash
git clone https://github.com/your-username/iris-spss-nn-python.git
cd iris-spss-nn-python
```

### 2. 実行

```bash
python iris_nn_predict.py
```

### 3. オプション指定

```bash
python iris_nn_predict.py --input iris.csv --output result.csv
```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--input` | `iris.csv` | 入力 CSV ファイルのパス |
| `--output` | `result.csv` | 出力 CSV ファイルのパス |

## 実行結果

```
予測完了: 150 件  正解: 145 件  正解率: 96.7%
結果ファイル: result.csv
```

### 出力 CSV の列

| 列名 | 説明 |
|---|---|
| sepal length | がく片の長さ（入力値） |
| sepal width | がく片の幅（入力値） |
| petal length | 花弁の長さ（入力値） |
| petal width | 花弁の幅（入力値） |
| true_class | 正解クラス |
| $N-class | 予測クラス |
| correct | 正誤（○ / ×） |
| prob_Iris-setosa | Iris-setosa の予測確率 |
| prob_Iris-versicolor | Iris-versicolor の予測確率 |
| prob_Iris-virginica | Iris-virginica の予測確率 |

## 実装のポイント

PMML の各要素を Python で実装した対応表です。

| PMML の処理 | Python での実装 |
|---|---|
| `NormContinuous` / `LinearNorm` による正規化 | 2点線形補間 |
| 隠れ層（`activationFunction="tanh"`） | `math.tanh()` |
| 出力層（`normalizationMethod="softmax"`） | softmax 関数（自作） |
| 最大確率クラスの選択 | `list.index(max(...))` |

## 関連記事

Qiita に解説記事を公開しています。ニューラルネットワークの仕組みから PMML の読み方まで、初心者向けに図解で説明しています。

👉 [SPSS Modelerのニューラルネットワークモデルを外部プログラムで実装する（Qiita）](https://qiita.com/your-username/items/xxxxxxxxxxxx)

## ライセンス

MIT License

## 作者

your-username
