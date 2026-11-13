"""
iris_nn_predict.py
SPSS Modeler でエクスポートした iris_class.xml（PMML）の
ニューラルネットワークモデルを Python で再現し、iris.csv を予測する。

モデル構成:
  入力層  : 4 ノード (petal length, petal width, sepal length, sepal width)
  隠れ層  : 3 ノード  (活性化関数: tanh)
  出力層  : 3 ノード  (活性化関数: identity → softmax)
  クラス  : Iris-setosa / Iris-versicolor / Iris-virginica

使い方:
  python iris_nn_predict.py
  python iris_nn_predict.py --input iris.csv --output result.csv
"""

import argparse
import math
import csv


# ---------------------------------------------------------------------------
# 1. モデルパラメータ（iris_class.xml から抽出）
# ---------------------------------------------------------------------------

# --- 入力の正規化パラメータ (LinearNorm: orig → norm) ---
# PMML の NormContinuous は 2 点の線形補間で正規化値を求める。
#   norm = norm0 + (orig - orig0) * (norm1 - norm0) / (orig1 - orig0)
NORM_PARAMS = {
    "petal length": {"orig0": 1.0,   "norm0": -1.46586538248623,
                     "orig1": 6.7,   "norm1":  1.6518333787718},
    "petal width":  {"orig0": 0.1,   "norm0": -1.34050985136666,
                     "orig1": 2.5,   "norm1":  1.66757993086306},
    "sepal length": {"orig0": 4.3,   "norm0": -1.70103748484687,
                     "orig1": 7.9,   "norm1":  2.33532264868807},
    "sepal width":  {"orig0": 2.2,   "norm0": -2.21928167345622,
                     "orig1": 4.4,   "norm1":  3.35910860476342},
}

# --- 隠れ層（ニューロン 4, 5, 6）の重みとバイアス ---
# 入力順: id=0(petal length), id=1(petal width), id=2(sepal length), id=3(sepal width)
HIDDEN_WEIGHTS = [
    # [w0, w1, w2, w3, bias]
    [ 1.63524086591826,  1.69202607500018, -0.368620439006194, -0.394332172983169, -1.952156186602],    # ニューロン4
    [ 0.961066704193676, 0.725056813190881, 0.744188269304133, -1.93454939563185,   0.672326703458472],  # ニューロン5
    [-0.936987592959351,-0.846862749302285,-0.20728897126586,   0.614758140205065,  0.426022369035855],  # ニューロン6
]

# --- 出力層（ニューロン 7, 8, 9）の重みとバイアス ---
# 入力順: id=4(ニューロン4), id=5(ニューロン5), id=6(ニューロン6)
OUTPUT_WEIGHTS = [
    # [w4, w5, w6, bias]  → ニューロン7 → Iris-setosa
    [-1.63949201106099, -2.50383593866568,  2.49734079640179,  0.315540222577777],
    # → ニューロン8 → Iris-versicolor
    [-1.80090780169747,  1.19192521804626, -0.161690750885013, 0.844845840436693],
    # → ニューロン9 → Iris-virginica
    [ 3.4229914944065,   0.767083137839152,-2.06342675033851, -0.677592466441299],
]

CLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


# ---------------------------------------------------------------------------
# 2. ヘルパー関数
# ---------------------------------------------------------------------------

def normalize(value: float, params: dict) -> float:
    """LinearNorm で正規化（2 点の線形補間）"""
    orig0, norm0 = params["orig0"], params["norm0"]
    orig1, norm1 = params["orig1"], params["norm1"]
    slope = (norm1 - norm0) / (orig1 - orig0)
    return norm0 + (value - orig0) * slope


def tanh(x: float) -> float:
    """双曲線正接（活性化関数: 隠れ層）"""
    return math.tanh(x)


def softmax(values: list) -> list:
    """ソフトマックス関数（出力層の確率変換）"""
    max_v = max(values)  # オーバーフロー防止
    exp_v = [math.exp(v - max_v) for v in values]
    total = sum(exp_v)
    return [e / total for e in exp_v]


# ---------------------------------------------------------------------------
# 3. 推論関数
# ---------------------------------------------------------------------------

def predict(petal_length: float, petal_width: float,
            sepal_length: float, sepal_width: float) -> dict:
    """
    1 件分の推論を行い、予測クラスと各クラスの確率を返す。

    Returns
    -------
    dict
        {
            "$N-class": str,            # 予測クラス名
            "prob_Iris-setosa": float,
            "prob_Iris-versicolor": float,
            "prob_Iris-virginica": float,
        }
    """
    # ---- Step 1: 入力値を正規化 ----
    x = [
        normalize(petal_length, NORM_PARAMS["petal length"]),
        normalize(petal_width,  NORM_PARAMS["petal width"]),
        normalize(sepal_length, NORM_PARAMS["sepal length"]),
        normalize(sepal_width,  NORM_PARAMS["sepal width"]),
    ]

    # ---- Step 2: 隠れ層の計算（活性化: tanh）----
    hidden = []
    for w in HIDDEN_WEIGHTS:
        z = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + w[3]*x[3] + w[4]  # 線形結合 + バイアス
        hidden.append(tanh(z))

    # ---- Step 3: 出力層の計算（活性化: identity → softmax）----
    pre_softmax = []
    for w in OUTPUT_WEIGHTS:
        z = w[0]*hidden[0] + w[1]*hidden[1] + w[2]*hidden[2] + w[3]  # 線形結合 + バイアス
        pre_softmax.append(z)

    probs = softmax(pre_softmax)

    # ---- Step 4: 最大確率のクラスを選択 ----
    max_idx = probs.index(max(probs))
    predicted_class = CLASSES[max_idx]

    return {
        "$N-class": predicted_class,
        "prob_Iris-setosa":     probs[0],
        "prob_Iris-versicolor": probs[1],
        "prob_Iris-virginica":  probs[2],
    }


# ---------------------------------------------------------------------------
# 4. CSV 読み込み → 推論 → 結果出力
# ---------------------------------------------------------------------------

def run(input_path: str, output_path: str) -> None:
    results = []
    correct = 0
    total = 0

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            petal_length = float(row["petal length"])
            petal_width  = float(row["petal width"])
            sepal_length = float(row["sepal length"])
            sepal_width  = float(row["sepal width"])
            true_class   = row["class"].strip('"')

            pred = predict(petal_length, petal_width, sepal_length, sepal_width)
            pred["true_class"]   = true_class
            pred["petal length"] = petal_length
            pred["petal width"]  = petal_width
            pred["sepal length"] = sepal_length
            pred["sepal width"]  = sepal_width
            pred["correct"]      = "○" if pred["$N-class"] == true_class else "×"

            results.append(pred)
            total += 1
            if pred["$N-class"] == true_class:
                correct += 1

    # --- CSV 出力 ---
    fieldnames = [
        "sepal length", "sepal width", "petal length", "petal width",
        "true_class", "$N-class", "correct",
        "prob_Iris-setosa", "prob_Iris-versicolor", "prob_Iris-virginica",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    accuracy = correct / total * 100
    print(f"予測完了: {total} 件  正解: {correct} 件  正解率: {accuracy:.1f}%")
    print(f"結果ファイル: {output_path}")


# ---------------------------------------------------------------------------
# 5. エントリーポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPSS Modeler ニューラルネットワークの Python 実装")
    parser.add_argument("--input",  default="iris.csv",    help="入力 CSV ファイルパス")
    parser.add_argument("--output", default="result.csv",  help="出力 CSV ファイルパス")
    args = parser.parse_args()

    run(args.input, args.output)
