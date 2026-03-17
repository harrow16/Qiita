# ナップサック問題 - CPLEX OPL実装

## 概要

このディレクトリには、CPLEX OPL（Optimization Programming Language）を使用したナップサック問題の実装が含まれています。

## ファイル構成

- `knapsack_opl.mod` - OPLモデルファイル（問題の定式化）
- `knapsack_opl.dat` - データファイル（問題のインスタンス）
- `README_KNAPSACK_OPL.md` - このファイル

## 問題の説明

### ナップサック問題（0-1 Knapsack Problem）

容量が限られたナップサックに、価値と重さが異なる複数のアイテムを詰める問題です。各アイテムは1つまで選択でき、ナップサックの容量を超えない範囲で、価値の合計を最大化することが目標です。

### 数理モデル

**決定変数:**
- `x[i]`: アイテムiを選択する場合1、選択しない場合0（二値変数）

**目的関数:**
```
maximize Σ(value[i] * x[i])
```

**制約条件:**
```
Σ(weight[i] * x[i]) ≤ capacity
x[i] ∈ {0, 1}
```

## 実行方法

### 1. IBM ILOG CPLEX Optimization Studioを使用する場合

#### ステップ1: プロジェクトの作成
1. CPLEX IDEを起動
2. File → New → OPL Project を選択
3. プロジェクト名を入力（例：KnapsackProblem）
4. Finish をクリック

#### ステップ2: ファイルの追加
1. プロジェクトを右クリック → New → Model を選択
2. `knapsack_opl.mod` の内容をコピー＆ペースト
3. プロジェクトを右クリック → New → Data を選択
4. `knapsack_opl.dat` の内容をコピー＆ペースト

#### ステップ3: 実行設定
1. プロジェクトを右クリック → New → Run Configuration を選択
2. モデルファイルとデータファイルを選択
3. OK をクリック

#### ステップ4: 実行
1. Run Configuration を右クリック → Run を選択
2. 結果がコンソールに表示されます

### 2. コマンドラインから実行する場合

```bash
oplrun knapsack_opl.mod knapsack_opl.dat
```

## サンプルデータの説明

`knapsack_opl.dat` には以下のサンプルデータが含まれています：

| アイテム | 価値 | 重さ |
|---------|------|------|
| 1       | 10   | 5    |
| 2       | 40   | 4    |
| 3       | 30   | 6    |
| 4       | 50   | 3    |
| 5       | 35   | 7    |

- ナップサックの容量: 15

### 期待される解

最適解では、アイテム2、4、5が選択され：
- 合計価値: 125
- 合計重さ: 14
- 容量使用率: 93.33%

## 出力例

```
=== ナップサック問題の解 ===
ナップサックの容量: 15

選択されたアイテム:
  アイテム 2: 価値=40, 重さ=4
  アイテム 4: 価値=50, 重さ=3
  アイテム 5: 価値=35, 重さ=7

合計価値: 125
合計重さ: 14
容量使用率: 93.33%
```

## データファイルのカスタマイズ

独自の問題を解くには、`knapsack_opl.dat` を編集してください：

```opl
// アイテムの数を変更
n = 10;

// 容量を変更
capacity = 50;

// 価値の配列を変更（n個の値）
value = [12, 25, 18, 30, 45, 22, 38, 15, 28, 35];

// 重さの配列を変更（n個の値）
weight = [5, 8, 6, 10, 12, 7, 9, 4, 8, 11];
```

## モデルの拡張

### 多重ナップサック問題への拡張

各アイテムを複数個選択できるようにする場合：

```opl
// 決定変数を整数変数に変更
dvar int+ x[Items];

// 各アイテムの最大個数を追加
int maxQuantity[Items] = ...;

// 制約を追加
subject to {
  ctCapacity:
    sum(i in Items) weight[i] * x[i] <= capacity;
    
  forall(i in Items)
    ctMaxQuantity:
      x[i] <= maxQuantity[i];
}
```

### 多次元ナップサック問題への拡張

複数の制約（重さ、体積など）を追加する場合：

```opl
int volume[Items] = ...;
int maxVolume = ...;

subject to {
  ctCapacity:
    sum(i in Items) weight[i] * x[i] <= capacity;
  
  ctVolume:
    sum(i in Items) volume[i] * x[i] <= maxVolume;
}
```

### 価値の最小化（コスト最小化）

価値を最小化する場合（例：コスト最小化）：

```opl
// 目的関数を変更
minimize
  sum(i in Items) value[i] * x[i];
```

### 必須アイテムの追加

特定のアイテムを必ず選択する制約：

```opl
// 必須アイテムのセット
{int} MandatoryItems = {1, 3};

subject to {
  forall(i in MandatoryItems)
    ctMandatory:
      x[i] == 1;
}
```

## トラブルシューティング

### エラー: "No solution found"

**原因:**
- データファイルの値が不正
- 容量が小さすぎて、どのアイテムも入らない
- 必須制約が矛盾している

**対策:**
1. データファイルの値を確認
2. 容量を増やす
3. 制約条件を見直す

### エラー: "Array size mismatch"

**原因:**
- `n` の値と配列のサイズが一致していない

**対策:**
```opl
// 正しい例
n = 5;
value = [10, 40, 30, 50, 35];  // 5個
weight = [5, 4, 6, 3, 7];      // 5個
```

### エラー: "undefined method 'toFixed'"

**原因:**
- OPLでは一部のJavaScriptメソッドが使用できない

**対策:**
- 数値の丸め処理には `Opl.round()` を使用
```opl
// 誤り
var result = (value * 100.0).toFixed(2);

// 正しい
var result = Opl.round(value * 100) / 100;
```

### 実行が遅い

**原因:**
- アイテム数が多い場合、計算時間が増加

**対策:**
1. CPLEXのパラメータを調整
2. 問題を分割して段階的に解く
3. ヒューリスティック手法を検討

## 実用的な使用例

### ケース1: 投資ポートフォリオ最適化

```opl
// アイテム = 投資案件
// 価値 = 期待リターン
// 重さ = 投資額
// 容量 = 予算
```

### ケース2: プロジェクト選択

```opl
// アイテム = プロジェクト
// 価値 = プロジェクトの利益
// 重さ = 必要なリソース
// 容量 = 利用可能なリソース
```

### ケース3: 配送最適化

```opl
// アイテム = 荷物
// 価値 = 配送料金
// 重さ = 荷物の重量
// 容量 = トラックの積載量
```

## パフォーマンスのヒント

### 大規模問題の場合

1. **前処理を活用:**
   - 明らかに選択されるアイテムを事前に固定
   - 明らかに選択されないアイテムを除外

2. **分枝限定法のパラメータ調整:**
```opl
execute {
  cplex.tilim = 300;  // 制限時間（秒）
  cplex.epgap = 0.01; // 最適性ギャップ（1%）
}
```

3. **ヒューリスティック解の利用:**
   - 貪欲法で初期解を生成
   - warmstart機能を使用

## 関連する問題

- **部分和問題**: 価値と重さが同じ場合
- **ビンパッキング問題**: 複数のナップサックに詰める
- **多次元ナップサック問題**: 複数の制約条件
- **多重ナップサック問題**: アイテムを複数個選択可能

## 参考資料

- [IBM ILOG CPLEX Optimization Studio Documentation](https://www.ibm.com/docs/en/icos)
- [OPL Language Reference Manual](https://www.ibm.com/docs/en/icos/latest?topic=cplex-opl-language-reference-manual)
- [Knapsack Problem - Wikipedia](https://en.wikipedia.org/wiki/Knapsack_problem)

## ライセンス

このコードは教育目的で提供されています。CPLEX の使用には適切なライセンスが必要です。

## バージョン履歴

- v1.0 (2026-03-10): 初版リリース
  - 基本的な0-1ナップサック問題の実装
  - サンプルデータファイル
  - 詳細な出力表示