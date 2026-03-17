/*********************************************
 * ナップサック問題 (0-1 Knapsack Problem)
 * CPLEX OPL モデル
 *********************************************/

// パラメータの宣言
int n = ...; // アイテムの数
int capacity = ...; // ナップサックの容量

range Items = 1..n; // アイテムのインデックス

int value[Items] = ...; // 各アイテムの価値
int weight[Items] = ...; // 各アイテムの重さ

// 決定変数
dvar boolean x[Items]; // x[i] = 1 ならアイテムiを選択、0なら選択しない

// 目的関数：価値の合計を最大化
maximize
  sum(i in Items) value[i] * x[i];

// 制約条件
subject to {
  // 容量制約：選択したアイテムの重さの合計が容量以下
  ctCapacity:
    sum(i in Items) weight[i] * x[i] <= capacity;
}

// 結果の出力
execute DISPLAY {
  writeln("=== ナップサック問題の解 ===");
  writeln("ナップサックの容量: ", capacity);
  writeln();
  
  var totalValue = 0;
  var totalWeight = 0;
  
  writeln("選択されたアイテム:");
  for(var i in Items) {
    if(x[i] == 1) {
      writeln("  アイテム ", i, ": 価値=", value[i], ", 重さ=", weight[i]);
      totalValue += value[i];
      totalWeight += weight[i];
    }
  }
  
  writeln();
  writeln("合計価値: ", totalValue);
  writeln("合計重さ: ", totalWeight);
  writeln("容量使用率: ", Opl.round(totalWeight * 100.0 / capacity * 100) / 100, "%");
}