# プロジェクト人員配置問題 - CPLEX OPL実装

## 概要

このディレクトリには、CPLEX OPL（Optimization Programming Language）を使用したプロジェクト人員配置問題の実装が含まれています。従業員のスキルとプロジェクトの要件を考慮し、期間内に完了できる最適な人員配置を求めます。

## ファイル構成

- `project_assignment.mod` - OPLモデルファイル（問題の定式化）
- `project_assignment.dat` - データファイル（問題のインスタンス）
- `README_PROJECT_ASSIGNMENT.md` - このファイル

## 問題の説明

### プロジェクト人員配置問題

複数のプロジェクトに対して、以下の条件を満たす最適な人員配置を求める問題です：

1. **スキル要件**: 従業員が必要なスキルを持っている
2. **最小人数**: プロジェクトに必要な最小人数を満たす
3. **期間制約**: プロジェクト期間内に必要工数を完了できる
4. **人員の均等配置**: プロジェクト間で人員配置の偏りを最小化

### 主な特徴

- **スキルマッチング**: 従業員のスキルレベル（0-10）とプロジェクトの要件を考慮
- **生産性モデル**: スキルレベルに基づいて従業員の生産性を計算
- **期間制約**: プロジェクトの期間内に完了できる人員を配置
- **均等配置**: プロジェクト間で人員配置の偏りを最小化

## 数理モデル

### 決定変数

- `assign[e][p]`: 従業員eをプロジェクトpに配置する場合1、しない場合0（二値変数）
- `projectSize[p]`: プロジェクトpに配置された人数
- `projectProductivity[p]`: プロジェクトpの総生産性

### 目的関数

```
maximize スキルマッチングスコア - 人員配置の偏り

= Σ(employeeSkill[e][s] × projectRequiredSkill[p][s] × assign[e][p])
  - 100 × (max(projectSize) - min(projectSize))
```

### 制約条件

1. **1人1プロジェクト制約**: 各従業員は最大1つのプロジェクトにのみ配置
   ```
   Σ assign[e][p] ≤ 1  (∀e)
   ```

2. **最小人数制約**: プロジェクトに必要な最小人数を満たす
   ```
   Σ assign[e][p] ≥ minEmployees[p]  (∀p)
   ```

3. **最大人数制約**: プロジェクトの最大人数を超えない
   ```
   Σ assign[e][p] ≤ maxEmployees[p]  (∀p)
   ```

4. **スキル要件制約**: 必要なスキルを持つ従業員のみ配置可能
   ```
   assign[e][p] × projectRequiredSkill[p][s] ≤ (employeeSkill[e][s] ≥ 1)  (∀e,p,s)
   ```

5. **期間制約**: プロジェクト期間内に完了できる
   ```
   projectProductivity[p] × projectDuration[p] ≥ projectWorkload[p]  (∀p)
   ```

### 生産性モデル

従業員の生産性は、必要なスキルの平均レベルに基づいて計算されます：

```
生産性 = 0.5 + (平均スキルレベル × 0.1)
```

**例:**
- スキルレベル0（スキルなし）: 生産性 0.5
- スキルレベル5（中級）: 生産性 1.0
- スキルレベル10（上級）: 生産性 1.5

プロジェクトの総生産性は、配置された全従業員の生産性の合計です。

## 実行方法

### IBM ILOG CPLEX Optimization Studioを使用する場合

1. CPLEX IDEを起動
2. File → New → OPL Project を選択
3. プロジェクト名を入力（例：ProjectAssignment）
4. プロジェクトを右クリック → New → Model を選択し、`project_assignment.mod` の内容をコピー
5. プロジェクトを右クリック → New → Data を選択し、`project_assignment.dat` の内容をコピー
6. プロジェクトを右クリック → New → Run Configuration を選択
7. モデルファイルとデータファイルを選択
8. Run Configuration を右クリック → Run を選択

### コマンドラインから実行する場合

```bash
oplrun project_assignment.mod project_assignment.dat
```

## サンプルデータの説明

### 従業員（20人）

従業員は4つのグループに分類されています：

| グループ | 従業員番号 | 特徴 |
|---------|-----------|------|
| プログラミング重視 | 1-5 | プログラミングスキルが高い（7-10） |
| データベース/インフラ重視 | 6-10 | データベース・インフラスキルが高い（7-10） |
| UI/UXデザイン重視 | 11-15 | UI/UXデザインスキルが高い（8-10） |
| オールラウンダー/PM重視 | 16-20 | バランスが良く、PM能力が高い（8-10） |

### スキルの種類（6種類）

1. プログラミング
2. データベース
3. UI/UXデザイン
4. プロジェクト管理
5. テスト/QA
6. インフラ/DevOps

### プロジェクト（5つ）

| プロジェクト | 必要スキル | 必要工数 | 期間 | 最小人数 | 最大人数 |
|-------------|-----------|---------|------|---------|---------|
| 1: Webアプリ開発 | プログラミング、DB、UI/UX、テスト | 80人日 | 40日 | 2人 | 5人 |
| 2: データ分析システム | プログラミング、DB、インフラ | 100人日 | 50日 | 2人 | 6人 |
| 3: モバイルアプリ | プログラミング、UI/UX、テスト | 60人日 | 30日 | 2人 | 4人 |
| 4: 基幹システム刷新 | 全スキル | 150人日 | 60日 | 3人 | 8人 |
| 5: クラウド移行 | プログラミング、DB、PM、インフラ | 120人日 | 50日 | 2人 | 5人 |

## 出力例

```
=== プロジェクト人員配置の最適解 ===

プロジェクト 1
  配置人数: 3 (最小: 2, 最大: 5)
  配置された従業員: 1 11 15
  スキルマッチスコア: 45
  総生産性: 3.2 人日/日
  必要工数: 80 人日
  プロジェクト期間: 40 日
  予想完了時間: 25.0 日
  期間内完了: 可能
  必要スキル: スキル1 スキル2 スキル3 スキル5
    従業員1 - スキル: スキル1=9 スキル2=6 スキル3=3 スキル5=5 (生産性: 1.08)
    従業員11 - スキル: スキル1=4 スキル2=3 スキル3=10 スキル5=5 (生産性: 1.05)
    従業員15 - スキル: スキル1=7 スキル2=4 スキル3=10 スキル5=5 (生産性: 1.15)

プロジェクト 2
  配置人数: 4 (最小: 2, 最大: 6)
  配置された従業員: 6 7 9 10
  ...

配置されなかった従業員: なし

=== サマリー ===
総スキルマッチスコア: 450
配置従業員数: 20 / 20

人員配置の均等性:
  平均人数: 4.0
  最大人数: 5
  最小人数: 3
  差: 2
```

## データファイルのカスタマイズ

### 基本パラメータの変更

```opl
// 従業員、プロジェクト、スキルの数を変更
numEmployees = 30;
numProjects = 8;
numSkills = 8;
```

### 従業員のスキルレベルの設定

```opl
// 各従業員のスキルレベル（0-10）
// 0 = スキルなし、1-3 = 初級、4-6 = 中級、7-9 = 上級、10 = エキスパート
employeeSkill = [
  [9, 7, 3, 4, 5, 2, 6, 4],  // 従業員1
  [8, 8, 2, 3, 6, 3, 5, 5],  // 従業員2
  // ...
];
```

### プロジェクトの設定

```opl
// 必要なスキル（1=必要、0=不要）
projectRequiredSkill = [
  [1, 1, 1, 0, 1, 0, 0, 1],  // プロジェクト1
  // ...
];

// 必要工数（人日）
projectWorkload = [100, 80, 120, 150, 90];

// 期間（日数）
projectDuration = [50, 40, 60, 70, 45];

// 最小・最大人数
minEmployees = [2, 2, 3, 4, 2];
maxEmployees = [5, 4, 6, 8, 5];
```

## モデルの拡張

### 1. コスト最小化の追加

従業員のコストを考慮する場合：

```opl
// データファイルに追加
float employeeCost[Employees] = [5000, 6000, 4500, ...];

// 目的関数を変更
minimize
  sum(e in Employees, p in Projects) employeeCost[e] * assign[e][p]
  - sum(e in Employees, p in Projects, s in Skills) 
      employeeSkill[e][s] * projectRequiredSkill[p][s] * assign[e][p] * 10;
```

### 2. 複数プロジェクトへの配置

従業員が複数のプロジェクトに参加できる場合：

```opl
// 制約1を変更
forall(e in Employees)
  ctMaxProjects:
    sum(p in Projects) assign[e][p] <= 2; // 最大2プロジェクトまで

// 期間の重複チェックを追加
int projectStartDate[Projects] = ...;
int projectEndDate[Projects] = ...;

forall(e in Employees, p1 in Projects, p2 in Projects: p1 < p2)
  ctNoOverlap:
    assign[e][p1] + assign[e][p2] <= 1 + 
    (projectEndDate[p1] < projectStartDate[p2] || 
     projectEndDate[p2] < projectStartDate[p1]);
```

### 3. プロジェクトの優先度

プロジェクトに優先度を設定する場合：

```opl
// データファイルに追加
int projectPriority[Projects] = [10, 5, 8, 9, 6];

// 補助変数を追加
dvar boolean projectActive[Projects];

// 目的関数に追加
+ sum(p in Projects) projectPriority[p] * projectActive[p] * 100;

// 制約を追加
forall(p in Projects)
  ctProjectActive:
    sum(e in Employees) assign[e][p] >= projectActive[p];
```

### 4. スキルレベルの最小要件

プロジェクトに必要な最小スキルレベルを設定：

```opl
// データファイルに追加
int minSkillLevel[Projects][Skills] = ...;

// 制約を追加
forall(e in Employees, p in Projects, s in Skills)
  ctMinSkillLevel:
    assign[e][p] * projectRequiredSkill[p][s] * minSkillLevel[p][s] <= 
    employeeSkill[e][s];
```

### 5. チームバランスの考慮

各プロジェクトに異なるスキルを持つ従業員を配置：

```opl
// 各スキルについて最低1人は配置
forall(p in Projects, s in Skills: projectRequiredSkill[p][s] == 1)
  ctSkillCoverage:
    sum(e in Employees: employeeSkill[e][s] >= 5) assign[e][p] >= 1;
```

## トラブルシューティング

### エラー: "No solution found"

**原因:**
- 制約が厳しすぎて実行可能解が存在しない
- プロジェクトの期間が短すぎる
- 必要なスキルを持つ従業員が不足
- 最小人数の合計が従業員数を超えている

**対策:**
1. プロジェクトの期間を延長
2. 最小人数の要件を緩和
3. 従業員のスキルレベルを確認
4. 必要工数を削減

### 警告: "期間内完了: 不可能"

**原因:**
- 配置された従業員の生産性が不足
- プロジェクトの必要工数が多すぎる

**対策:**
1. より高いスキルレベルの従業員を配置
2. プロジェクトの最大人数を増やす
3. プロジェクトの期間を延長
4. 必要工数を見直す

### 人員配置が偏っている

**原因:**
- 目的関数の重み付けが適切でない
- スキルマッチングの優先度が高すぎる

**対策:**
```opl
// 均等配置の重みを増やす
maximize
  sum(...) // スキルマッチング
  - 200 * (max(p in Projects) projectSize[p] - min(p in Projects) projectSize[p]);
```

### 最適化が遅い

**原因:**
- 従業員数やプロジェクト数が多い
- 制約が複雑

**対策:**
1. CPLEXのパラメータを調整
```opl
execute {
  cplex.tilim = 600;  // 制限時間（秒）
  cplex.epgap = 0.05; // 最適性ギャップ（5%）
  cplex.threads = 4;  // 使用スレッド数
}
```

2. 問題を分割して段階的に解く
3. 一部の制約を緩和

## 実用的な使用例

### ケース1: ソフトウェア開発会社

```opl
// 30人の開発者を10のプロジェクトに配置
numEmployees = 30;
numProjects = 10;
numSkills = 8; // Java, Python, React, SQL, AWS, Docker, Agile, Testing
```

### ケース2: コンサルティング会社

```opl
// 50人のコンサルタントを15のプロジェクトに配置
numEmployees = 50;
numProjects = 15;
numSkills = 6; // 戦略, 財務, IT, 人事, マーケティング, 業務改善
```

### ケース3: 建設プロジェクト

```opl
// 40人の作業員を20の現場に配置
numEmployees = 40;
numProjects = 20;
numSkills = 5; // 電気, 配管, 大工, 塗装, 重機操作
```

### ケース4: 病院のシフト管理

```opl
// 100人の医療スタッフを30のシフトに配置
numEmployees = 100;
numProjects = 30; // シフト
numSkills = 4; // 内科, 外科, 小児科, 救急
```

## パフォーマンスのヒント

### 大規模問題の場合

1. **段階的な最適化:**
   - まず重要なプロジェクトに人員を配置
   - 残りのプロジェクトを順次最適化

2. **制約の緩和:**
   - 最小人数を緩和
   - 期間制約を緩和
   - 段階的に制約を追加

3. **ヒューリスティック解の利用:**
   - 貪欲法で初期解を生成
   - warmstart機能を使用

4. **並列処理:**
```opl
execute {
  cplex.threads = 8;  // 使用可能なCPUコア数
}
```

## 関連する問題

- **割り当て問題（Assignment Problem）**: 1対1の割り当て
- **ジョブショップスケジューリング**: 時間的制約を考慮
- **ナース・スケジューリング**: シフト管理
- **車両配送問題**: ルート最適化と組み合わせ

## 参考資料

- [IBM ILOG CPLEX Optimization Studio Documentation](https://www.ibm.com/docs/en/icos)
- [OPL Language Reference Manual](https://www.ibm.com/docs/en/icos/latest?topic=cplex-opl-language-reference-manual)
- [Assignment Problem - Wikipedia](https://en.wikipedia.org/wiki/Assignment_problem)
- [Resource-Constrained Project Scheduling](https://en.wikipedia.org/wiki/Resource-constrained_project_scheduling)

## ライセンス

このコードは教育目的で提供されています。CPLEX の使用には適切なライセンスが必要です。

## バージョン履歴

- v1.0 (2026-03-11): 初版リリース
  - 基本的なプロジェクト人員配置問題の実装
  - スキルマッチングと期間制約
  - 20人の従業員、5つのプロジェクトのサンプルデータ
  - 詳細な出力表示