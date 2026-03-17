/*********************************************
 * プロジェクト人員配置問題（期間制約付き）
 * Project Assignment Problem with Skills and Duration
 * CPLEX OPL モデル
 * 
 * 目的：人員の均等配置とスキルマッチングの最大化
 * 制約：プロジェクト期間内に完了できる人員を配置
 *********************************************/

// パラメータの宣言
int numEmployees = ...; // 従業員の数
int numProjects = ...; // プロジェクトの数
int numSkills = ...; // スキルの種類数

range Employees = 1..numEmployees; // 従業員のインデックス
range Projects = 1..numProjects; // プロジェクトのインデックス
range Skills = 1..numSkills; // スキルのインデックス

// 従業員のスキルレベル（0-10: 0=持っていない、1-10=スキルレベル）
int employeeSkill[Employees][Skills] = ...;

// プロジェクトに必要なスキル（1=必要、0=不要）
int projectRequiredSkill[Projects][Skills] = ...;

// プロジェクトの必要工数（人日）
float projectWorkload[Projects] = ...;

// プロジェクトの期間（日数）
int projectDuration[Projects] = ...;

// プロジェクトに必要な最小人数
int minEmployees[Projects] = ...;

// プロジェクトに配置可能な最大人数
int maxEmployees[Projects] = ...;

// 従業員の生産性係数（スキルレベルに基づく）
// スキルレベル1なら0.5、レベル10なら1.5の生産性
// productivity = 0.5 + (skillLevel / 10) * 1.0

// 決定変数
dvar boolean assign[Employees][Projects]; // assign[e][p] = 1 なら従業員eをプロジェクトpに配置

// 補助変数：各プロジェクトに配置された人数
dvar int+ projectSize[Projects];

// 補助変数：各プロジェクトの総生産性
dvar float+ projectProductivity[Projects];

// 目的関数：スキルマッチングを最大化し、人員配置の偏りを最小化
maximize
  // スキルマッチングスコアの最大化
  sum(e in Employees, p in Projects, s in Skills) 
    employeeSkill[e][s] * projectRequiredSkill[p][s] * assign[e][p]
  // 人員配置の均等性（最大値と最小値の差を最小化）
  - 100 * (max(p in Projects) projectSize[p] - min(p in Projects) projectSize[p]);

// 制約条件
subject to {
  
  // 制約1: 各従業員は最大1つのプロジェクトにのみ配置
  forall(e in Employees)
    ctOneProjectPerEmployee:
      sum(p in Projects) assign[e][p] <= 1;
  
  // 制約2: プロジェクトに必要な最小人数を満たす
  forall(p in Projects)
    ctMinEmployees:
      sum(e in Employees) assign[e][p] >= minEmployees[p];
  
  // 制約3: プロジェクトの最大人数を超えない
  forall(p in Projects)
    ctMaxEmployees:
      sum(e in Employees) assign[e][p] <= maxEmployees[p];
  
  // 制約4: スキル要件を満たす（必要なスキルを持っていない従業員は配置できない）
  forall(e in Employees, p in Projects, s in Skills)
    ctSkillRequirement:
      assign[e][p] * projectRequiredSkill[p][s] <= 
        (employeeSkill[e][s] >= 1); // スキルレベルが1以上なら持っているとみなす
  
  // 制約5: プロジェクトサイズの計算
  forall(p in Projects)
    ctProjectSize:
      projectSize[p] == sum(e in Employees) assign[e][p];
  
  // 制約6: プロジェクトの総生産性を計算
  // 各従業員の生産性 = 必要スキルの平均レベル * 0.1 + 0.5
  forall(p in Projects)
    ctProjectProductivity:
      projectProductivity[p] == 
        sum(e in Employees) (
          assign[e][p] * (
            0.5 + (
              sum(s in Skills) (employeeSkill[e][s] * projectRequiredSkill[p][s])
            ) / (sum(s in Skills) projectRequiredSkill[p][s]) * 0.1
          )
        );
  
  // 制約7: プロジェクト期間内に完了できる
  // 総生産性 * プロジェクト期間 >= プロジェクト必要工数
  forall(p in Projects)
    ctDurationConstraint:
      projectProductivity[p] * projectDuration[p] >= projectWorkload[p];
}

// 結果の出力
execute DISPLAY {
  writeln("=== プロジェクト人員配置の最適解 ===");
  writeln();
  
  var totalSkillMatch = 0;
  var totalAssigned = 0;
  var maxSize = 0;
  var minSize = 999999;
  var sumSize = 0;
  
  // プロジェクトごとの配置状況
  for(var p in Projects) {
    var projectSkillScore = 0;
    var totalProductivity = 0;
    var assignedCount = 0;
    
    // 配置された従業員をカウントし、スキルスコアと生産性を計算
    for(var e in Employees) {
      if(assign[e][p] == 1) {
        assignedCount++;
        
        // このプロジェクトでのスキルマッチスコアを計算
        var employeeSkillSum = 0;
        var requiredSkillCount = 0;
        
        for(var s in Skills) {
          if(projectRequiredSkill[p][s] == 1) {
            projectSkillScore += employeeSkill[e][s];
            employeeSkillSum += employeeSkill[e][s];
            requiredSkillCount++;
          }
        }
        
        // 従業員の生産性を計算
        var avgSkill = requiredSkillCount > 0 ? employeeSkillSum / requiredSkillCount : 0;
        var productivity = 0.5 + avgSkill * 0.1;
        totalProductivity += productivity;
      }
    }
    
    // プロジェクトサイズの統計を更新
    if(assignedCount > maxSize) maxSize = assignedCount;
    if(assignedCount < minSize) minSize = assignedCount;
    sumSize += assignedCount;
    
    var completionTime = totalProductivity > 0 ? projectWorkload[p] / totalProductivity : 999999;
    var canComplete = completionTime <= projectDuration[p];
    
    writeln("プロジェクト ", p);
    writeln("  配置人数: ", assignedCount, " (最小: ", minEmployees[p], ", 最大: ", maxEmployees[p], ")");
    write("  配置された従業員: ");
    for(var e in Employees) {
      if(assign[e][p] == 1) {
        write(e, " ");
      }
    }
    writeln();
    writeln("  スキルマッチスコア: ", projectSkillScore);
    writeln("  総生産性: ", Opl.round(totalProductivity * 100) / 100, " 人日/日");
    writeln("  必要工数: ", projectWorkload[p], " 人日");
    writeln("  プロジェクト期間: ", projectDuration[p], " 日");
    writeln("  予想完了時間: ", Opl.round(completionTime * 100) / 100, " 日");
    writeln("  期間内完了: ", canComplete ? "可能" : "不可能");
    
    // 必要なスキルと配置された従業員のスキルを表示
    write("  必要スキル: ");
    for(var s in Skills) {
      if(projectRequiredSkill[p][s] == 1) {
        write("スキル", s, " ");
      }
    }
    writeln();
    
    // 各従業員のスキル詳細と生産性
    for(var e in Employees) {
      if(assign[e][p] == 1) {
        write("    従業員", e, " - スキル: ");
        var empSkillSum = 0;
        var empSkillCount = 0;
        for(var s in Skills) {
          if(projectRequiredSkill[p][s] == 1) {
            write("スキル", s, "=", employeeSkill[e][s], " ");
            empSkillSum += employeeSkill[e][s];
            empSkillCount++;
          }
        }
        var empAvgSkill = empSkillCount > 0 ? empSkillSum / empSkillCount : 0;
        var empProductivity = 0.5 + empAvgSkill * 0.1;
        writeln("(生産性: ", Opl.round(empProductivity * 100) / 100, ")");
      }
    }
    
    totalSkillMatch += projectSkillScore;
    totalAssigned += assignedCount;
    writeln();
  }
  
  // 配置されなかった従業員
  var unassignedCount = 0;
  write("配置されなかった従業員: ");
  for(var e in Employees) {
    var assigned = false;
    for(var p in Projects) {
      if(assign[e][p] == 1) {
        assigned = true;
        break;
      }
    }
    if(!assigned) {
      write(e, " ");
      unassignedCount++;
    }
  }
  if(unassignedCount > 0) {
    writeln();
  } else {
    writeln("なし");
  }
  writeln();
  
  // 人員配置の均等性を計算
  var avgSize = sumSize / numProjects;
  
  // サマリー
  writeln("=== サマリー ===");
  writeln("総スキルマッチスコア: ", totalSkillMatch);
  writeln("配置従業員数: ", totalAssigned, " / ", numEmployees);
  writeln();
  writeln("人員配置の均等性:");
  writeln("  平均人数: ", Opl.round(avgSize * 100) / 100);
  writeln("  最大人数: ", maxSize);
  writeln("  最小人数: ", minSize);
  writeln("  差: ", (maxSize - minSize));
}