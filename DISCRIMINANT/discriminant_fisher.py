"""
SPSSの判別モデルをPythonで再現する
Fisherの線形判別関数（分類関数係数）を使った顧客グループ分類

参考記事: SPSSの判別モデルを外部プログラムで実装する
使用データ: input.csv（顧客1,000件、4グループ分類）
"""

import pandas as pd

# =============================================================================
# 1. Fisherの線形判別関数係数（SPSSの「分類関数係数」テーブルより）
# =============================================================================
FISHER_COEF = {
    "ベーシック サービス": {"収入": 0.002, "雇用": 0.101, "世帯人数": 1.184, "定数": -3.165},
    "E-サービス"        : {"収入": 0.003, "雇用": 0.118, "世帯人数": 1.276, "定数": -3.610},
    "プラス サービス"   : {"収入": 0.002, "雇用": 0.160, "世帯人数": 1.256, "定数": -4.028},
    "トータル サービス" : {"収入": 0.005, "雇用": 0.104, "世帯人数": 1.410, "定数": -3.992},
}


# =============================================================================
# 2. 単一レコードの分類
# =============================================================================
def 判別分類(収入, 雇用, 世帯人数):
    """
    Fisherの線形判別関数による顧客グループ分類
    生の値をそのまま渡す（標準化不要）

    Parameters
    ----------
    収入     : float
    雇用     : float  （雇用年数）
    世帯人数 : float

    Returns
    -------
    str : 予測グループ名
    """
    scores = {
        g: coef["収入"] * 収入 + coef["雇用"] * 雇用
           + coef["世帯人数"] * 世帯人数 + coef["定数"]
        for g, coef in FISHER_COEF.items()
    }
    return max(scores, key=scores.get)  # スコア最大のグループ


# =============================================================================
# 3. CSVファイルを読み込んで一括分類
# =============================================================================
def classify_csv(input_path, output_path):
    # --- 1. データ読み込み ---
    # 日本語環境のSPSS出力CSVはShift-JIS（cp932）が多い
    df = pd.read_csv(input_path, encoding='cp932')

    # 使用する列の確認
    print("--- 入力データ（先頭3件）---")
    print(df[['収入', '雇用', '世帯人数']].head(3))
    print()

    # --- 2. 欠損値チェック（SPSSと同様、欠損レコードはNullとして扱う）---
    required_cols = ['収入', '雇用', '世帯人数']
    valid_mask = df[required_cols].notna().all(axis=1)
    df["予測グループ"] = None  # デフォルトはNull

    # --- 3. 各グループのFisherスコアを計算（有効レコードのみ）---
    for group, coef in FISHER_COEF.items():
        df.loc[valid_mask, f"score_{group}"] = (
              coef["収入"]     * df.loc[valid_mask, "収入"]
            + coef["雇用"]     * df.loc[valid_mask, "雇用"]
            + coef["世帯人数"] * df.loc[valid_mask, "世帯人数"]
            + coef["定数"]
        )

    # --- 4. スコア最大のグループに分類（idxmax）---
    score_cols = {g: f"score_{g}" for g in FISHER_COEF}
    col_to_group = {v: k for k, v in score_cols.items()}

    df.loc[valid_mask, "予測グループ"] = (
        df.loc[valid_mask, list(score_cols.values())]
        .idxmax(axis=1)
        .map(col_to_group)
    )

    # --- 5. 作業列を削除して結果を出力 ---
    df.drop(columns=list(score_cols.values()), inplace=True)
    df.to_csv(output_path, encoding='cp932', index=False)

    print("--- 分類結果（先頭3件）---")
    print(df[['収入', '雇用', '世帯人数', '予測グループ']].head(3))
    print()

    # --- 6. SPSSモデルとの一致率（$D-顧客 列がある場合）---
    if "$D-顧客" in df.columns:
        match = (df["予測グループ"] == df["$D-顧客"]).sum()
        total = len(df)
        print(f"--- SPSSモデルとの一致率 ---")
        print(f"{match} / {total}（{match / total * 100:.1f}%）")
        print()

    return df


# =============================================================================
# メイン
# =============================================================================
if __name__ == "__main__":
    # 単一レコード分類の例
    print("--- 単一レコード分類の例 ---")
    result = 判別分類(収入=80, 雇用=12, 世帯人数=3)
    print(f"判別分類(収入=80, 雇用=12, 世帯人数=3) → '{result}'")
    print()

    # CSVファイル一括分類
    classify_csv('input.csv', 'output.csv')
    print("output.csv に出力しました。")
