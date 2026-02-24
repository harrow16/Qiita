#--------------------------------------------------------------
# SPSS Modeler Extension Output Node
# Python Syntax 
# Partial Correlation (All Variables)
#--------------------------------------------------------------

#=============================
# ライブラリ導入
#=============================
import modelerpy
import pandas as pd
import numpy as np

#==============================
# データ入力
#==============================
mdf = modelerpy.readPandasDataframe()

#==============================
# 説明変数設定（数値列のみ抽出）
#==============================
mdf_numeric = mdf.select_dtypes(include=['number'])

#==============================
# 偏相関係数の計算関数
#==============================
def get_partial_corr_matrix(indf):
    # 相関行列の計算
    corr_mat = indf.corr()

    try:
        # 逆行列（精度行列）の計算
        # 多重共線性がある場合はここでエラーになる
        inv_corr = np.linalg.inv(corr_mat.values)
    except np.linalg.LinAlgError:
        return None
    
    # 対角成分の平方根（分母の計算用）
    d = np.sqrt(np.diag(inv_corr))

    # 偏相関係数の計算: -w_ij / sqrt(w_ii * w_jj)
    partial_corr = -inv_corr / np.outer(d, d)

    # 対角成分を1.0に修正
    np.fill_diagonal(partial_corr, 1.0)
    
    return pd.DataFrame(partial_corr, index=corr_mat.index, columns=corr_mat.columns)

#==================================
# 計算の実行
#===================================
pcorr_matrix = get_partial_corr_matrix(mdf_numeric)

#===================================
# 全カラムを対象とした結果出力
#===================================
if pcorr_matrix is not None:
    print("=== Partial Correlation Analysis (All Columns) ===")
    
    # 数値列のリストをループで回す
    for target in mdf_numeric.columns:
        print(f"\nTarget Variable: [ {target} ]")
        print("-" * 40)
        
        # 自身を除外して降順（正の相関が強い順）にソート
        result = pcorr_matrix[target].drop(target).sort_values(ascending=False)
        
        # 結果の表示
        if result.empty:
            print("No other numeric variables to compare.")
        else:
            print(result)
        print("-" * 40)
else:
    print("Calculation Error: Singular matrix (multicollinearity).")
