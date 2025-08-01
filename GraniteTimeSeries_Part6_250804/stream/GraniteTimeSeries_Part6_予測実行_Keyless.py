#-------------------------------------------------
# IBM Granite Time Series ( Tiny Time Mixer )
# Watsonx.ai上の Granite Time Series を利用
# Modeler連携 - 予測値取得シンタックス
#-------------------------------------------------
#-----------------------------------------------
# ライブラリ導入パート
# 　・データモデル定義時のために必要最低限を導入
#-----------------------------------------------
# Modeler用ライブラリ
import modelerpy

# すべての警告を無視
import warnings
warnings.filterwarnings("ignore")

#-----------------------------------------------
# カラム定義
#-----------------------------------------------
# タイムスタンプカラム定義
timestamp_column = "time"
# 予測対象カラム名
target_column = "total load actual"

#-----------------------------------------------
# データモデル定義パート
#-----------------------------------------------
if modelerpy.isComputeDataModelOnly():
    #データモデル取得
    modelerDataModel = modelerpy.getDataModel()
    #特に変更しない
    modelerpy.setOutputDataModel(modelerDataModel)

#-----------------------------------------------
# 時系列予測値取得パ―ド
#-----------------------------------------------
else:

    #-----------------------------------------------
    # モデル作成用ライブラリのインポート
    #-----------------------------------------------
    # Pandas & Numpy
    import numpy as np
    import pandas as pd

    # IBM Watsonx.ai ライブラリ
    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai import Credentials

    # Granite Time Series Forcasting on IBM Watsonx.ai ライブラリ
    from ibm_watsonx_ai.foundation_models import TSModelInference
    from ibm_watsonx_ai.foundation_models.schema import TSForecastParameters

    #-----------------------------------------------
    # データ入力 
    #-----------------------------------------------
    #ModelerからPandasでデータを入力
    modelerData = modelerpy.readPandasDataframe()
    input_df = modelerData

    #-----------------------------------------------
    # API Key などの準備
    #-----------------------------------------------
    WATSONX_APIKEY = "YOUR API KEY"
    WATSONX_PROJECT_ID = "YOUR PROJECT ID"
    WATSONX_URL = "YOUR ENDPOINT URL"

    #-----------------------------------------------
    # Watsonx.ai ランタイムへの接続情報設定
    #-----------------------------------------------
    # watsonx.aiランタイムへの接続情報を定義
    credentials = Credentials(
        url = WATSONX_URL,
        api_key = WATSONX_APIKEY,
    )

    # 接続情報の設定
    # クレデンシャルのセット
    client = APIClient(credentials)
    # プロジェクトのセット
    client.set.default_project( WATSONX_PROJECT_ID )

    #-----------------------------------------------
    # 学習データレコード数、予測レコード数設定
    #-----------------------------------------------
    # モデルが「過去何ステップ分のデータを見て」学習・予測するか（ここでは過去512時間分）
    context_length = 512  # the max context length for the 512-96 model
    # 予測対象（"total load actual"）
    prediction_length = 96  # the max forecast length for the 512-96 model

    #-----------------------------------------------
    # モデルカラム定義
    #-----------------------------------------------
    # 予測対象カラムをリストで定義
    target_columns = [ target_column ]

    # 時系列カラム"time"を文字列型に変更
    # Watsonx の裏側では、HTTP リクエストを通じてモデルにデータを渡すため、
    # JSON形式に変換可能なデータ（文字列、数値、配列など）しか渡せません。
    input_df[ timestamp_column ] = input_df[ timestamp_column ] .astype(str)

    #-----------------------------------------------
    # モデルの予測・検証用データの準備
    #-----------------------------------------------
    # 予測用データ - 最後から512レコードを使う
    future_data = input_df.iloc[-context_length:,]

    #-----------------------------------------------
    # モデルパラメータ定義
    #-----------------------------------------------
    forecasting_params = TSForecastParameters(
        id_columns=[],                       # 複数系列を識別するためのID（今回は1系列なので空）
        timestamp_column=timestamp_column,   # 時間
        freq="1h",                           # 時間単位 - 1時間ごとのデータ
        target_columns=target_columns,       # 予測対象（"total load actual"）
        prediction_length=prediction_length, # 予測する長さ 
    )

    #-----------------------------------------------
    # Watsonx.aiモデルのインスタンス化
    #-----------------------------------------------
    #モデルの指定 - 512レコードを使い96レコード先を予測するモデルを指定
    ts_model_id = client.foundation_models.TimeSeriesModels.GRANITE_TTM_512_96_R2
    # モデルインスタンスの初期化と設定 - APIKEYやProject IDをここで使用
    ts_model = TSModelInference(model_id=ts_model_id, api_client=client)

    #-----------------------------------------------
    # モデル実行
    #-----------------------------------------------
    # 予測用データ(最後から512レコード)でモデル実行
    results = ts_model.forecast(data=future_data, params=forecasting_params)['results'][0]
    # 予測結果をデータフレームに取り込み - 予測値 96 レコードが格納されている
    watsonx_gts_forecast = pd.DataFrame(results)

    #-----------------------------------------------
    # 入力データに予測値を追加して戻す
    #-----------------------------------------------
    # それぞれのtimeのカラムをtime stampに変換してから追加する
    modelerData[ timestamp_column ] = pd.to_datetime(modelerData[ timestamp_column ])
    watsonx_gts_forecast[ timestamp_column ] = pd.to_datetime(watsonx_gts_forecast[ timestamp_column ])

    # 追加
    df_combined = pd.concat([modelerData, watsonx_gts_forecast], ignore_index=True)
  
    # Modelerにデータを戻す
    modelerpy.writePandasDataframe(df_combined)