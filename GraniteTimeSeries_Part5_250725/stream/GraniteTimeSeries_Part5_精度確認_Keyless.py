#-------------------------------------------------
# IBM Granite Time Series ( Tiny Time Mixer )
# Watsonx.ai 上の Granite Time Series を利用
# 精度確認用シンタックス - Python版
#-------------------------------------------------
#-----------------------------------------------
# ライブラリインポート
#-----------------------------------------------
# Modeler用ライブラリ
import modelerpy

# すべての警告を無視
import warnings
warnings.filterwarnings("ignore")

# Pandas & Numpy
import numpy as np
import pandas as pd
import pprint

# scikit-learn - MAPE計算用
from sklearn.metrics import mean_absolute_percentage_error

# IBM Watsonx.ai ライブラリ
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials

# Granite Time Series Model on IBM Watsonx.ai ライブラリ
from ibm_watsonx_ai.foundation_models import TSModelInference
from ibm_watsonx_ai.foundation_models.schema import TSForecastParameters

#-----------------------------------------------
# API Key などの準備
#-----------------------------------------------
WATSONX_APIKEY = "YOUR API KEY"
WATSONX_PROJECT_ID = "YOUR PROJECT ID"
WATSONX_URL = "YOUR URL"

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
# Watsonx.ai で利用できるモデルの確認
#-----------------------------------------------
for model in client.foundation_models.get_time_series_model_specs()["resources"]:
    pprint.pp("--------------------------------------------------")
    pprint.pp(f'model_id: {model["model_id"]}')
    pprint.pp(f'functions: {model["functions"]}')
    pprint.pp(f'long_description: {model["long_description"]}')
    pprint.pp(f'label: {model["label"]}')


#-----------------------------------------------
# データ入力 
#-----------------------------------------------
#ModelerからPandasでデータを入力
modelerData = modelerpy.readPandasDataframe()
input_df = modelerData

#-----------------------------------------------
# カラム定義
#-----------------------------------------------
# タイムスタンプカラム定義
timestamp_column = "time"
# 予測対象カラム名
target_column = "total load actual"

#-----------------------------------------------
# モデルで使用するカラムの指定
#-----------------------------------------------
# 予測対象カラムをリストで定義
target_columns = [ target_column ]

# 時系列カラム"time"を文字列型に変更
# Watsonx の裏側では、HTTP リクエストを通じてモデルにデータを渡すため、
# JSON形式に変換可能なデータ（文字列、数値、配列など）しか渡せません。
input_df[ timestamp_column ] = input_df[ timestamp_column ] .astype(str)

#-----------------------------------------------
# モデルの学習・予測レコード数定義
#-----------------------------------------------
# モデルが「過去何ステップ分のデータを見て」学習・予測するか（ここでは過去512時間分）
context_length = 512  # the max context length for the 512-96 model
# 予測対象（"total load actual"）
prediction_length = 96  # the max forecast length for the 512-96 model

#-----------------------------------------------
# モデルの予測・検証用データの準備
#-----------------------------------------------
# 精度検証用データ - 最後から96レコードを検証用にとっておく
future_data = input_df.iloc[-prediction_length:,]
# 予測用データ - 最後から96レコードを除いた512レコードをモデルに投入
train_data = input_df.iloc[-context_length-prediction_length:-prediction_length,]

#-----------------------------------------------
# モデル用パラメータ定義
#-----------------------------------------------
# モデル用パラメータの指定
forecasting_params = TSForecastParameters(
    id_columns=[],                       # 複数系列を識別するためのID（今回は1系列なので空）
    timestamp_column=timestamp_column,   # 時間
    freq="1h",                           # 1時間単位
    target_columns=target_columns,       # 予測対象（"total load actual"）
    prediction_length=prediction_length, #予測する長さ <96レコード先まで予測>
)

#-----------------------------------------------
# モデルの実行 - APIで実行
#-----------------------------------------------
#モデルの指定
ts_model_id = client.foundation_models.TimeSeriesModels.GRANITE_TTM_512_96_R2
# モデルインスタンスの初期化と設定
ts_model = TSModelInference(model_id=ts_model_id, api_client=client)
# Watsonx.ai Granite TimeSeries の実行 <予測用データ> - 512レコードを入力して96レコードを予測
results = ts_model.forecast(data=train_data, params=forecasting_params)['results'][0]

#-----------------------------------------------
# 予測精度の確認
#-----------------------------------------------
# 予測結果をデータフレームに取り込み - 予測した96レコードが格納されている。
watsonx_gts_forecast = pd.DataFrame(results)

#最後 検証用にとっておいた 96レコードと予測した96レコードでMAPEを計算
mape = mean_absolute_percentage_error(future_data[target_column], watsonx_gts_forecast[target_column]) * 100
print("mape = ", mape)
