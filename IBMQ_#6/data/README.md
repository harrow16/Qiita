# data/ ディレクトリ

## prediction_results.csv

**このファイルはリポジトリにサンプルデータとして同梱されています。**

本プロジェクトのスコープは「**株価予測が完了した後、最適化部分を CPLEX から量子コンピューター（QAOA / VQE）に置き換える**」ことです。
LightGBM による株価予測は別プロジェクト（`20260420`）で実施済みであり、その出力結果をそのまま使います。

> クローン後すぐに `python run_quantum_optimization.py` を実行できます。

### フォーマット

| カラム | 型 | 説明 |
|---|---|---|
| `Ticker` | string | 銘柄コード（例: `6976.T`）|
| `Current_Price` | float | 現在の株価（円）|
| `Predicted_Return` | float | 5営業日後の予測収益率（例: `0.0299` = 2.99%）|
| `Sector` | string | セクター名（例: `Technology`）|
| `Name` | string | 企業名（例: `Taiyo Yuden Co., Ltd.`）|

### サンプル（先頭4行）

```csv
Ticker,Current_Price,Predicted_Return,Sector,Name
6976.T,4160.0,0.029916368057598862,Technology,"Taiyo Yuden Co., Ltd."
1605.T,4687.0,0.027655653000458576,Energy,Inpex Corporation
9501.T,623.4,0.027292516561146053,Utilities,"Tokyo Electric Power Company Holdings, Incorporated"
4005.T,507.4,0.027134979857798596,Basic Materials,"Sumitomo Chemical Company, Limited"
```

### 予測を更新したい場合

別プロジェクト（`20260420`）の LightGBM モデルを実行し、出力を上書きしてください。

```bash
cd ../20260420/models
python stock_prediction.py
cp ../20260420/data/prediction_results.csv ../IBMQ_#6/data/prediction_results.csv
```

## セクター一覧

対象の119銘柄は以下のセクターに分類されています：

| セクター | 説明 |
|---|---|
| Technology | テクノロジー |
| Energy | エネルギー |
| Utilities | 公益事業 |
| Basic Materials | 素材 |
| Industrials | 資本財 |
| Financial Services | 金融サービス |
| Healthcare | ヘルスケア |
| Consumer Defensive | 生活必需品 |
| Consumer Cyclical | 一般消費財 |
| Communication Services | 通信サービス |
| Real Estate | 不動産 |
