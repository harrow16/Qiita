@echo off
chcp 65001 > nul
echo ============================================================
echo   H2 基底エネルギー計算 -- IBM Quantum 実機版
echo ============================================================
echo.
echo 実機に接続してジョブを投入します。
echo キュー待ちによっては数分〜数十分かかります。
echo.
set PYTHONUTF8=1
"C:\Python\Python311\python.exe" "%~dp0src\h2_realdevice.py"
echo.
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 実行中にエラーが発生しました。
    echo         .env ファイルに IBMQ_API_KEY が設定されているか確認してください。
) else (
    echo [完了] 結果は results\ フォルダに保存されました。
)
echo.
pause
