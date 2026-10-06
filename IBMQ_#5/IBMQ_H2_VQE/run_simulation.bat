@echo off
chcp 65001 > nul
echo ============================================================
echo   H2 基底エネルギー計算 -- シミュレーション版
echo ============================================================
echo.
set PYTHONUTF8=1
"C:\Python\Python311\python.exe" "%~dp0src\h2_simulation.py"
echo.
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 実行中にエラーが発生しました。
) else (
    echo [完了] 結果は results\ フォルダに保存されました。
)
echo.
pause
