@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run src/dashboard.py --server.headless true --server.port 8511
) else (
    "D:\codex\Excel-\trade_dashboard\.venv\Scripts\python.exe" -m streamlit run src/dashboard.py --server.headless true --server.port 8511
)
