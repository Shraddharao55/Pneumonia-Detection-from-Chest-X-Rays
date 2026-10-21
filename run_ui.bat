@echo off
title PneumoVision AI - Chest Radiograph Diagnostic System
cd /d "%~dp0"
echo ====================================================================
echo             PNEUMOVISION AI: CLINICAL DIAGNOSTIC SUITE
echo       Neural Networks & Deep Learning (NNDL) Open-Ended Project
echo ====================================================================
echo.
echo [1/2] Verifying Python environment...
python --version
echo.
echo [2/2] Launching Streamlit Web Application on http://localhost:8501 ...
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
