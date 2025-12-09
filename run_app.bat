@echo off
REM ==============================
REM 1. Ir a la carpeta del proyecto
REM ==============================
cd /d "%~dp0"

REM ==============================
REM 2. Lanzar Streamlit con el python por defecto
REM ==============================
python -m streamlit run app.py

REM ==============================
REM 3. Mantener la ventana abierta
REM ==============================
echo.
pause