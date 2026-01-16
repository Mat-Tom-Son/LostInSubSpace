@echo off
echo ================================================================================
echo STROKE TEST - One Command Execution
echo ================================================================================
echo.
echo This will:
echo   1. Train healthy victim (d=40, 20 epochs) - takes ~3-5 minutes
echo   2. Apply stroke (load + constraint + noise) - takes ~1-2 minutes
echo   3. Analyze results and detect reflex
echo.
echo Press Ctrl+C to cancel, or any key to continue...
pause >nul

python run_stroke_test.py

echo.
echo ================================================================================
echo Done! Check stroke_test_logs/ for detailed output
echo ================================================================================
pause
