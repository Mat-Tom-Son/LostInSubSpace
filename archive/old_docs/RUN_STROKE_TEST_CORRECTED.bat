@echo off
echo ================================================================================
echo STROKE TEST - CORRECTED VERSION (d=64, noise=0.3)
echo ================================================================================
echo.
echo Previous attempt failed because:
echo   - d=40 was too constrained (victim only 76%%, not 95%%)
echo   - noise=0.15 was too weak (only -1%% drop)
echo   - No crisis = No reflex
echo.
echo This version:
echo   1. Trains victim at d=64 for 50 epochs (should reach ~95%%)
echo   2. Applies VIOLENT stroke (noise=0.3, expect 95%% -^> 40%%)
echo   3. Watches for amplitude spike (2x baseline)
echo.
echo Expected runtime: ~15-20 minutes (50 epochs + 5 epochs)
echo.
echo Press Ctrl+C to cancel, or any key to continue...
pause >nul

python run_stroke_test_corrected.py

echo.
echo ================================================================================
echo Done! Check stroke_test_logs/ for detailed output
echo ================================================================================
pause
