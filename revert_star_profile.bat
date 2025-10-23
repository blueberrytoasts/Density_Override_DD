@echo off
REM Revert star profile implementation to original percentile-based method

echo Reverting to original metal detection (percentile-based)...
echo.

if exist "app\core\metal_detection.py.before_star_profile" (
    copy /Y "app\core\metal_detection.py.before_star_profile" "app\core\metal_detection.py" >nul
    echo [SUCCESS] Reverted metal_detection.py
) else (
    echo [ERROR] Backup file not found: app\core\metal_detection.py.before_star_profile
    exit /b 1
)

echo.
echo Revert complete! The system is back to percentile-based metal detection.
echo Restart the Streamlit app to see changes.
echo.
pause
