@echo off
echo ========================================
echo   Running Dynamic Pricing Scraper
echo   %date% %time%
echo ========================================

cd /d "F:\Dynamic Pricing\dp_pipeline\src"

REM Activate virtual environment if you have one
REM call venv\Scripts\activate

REM Run the scraper
python scraper.py --single-run

echo.
echo Scraper finished at %time%
pause
