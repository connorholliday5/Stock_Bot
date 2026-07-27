@echo off
REM ============================================================
REM start_bot.bat - launch the bot with its venv activated.
REM
REM Manual use:  double-click, or run  .\start_bot.bat
REM
REM Auto-start on boot (survives Windows Update reboots):
REM   1. Start menu -> "Task Scheduler" -> Create Basic Task
REM   2. Name: Stock Bot
REM   3. Trigger: "When the computer starts"
REM   4. Action: "Start a program" -> browse to this file
REM   5. Finish, then open the task's Properties and tick
REM      "Run whether user is logged on or not"
REM   Also set Settings -> System -> Power -> Sleep = Never.
REM ============================================================

cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo [start_bot] no venv found - using system python
)

echo [start_bot] starting Stock Bot from %CD%
python main.py

REM Keep the window open if the bot exits, so the reason stays readable.
echo.
echo [start_bot] bot exited with code %ERRORLEVEL%
pause
