@echo off
setlocal enabledelayedexpansion

set BASE_SAVE_DIR=C:\\Workspace\\gta_data_export\\x_capture_

for /l %%i in (1,1,40) do (
    echo Running iteration %%i...
    set "SAVE_DIR=!BASE_SAVE_DIR!%%i"
    python BugsBunny-Getdata-FlyingCar.py --random_location --camera_preset first_person --frames_to_capture 250 --save_dir "!SAVE_DIR!"
    echo Waiting for 10 seconds before the next run...
    timeout /t 10 /nobreak >nul
)

echo All iterations completed.
pause