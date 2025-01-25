@echo off
setlocal enabledelayedexpansion

set BASE_SAVE_DIR=C:\\Workspace\\gta_data_export\\third_person_capture_

for /l %%i in (4,1,6) do (
    echo Running iteration %%i...
    set "SAVE_DIR=!BASE_SAVE_DIR!%%i"
    python BugsBunny-Getdata-FlyingCar.py --random_location --camera_preset third_person --frames_to_capture 2000 --save_dir "!SAVE_DIR!" --weather random
    echo Waiting for 10 seconds before the next run...
    timeout /t 10 /nobreak >nul
)

echo All iterations completed.
pause