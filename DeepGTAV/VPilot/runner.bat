@echo off
setlocal

for /l %%i in (1,1,10) do (
    echo Running iteration %%i...
    python BugsBunny-Getdata-FlyingCar.py --random_location --camera_preset third_person --frames_to_capture 100
    echo Waiting for 10 seconds before the next run...
    timeout /t 10 /nobreak >nul
)

echo All iterations completed.
pause