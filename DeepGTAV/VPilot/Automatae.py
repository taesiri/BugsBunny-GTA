import subprocess
import time
import psutil
import os
import sys

def kill_gta5():
    """Kill GTA5.exe if it is running."""
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] and proc.info['name'].lower() == 'gta5.exe':
            print(f"Killing GTA5.exe (PID: {proc.info['pid']})")
            proc.kill()

def main():
    # 1. Launch GTA5 via Steam
    print("Launching GTA5 via Steam...")
    # On Windows, you can call 'start steam://rungameid/271590' using cmd
    # or you can use a direct open command as shown below.
    subprocess.Popen(["cmd", "/c", "start", "steam://rungameid/271590"], shell=True)
    
    # 2. Wait for 40 seconds for GTA5 to load
    print("Waiting 40 seconds for GTA5 to load...")
    time.sleep(40)
    
    # 3. Invoke the capture script
    #    Adjust the arguments to match your environment
    capture_script = "BugsBunny-Getdata-buzzard.py"
    capture_args = [
        "python", capture_script,
        "--host", "127.0.0.1",
        "--port", "8000",
        "--save_dir", "C:\\Workspace\\export_data\\sample",
        "--loc_x", "200",
        "--loc_y", "10",
        "--base_height", "15",
        "--current_height", "3",
        "--weather", "CLEAR",
        "--time_hour", "17",
        "--time_min", "30",
        "--frames_to_capture", "50",
        "--cam_y", "5.0",
        "--cam_z", "2.0",
        "--rot_x", "10",
        "--rot_y", "0"
    ]
    print("Running the capture script...")
    process = subprocess.Popen(capture_args)
    
    # 4. Wait for the capture script to finish
    print("Waiting for the capture script to finish...")
    process.wait()
    print("Capture script has finished.")
    
    # 5. Kill GTA5.exe
    print("Killing GTA5.exe if running...")
    kill_gta5()
    print("Done.")

if __name__ == "__main__":
    main()
