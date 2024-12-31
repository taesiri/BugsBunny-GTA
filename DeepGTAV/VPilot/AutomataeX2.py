import subprocess
import time
import psutil
import os
import sys
import random
import win32gui
import win32con
import win32process
import win32com.client

def kill_gta5():
    """Kill GTA V process if running"""
    try:
        # Look for both possible process names
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] in ['GTA5.exe', 'GTAV.exe']:
                print(f"Killing {proc.info['name']}...")
                proc.kill()
                proc.wait()  # Wait for the process to actually terminate
                return True
        print("GTA V process not found")
        return False
    except Exception as e:
        print(f"Error killing GTA V: {e}")
        return False

def kill_notepad():
    """Kill Notepad if it is running."""
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] and proc.info['name'].lower() == 'notepad.exe':
                try:
                    print(f"Killing Notepad.exe (PID: {proc.info['pid']})")
                    proc.kill()
                    proc.wait(timeout=3)  # Wait up to 3 seconds for the process to terminate
                except (psutil.NoSuchProcess, psutil.TimeoutExpired) as e:
                    print(f"Process already terminated or timeout reached: {e}")
                except Exception as e:
                    print(f"Error killing Notepad: {e}")
    except Exception as e:
        print(f"Error iterating processes: {e}")

# Add list of valid weather conditions
WEATHER_CONDITIONS = [
    'EXTRASUNNY', 'CLEAR', 'CLOUDS', 'SMOG', 'FOGGY', 'OVERCAST', 
    'RAIN', 'THUNDER', 'CLEARING', 'NEUTRAL', 'SNOW', 'BLIZZARD', 
    'SNOWLIGHT'
]

def get_random_location():
    """Generate random location coordinates within GTA V map bounds"""
    # Approximate GTA V map bounds
    x = random.uniform(-4000, 4000)
    y = random.uniform(-4000, 4000)
    base_height = random.uniform(10, 30)
    return x, y, base_height

def list_window_titles():
    """List all visible window titles for debugging"""
    def callback(hwnd, titles):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                titles.append(title)
        return True
    
    titles = []
    win32gui.EnumWindows(callback, titles)
    # print list of titles
    print("Available windows:")
    for title in titles:
        print(f"  - {title}")

    return titles

def focus_window(window_title, partial_match=True, duration=0.5):
    """Focus on a window by its title, then unfocus"""
    target_hwnd = None
    
    def window_enum_handler(hwnd, results):
        nonlocal target_hwnd
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            matches = (
                (partial_match and window_title.lower() in window_text.lower()) or
                (not partial_match and window_title.lower() == window_text.lower())
            )
            if matches:
                target_hwnd = hwnd
                print(f"Found window: '{window_text}'")  # Debug print
                return False
        return True
    
    # Find the window
    win32gui.EnumWindows(window_enum_handler, None)
    
    # If window found, focus it
    if target_hwnd:
        try:
            # Show and restore the window
            win32gui.ShowWindow(target_hwnd, win32con.SW_RESTORE)
            
            # Use WScript.Shell to force focus
            shell = win32com.client.Dispatch("WScript.Shell")
            win32gui.SetForegroundWindow(target_hwnd)
            shell.SendKeys('%')  # Send Alt key to force focus
            
            time.sleep(duration)
            
            # Return focus to desktop (optional)
            win32gui.SetForegroundWindow(win32gui.GetDesktopWindow())
        except Exception as e:
            print(f"Error focusing window: {e}")
    else:
        print(f"Window with title '{window_title}' not found!")
        print("Available windows:")
        for title in list_window_titles():
            print(f"  - {title}")

def main():
    # Number of capture rounds
    num_rounds = 100  # Adjust this number as needed

    for round_idx in range(num_rounds):
        # Launch GTA5 via Steam
        print(f"\nStarting capture round {round_idx + 1}/{num_rounds}")
        print("Launching GTA5 via Steam...")
        subprocess.Popen(["cmd", "/c", "start", "steam://rungameid/271590"], shell=True)
        
        # Wait for initial GTA5 load
        print("Waiting 25 seconds for initial GTA5 load...")
        time.sleep(25)
        
        # Focus and unfocus GTA V window
        print("Focusing GTA V window...")
        focus_window("Grand Theft Auto V")
        focus_window("GTAV")
        focus_window("GTA5")
        
        # Wait for full load
        print("Waiting 15 more seconds for GTA5 to fully load...")
        time.sleep(15)

        # Generate random location and weather
        loc_x, loc_y, base_height = get_random_location()
        weather = random.choice(WEATHER_CONDITIONS)
        
        # Create indexed directory for this round
        save_dir = f"C:\\Workspace\\export_data\\record_2\sample_{round_idx}"
        os.makedirs(save_dir, exist_ok=True)

        # Open Notepad with a dummy file
        print("Opening Notepad...")
        subprocess.Popen(["notepad.exe"])
        time.sleep(2)  # Wait for Notepad to open

        # Focus on Notepad window
        print("Focusing on Notepad window...")
        focus_window("Notepad")
        time.sleep(2)
        focus_window("Untitled - Notepad")  # Try alternative title
        time.sleep(2)

        capture_args = [
            "python", "BugsBunny-Getdata-buzzard.py",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--save_dir", save_dir,
            "--loc_x", str(loc_x),
            "--loc_y", str(loc_y),
            "--base_height", str(base_height),
            "--current_height", "3",
            "--weather", weather,
            "--time_hour", str(random.randint(0, 23)),
            "--time_min", str(random.randint(0, 59)),
            "--frames_to_capture", "50",
            "--cam_y", "5.0",
            "--cam_z", "2.0",
            "--rot_x", "10",
            "--rot_y", "0"
        ]
        
        print(f"Location: ({loc_x:.1f}, {loc_y:.1f}, {base_height:.1f})")
        print(f"Weather: {weather}")
        print(f"Saving to: {save_dir}")
        
        process = subprocess.Popen(capture_args)
        process.wait()
        print(f"Completed capture round {round_idx + 1}")

        # Kill both GTA5 and Notepad after capture
        print("Killing GTA5.exe and Notepad.exe...")
        kill_gta5()
        kill_notepad()
        
        # Wait between rounds
        print("Waiting 120 seconds for GTA to fully close...")
        time.sleep(120)

    print("All capture rounds completed.")

if __name__ == "__main__":
    main()
