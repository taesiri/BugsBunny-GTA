import win32gui
import win32api
import win32con
import time
import logging
from datetime import datetime

def setup_logging():
    """Configure logging with timestamp"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gta_f12_sender.log'),
            logging.StreamHandler()
        ]
    )

def find_gta_window():
    """Find GTA V window and return its handle"""
    hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
    if hwnd:
        logging.info("GTA V window found")
        return hwnd
    logging.warning("GTA V window not found")
    return None

def send_f12_key(hwnd):
    """Send F12 key to GTA V window"""
    try:
        if hwnd:
            # Make GTA window active
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.1)  # Give time for window to become active
            
            # Send F12 key
            win32api.keybd_event(win32con.VK_F12, 0, 0, 0)  # Key down
            time.sleep(0.1)
            win32api.keybd_event(win32con.VK_F12, 0, win32con.KEYEVENTF_KEYUP, 0)  # Key up
            
            # Reposition window after key press
            win32gui.SetWindowPos(hwnd, None, 0, 0, 1920, 1080, 0)
            
            logging.info("F12 key sent successfully")
            return True
    except Exception as e:
        logging.error(f"Error sending F12 key: {str(e)}")
    return False

def main():
    setup_logging()
    logging.info("Starting GTA V F12 sender script")
    
    interval = 30  # seconds between F12 key presses
    
    try:
        while True:
            hwnd = find_gta_window()
            if hwnd:
                send_f12_key(hwnd)
            else:
                logging.warning("Waiting for GTA V window...")
            
            # Wait for next interval
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logging.info("Script stopped by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    finally:
        logging.info("Script terminated")

if __name__ == "__main__":
    main() 