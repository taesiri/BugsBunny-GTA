#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import cv2
import matplotlib
import logging
import traceback
import os
import base64
import numpy as np
import open3d
import win32gui
from tqdm import tqdm
from PIL import Image
from random import uniform
from math import sqrt

from deepgtav.messages import (
    Start, Stop, Scenario, Dataset, frame2numpy,
    GoToLocation, SetCameraPositionAndRotation,
    StartRecording, StopRecording, SetClockTime, SetWeather
)
from deepgtav.client import Client

from utils.Constants import IMG_WIDTH, IMG_HEIGHT
from utils.BoundingBoxes import (
    add_bboxes, parseBBoxesVisDroneStyle, parseBBox_YoloFormatStringToImage
)
from utils.utils import (
    save_image_and_bbox, save_meta_data,
    getRunCount
)

matplotlib.use('Agg')  # Set the backend to non-interactive before importing pyplot

###############################################################################
# Configuration Constants
###############################################################################

WEATHER_CONDITIONS = [
    'CLEAR', 'EXTRASUNNY', 'CLOUDS', 'OVERCAST',
    'RAIN', 'FOGGY',  'CLEARING'
]

# For demo, let's keep it small
WEATHER_CONDITIONS = [
    'THUNDER'
]

TIME_PERIODS = [
    (12, 0),  # Noon
]

# Define multiple locations with their heights (no location ID used)
LOCATIONS = [
    # x, y, base_height, list_of_heights
    (-33, 1, 135, [15, 15, 7]),      # Single location for testing
]

# Use only one camera position
CAMERA_POSITION = {
    'y': 4.5,
    'z': 1.8,
    'rot_x': 0,
    'rot_y': 0
}

###############################################################################
# Helper Functions
###############################################################################

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gtav_capture.log'),
            logging.StreamHandler()
        ]
    )

def move_gta_window():
    """Find and move GTA V window to (0,0) with size 1920x1080."""
    hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
    if hwnd:
        win32gui.SetWindowPos(hwnd, None, 0, 0, 1920, 1080, 0)
        return True
    return False

def setup_directories(base_dir):
    """Create all necessary directories for data storage."""
    directories = [
        'images', 'labels', 'meta_data',
        'image', 'depth', 'StencilImage',
        'SegmentationAndBBox', 'semantic_vis', 'LiDAR'
    ]
    for dir_name in directories:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def process_visualization(message, args, filename, bbox_image=None):
    """Handle visualization windows and saving visualization data."""
    try:
        if message["segmentationImage"] is None:
            logging.warning("Segmentation image is None")
            return
            
        if message["segmentationImage"] == "":
            logging.warning("Segmentation image is empty")
            return

        nparr = np.frombuffer(base64.b64decode(message["segmentationImage"]), np.uint8)
        segmentationImage = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        
        if segmentationImage is None:
            logging.error("Failed to decode segmentation image")
            return

        # Create and position preview windows
        overlay = cv2.addWeighted(bbox_image, 0.5, segmentationImage, 0.5, 0.0)
        windows = {
            "Original with BBoxes": bbox_image,
            "Semantic Segmentation": segmentationImage,
            "Overlay": overlay
        }

        for idx, (name, img) in enumerate(windows.items()):
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 640, 360)
            # Offsets for a second monitor, adjust as needed
            cv2.moveWindow(name, 1920 + (50 if idx < 2 else 740), 50 + (410 * (idx % 2)))
            cv2.imshow(name, img)

        # Save visualization files
        cv2.imwrite(os.path.join(args.save_dir, "image", f"{filename}.png"), bbox_image)
        cv2.imwrite(
            os.path.join(args.save_dir, "SegmentationAndBBox", f"{filename}.png"),
            overlay
        )

    except Exception as e:
        logging.error(f"Error in process_visualization: {str(e)}\n{traceback.format_exc()}")

def process_lidar(message, args, filename):
    """Process and save LiDAR data."""
    if message["LiDAR"] != None and message["LiDAR"] != "":
        points_data = np.frombuffer(base64.b64decode(message["LiDAR"]), np.float32)
        points3d = np.delete(points_data.reshape((-1, 4)), 3, 1)

        # Create color gradient based on height
        z_norm = (points3d[:, 0] - points3d[:, 0].min()) / (points3d[:, 0].max() - points3d[:, 0].min())
        colors = np.zeros((points3d.shape[0], 3))
        colors[:, 0] = z_norm
        colors[:, 2] = 1 - z_norm

        # Create and save point cloud
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(points3d)
        point_cloud.colors = open3d.utility.Vector3dVector(colors)
        open3d.io.write_point_cloud(
            os.path.join(args.save_dir, "LiDAR", filename.replace('.png', '.ply')),
            point_cloud
        )

###############################################################################
# Capture Function
###############################################################################

def capture_data_for_configuration(
    client,
    args,
    run_count,
    loc_x,
    loc_y,
    base_height,
    current_height,
    weather,
    time_hour,
    time_min,
    frames_to_capture=25
):
    """
    This function starts a scenario with the given configuration,
    then captures frames for N (= frames_to_capture) iterations.

    :param client: The global Client instance.
    :param args: command line arguments (including host, port, save_dir)
    :param run_count: run count for file naming
    :param loc_x: x-coordinate of the location
    :param loc_y: y-coordinate of the location
    :param base_height: ground level for the location
    :param current_height: "above ground" height to maintain
    :param weather: weather condition string
    :param time_hour, time_min: in-game time to set
    :param frames_to_capture: number of frames to record
    """
    try:
        print(f"\nCapturing configuration for location ({loc_x}, {loc_y}):")
        print(f" - Weather: {weather}")
        print(f" - Time: {time_hour:02d}:{time_min:02d}")
        print(f" - Height: {current_height} m")
        print(f" - Camera position: {CAMERA_POSITION}")

        # Initialize scenario
        scenario = Scenario(
            drivingMode=0,
            vehicle="buzzard",
            location=[loc_x, loc_y, base_height]
        )
        dataset = Dataset(
            location=True,
            time=True,
            exportBBox2D=True,
            segmentationImage=True,
            exportLiDAR=False,
            maxLidarDist=5000,
            exportStencilImage=True,
            exportLiDARRaycast=False,
            exportDepthBuffer=True
        )

        # Start scenario and configure environment
        client.sendMessage(Start(scenario=scenario, dataset=dataset))
        client.sendMessage(SetCameraPositionAndRotation(
            y=CAMERA_POSITION['y'],
            z=CAMERA_POSITION['z'],
            rot_x=CAMERA_POSITION['rot_x'],
            rot_y=CAMERA_POSITION['rot_y']
        ))
        client.sendMessage(SetClockTime(time_hour, time_min))
        client.sendMessage(SetWeather(weather))

        # Wait for scene to stabilize
        time.sleep(2)

        with tqdm(total=frames_to_capture, desc=f"Capturing frames at ({loc_x}, {loc_y})") as pbar:
            for count in range(1, frames_to_capture + 1):
                try:
                    # Start recording for this frame
                    client.sendMessage(StartRecording())

                    message = client.recvMessage()
                    if message is None:
                        logging.warning("Received null message from client")
                        client.sendMessage(StopRecording())  # Make sure to stop recording
                        pbar.update(1)
                        continue

                    # Check if we have essential data
                    required_keys = ["segmentationImage", "bbox2d", "frame"]
                    missing_keys = [
                        key for key in required_keys
                        if key not in message or message[key] is None
                    ]
                    if missing_keys:
                        logging.warning(f"Missing required data: {missing_keys}")
                        client.sendMessage(StopRecording())  # Make sure to stop recording
                        pbar.update(1)
                        continue

                    # Height control
                    estimated_ground_height = message["location"][2] - message["HeightAboveGround"]
                    target_height = min(current_height, 5)  # Cap maximum height at 5m
                    current_actual_height = message["HeightAboveGround"]
                    if abs(current_actual_height - target_height) > 0.2:
                        new_height = estimated_ground_height + target_height
                        # Add a maximum height limit
                        new_height = min(new_height, estimated_ground_height + 5)
                        client.sendMessage(GoToLocation(loc_x, loc_y, new_height))

                    # Process and save frame
                    if message["segmentationImage"] and message["bbox2d"]:
                        # Build a filename using run_count, config info, and frame count
                        filename = (
                            f'{run_count:04}_{weather}_'
                            f'{time_hour:02d}{time_min:02d}_h{current_height:03d}_'
                            f'x{int(loc_x)}y{int(loc_y)}_{count:010}'
                        )

                        bboxes = parseBBoxesVisDroneStyle(message["bbox2d"])
                        frame = frame2numpy(message['frame'])
                        bbox_image = add_bboxes(
                            frame,
                            parseBBox_YoloFormatStringToImage(bboxes)
                        )

                        # Save image, bounding boxes, metadata
                        save_image_and_bbox(args.save_dir, filename, frame, bboxes)
                        save_meta_data(
                            args.save_dir, filename,
                            message["location"],
                            message["HeightAboveGround"],
                            message.get("CameraPosition", {}),
                            message.get("CameraAngle", {}),
                            message.get("time", {}),
                            weather
                        )

                        # Visualization
                        process_visualization(message, args, filename, bbox_image)

                    # Stop recording after processing the frame
                    client.sendMessage(StopRecording())
                    cv2.waitKey(1)
                    pbar.update(1)

                except Exception as e:
                    logging.error(f"Error in capture loop: {str(e)}\n{traceback.format_exc()}")
                    # Make sure to stop recording even if an error occurs
                    client.sendMessage(StopRecording())

        print("Finished capturing frames")

    except Exception as e:
        logging.error(f"Error in configuration: {str(e)}\n{traceback.format_exc()}")
    finally:
        # Stop the current scenario so we can start a new one next time
        try:
            client.sendMessage(Stop())
        except Exception as e:
            logging.error(f"Error stopping scenario: {str(e)}")


###############################################################################
# Main
###############################################################################

def main():
    setup_logging()
    logging.info("Starting capture session")
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='127.0.0.1', 
                        help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, type=int,
                        help='The port where DeepGTAV is running')
    parser.add_argument('-s', '--save_dir', 
                        default='C:\\workspace\\exported_data\\VisDrone_LiDAR_presentation_15',
                        help='Directory where generated data is saved')
    args = parser.parse_args()
    args.save_dir = os.path.normpath(args.save_dir)

    # Setup directories
    setup_directories(args.save_dir)
    
    # Get run count
    run_count = getRunCount(args.save_dir)

    # Attempt to move the GTA window
    try:
        if not move_gta_window():
            print("Could not find GTA V window, continuing anyway...")
    except Exception as e:
        print(f"Error moving GTA window: {e}")

    # Create a single, global client for the entire session
    client = None
    try:
        # Create the global client once
        client = Client(ip=args.host, port=args.port)
        print("Global client created.\n")

        # Iterate over all locations and configurations
        for (loc_x, loc_y, base_height, heights) in LOCATIONS:
            for weather in WEATHER_CONDITIONS:
                for time_hour, time_min in TIME_PERIODS:
                    for current_height in heights:
                        print(f"Capturing for weather: {weather}, time: {time_hour:02d}:{time_min:02d}, height: {current_height}")
                        # Capture for each configuration
                        capture_data_for_configuration(
                            client,
                            args=args,
                            run_count=run_count,
                            loc_x=loc_x,
                            loc_y=loc_y,
                            base_height=base_height,
                            current_height=current_height,
                            weather=weather,
                            time_hour=time_hour,
                            time_min=time_min,
                            frames_to_capture=25
                        )
                        # Wait 10 seconds after each configuration
                        print(f"Done with weather: {weather}, time: {time_hour:02d}:{time_min:02d}, height: {current_height}")
                        print("Waiting 10 seconds")
                        time.sleep(10)

    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error in main: {str(e)}\n{traceback.format_exc()}")
    finally:
        # Clean up global client at the very end
        if client:
            try:
                print("Stopping any ongoing recording...")
                client.sendMessage(StopRecording())
            except Exception as e:
                logging.error(f"Error stopping recording: {str(e)}")

            try:
                print("Stopping any active scenario...")
                client.sendMessage(Stop())
            except Exception as e:
                logging.error(f"Error stopping scenario: {str(e)}")

            try:
                print("Closing global client...")
                client.close()
            except Exception as e:
                logging.error(f"Error closing client: {str(e)}")

        print("Destroying OpenCV windows...")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
