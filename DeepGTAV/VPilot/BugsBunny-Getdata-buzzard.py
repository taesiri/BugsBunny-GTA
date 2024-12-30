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
import json

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
        'SegmentationAndBBox', 'semantic_vis', 'LiDAR',
        'bbox_json', 'segmentation_json',
        'frame_index'  # New directory for frame index files
    ]
    for dir_name in directories:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def save_frame_index(save_dir, frame_data):
    """Save frame index with file paths."""
    frame_index_dir = os.path.join(save_dir, "frame_index")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the index entry
    index_entry = {
        "frame_id": frame_data["filename"],
        "files": {
            "image": f"image/{frame_data['filename']}.png",
            "bbox": f"labels/{frame_data['filename']}.txt",
            "bbox_json": f"bbox_json/{frame_data['filename']}.json",
            "segmentation_json": f"segmentation_json/{frame_data['filename']}.json",
            "segmentation_overlay": f"SegmentationAndBBox/{frame_data['filename']}.png",
            "metadata": f"meta_data/{frame_data['filename']}.json",
            "lidar": f"LiDAR/{frame_data['filename']}.ply"
        },
        "timestamp": timestamp
    }
    
    # Save individual frame index
    index_file = os.path.join(frame_index_dir, f"{frame_data['filename']}.json")
    with open(index_file, 'w') as f:
        json.dump(index_entry, f, indent=2)
    
    return index_entry

def process_visualization(message, args, filename, bbox_image=None):
    """Handle visualization windows and saving visualization data."""
    try:
        if message["segmentationImage"] is None:
            logging.warning("Segmentation image is None")
            return
            
        if message["segmentationImage"] == "":
            logging.warning("Segmentation image is empty")
            return

        # Save segmentation data as JSON
        segmentation_json_path = os.path.join(args.save_dir, "segmentation_json", f"{filename}.json")
        with open(segmentation_json_path, 'w') as f:
            json.dump({"segmentationImage": message["segmentationImage"]}, f)

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

def verify_saved_files(save_dir, filename):
    """Verify that all expected files were saved."""
    expected_files = {
        "image": f"image/{filename}.png",
        "bbox": f"labels/{filename}.txt",
        "bbox_json": f"bbox_json/{filename}.json",
        "segmentation_json": f"segmentation_json/{filename}.json",
        "segmentation_overlay": f"SegmentationAndBBox/{filename}.png",
        "metadata": f"meta_data/{filename}.json",
        "lidar": f"LiDAR/{filename}.ply"
    }
    
    missing_files = []
    for file_type, path in expected_files.items():
        full_path = os.path.join(save_dir, path)
        if not os.path.exists(full_path):
            missing_files.append(file_type)
            
    if missing_files:
        logging.warning(f"Missing files for frame {filename}: {missing_files}")
        return False
    return True

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
    camera_position,
    frames_to_capture=25
):
    """
    This function starts a scenario with the given configuration,
    then captures frames for N (= frames_to_capture) iterations.
    """
    try:
        print(f"\nCapturing configuration for location ({loc_x}, {loc_y}):")
        print(f" - Weather: {weather}")
        print(f" - Time: {time_hour:02d}:{time_min:02d}")
        print(f" - Height: {current_height} m")
        print(f" - Camera position: {camera_position}")

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
            y=camera_position['y'],
            z=camera_position['z'],
            rot_x=camera_position['rot_x'],
            rot_y=camera_position['rot_y']
        ))
        client.sendMessage(SetClockTime(time_hour, time_min))
        client.sendMessage(SetWeather(weather))

        # Wait for scene to stabilize and start recording
        time.sleep(2)
        client.sendMessage(StartRecording())

        frame_indices = []  # Store all frame indices
        
        with tqdm(total=frames_to_capture, desc=f"Capturing frames at ({loc_x}, {loc_y})") as pbar:
            for count in range(1, frames_to_capture + 1):
                try:
                    message = client.recvMessage()
                    if message is None:
                        logging.warning("Received null message from client")
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
                        pbar.update(1)
                        continue
                    
                    if not message["bbox2d"]:
                        logging.warning(f"bbox2d is empty")
                        pbar.update(1)
                        continue

                    # Move the drone for each frame
                    target_height = min(current_height, 5)  # Cap maximum height at 5m
                    client.sendMessage(GoToLocation(loc_x, loc_y, base_height + target_height))

                    # Process and save frame
                    if message["segmentationImage"] and message["bbox2d"]:
                        # Build a filename using run_count, config info, and frame count
                        filename = (
                            f'{int(run_count):04}_{weather}_'
                            f'{time_hour:02d}{time_min:02d}_h{int(current_height):03d}_'
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

                        # Save bounding box data as JSON
                        bbox_json_path = os.path.join(args.save_dir, "bbox_json", f"{filename}.json")
                        with open(bbox_json_path, 'w') as f:
                            json.dump({"bbox2d": message["bbox2d"]}, f)

                        # Visualization
                        process_visualization(message, args, filename, bbox_image)

                        if verify_saved_files(args.save_dir, filename):
                            frame_data = {"filename": filename}
                            frame_index = save_frame_index(args.save_dir, frame_data)
                            frame_indices.append(frame_index)
                        else:
                            logging.error(f"Skipping frame index for {filename} due to missing files")

                    cv2.waitKey(1)
                    pbar.update(1)

                except Exception as e:
                    logging.error(f"Error in capture loop: {str(e)}\n{traceback.format_exc()}")

        print("Finished capturing frames")

        # Save complete session index
        session_index = {
            "session_id": f"session_{int(run_count):04}",
            "frames": frame_indices,
            "total_frames": len(frame_indices),
            "capture_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        session_file = os.path.join(args.save_dir, "frame_index", f"session_{int(run_count):04}.json")
        with open(session_file, 'w') as f:
            json.dump(session_index, f, indent=2)

    except Exception as e:
        logging.error(f"Error in configuration: {str(e)}\n{traceback.format_exc()}")
    finally:
        # Stop the current scenario so we can start a new one next time
        try:
            client.sendMessage(StopRecording())  # Stop recording when done
        except Exception as e:
            logging.error(f"Error stopping recording: {str(e)}")
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
    
    parser = argparse.ArgumentParser(description="Capture data from GTA V using DeepGTAV")

    # DeepGTAV connection settings
    parser.add_argument('-l', '--host', default='127.0.0.1', 
                        help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, type=int,
                        help='The port where DeepGTAV is running')

    # Data export settings
    parser.add_argument('-s', '--save_dir', 
                        default='C:\\workspace\\exported_data\\VisDrone_LiDAR_presentation_16',
                        help='Directory where generated data is saved')

    # Location & altitude parameters
    parser.add_argument('--loc_x', type=float, default=100, 
                        help='X coordinate of the location')
    parser.add_argument('--loc_y', type=float, default=3, 
                        help='Y coordinate of the location')
    parser.add_argument('--base_height', type=float, default=11, 
                        help='Base height above ground for the drone to spawn')
    parser.add_argument('--current_height', type=float, default=5, 
                        help='Current/target flight height above base_height')

    # Environment parameters
    parser.add_argument('--weather', type=str, default='THUNDER', 
                        help="Weather type, e.g. 'CLEAR', 'RAIN', 'THUNDER'")
    parser.add_argument('--time_hour', type=int, default=12, 
                        help='Hour of the day (0-23) for in-game time')
    parser.add_argument('--time_min', type=int, default=0, 
                        help='Minutes of the day (0-59) for in-game time')

    # Capture parameters
    parser.add_argument('--frames_to_capture', type=int, default=25, 
                        help='Number of frames to capture')

    # Camera position and rotation
    parser.add_argument('--cam_y', type=float, default=4.5, 
                        help='Camera position offset on the Y axis')
    parser.add_argument('--cam_z', type=float, default=1.8, 
                        help='Camera position offset on the Z axis')
    parser.add_argument('--rot_x', type=float, default=0, 
                        help='Camera rotation in X (pitch)')
    parser.add_argument('--rot_y', type=float, default=0, 
                        help='Camera rotation in Y (roll)')

    args = parser.parse_args()
    args.save_dir = os.path.normpath(args.save_dir)

    # Construct camera position dictionary from parser args
    camera_position = {
        'y': args.cam_y,
        'z': args.cam_z,
        'rot_x': args.rot_x,
        'rot_y': args.rot_y
    }

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

        # Now capture data for the configuration specified by command-line arguments
        capture_data_for_configuration(
            client,
            args=args,
            run_count=run_count,
            loc_x=args.loc_x,
            loc_y=args.loc_y,
            base_height=args.base_height,
            current_height=args.current_height,
            weather=args.weather,
            time_hour=args.time_hour,
            time_min=args.time_min,
            camera_position=camera_position,
            frames_to_capture=args.frames_to_capture
        )

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
                time.sleep(0.5)  # Brief pause after stopping recording
            except Exception as e:
                logging.error(f"Error stopping recording: {str(e)}")

            try:
                print("Stopping active scenario...")
                client.sendMessage(Stop())
                time.sleep(1)  # Give GTA time to process stop command
                
                # Simplified reset - just return to ground without forcing a new scenario
                print("Returning to ground level...")
                client.sendMessage(GoToLocation(-75.0, -818.0, 326.0))
                time.sleep(1)  # Let position update
                
            except Exception as e:
                logging.error(f"Error during scenario cleanup: {str(e)}")

            try:
                print("Closing client connection...")
                client.close()
                time.sleep(0.5)  # Brief pause after closing
            except Exception as e:
                logging.error(f"Error closing client: {str(e)}")

        print("Destroying OpenCV windows...")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
