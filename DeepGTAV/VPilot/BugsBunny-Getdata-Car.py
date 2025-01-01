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
import json

from deepgtav.messages import (
    Start, Stop, Scenario, Dataset, frame2numpy,
    Commands, GoToLocation, SetCameraPositionAndRotation,
    StartRecording, StopRecording, SetClockTime, SetWeather
)
from deepgtav.client import Client

# Local utility imports (adjust to your project structure)
from utils.Constants import IMG_WIDTH, IMG_HEIGHT
from utils.BoundingBoxes import (
    add_bboxes, parseBBoxesVisDroneStyle, parseBBox_YoloFormatStringToImage
)
from utils.utils import (
    save_image_and_bbox, save_meta_data,
    getRunCount
)

matplotlib.use('Agg')  # Use non-interactive backend

###############################################################################
# Example Self-Driving Model
###############################################################################
class Model:
    """
    A dummy agent that always drives straight at full throttle.
    You can replace 'run' with your own prediction logic.
    """
    def run(self, frame):
        # e.g. [throttle, brake, steering]
        # throttle=1.0 means full gas, brake=0.0 means no brake, steering=0.0 means go straight
        return [1.0, 0.0, 0.0]

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
        'frame_index'
    ]
    for dir_name in directories:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def save_frame_index(save_dir, frame_data):
    """Save frame index with file paths."""
    frame_index_dir = os.path.join(save_dir, "frame_index")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

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

    index_file = os.path.join(frame_index_dir, f"{frame_data['filename']}.json")
    with open(index_file, 'w') as f:
        json.dump(index_entry, f, indent=2)

    return index_entry

def process_visualization(message, args, filename, bbox_image=None):
    """Handle visualization windows and saving segmentation overlay."""
    try:
        seg_str = message.get("segmentationImage", "")
        if not seg_str:
            logging.warning("Segmentation image is None or empty")
            return

        # Save segmentation data as JSON
        segmentation_json_path = os.path.join(args.save_dir, "segmentation_json", f"{filename}.json")
        with open(segmentation_json_path, 'w') as f:
            json.dump({"segmentationImage": seg_str}, f)

        # Decode segmentation image from base64
        nparr = np.frombuffer(base64.b64decode(seg_str), np.uint8)
        segmentationImage = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        
        if segmentationImage is None:
            logging.error("Failed to decode segmentation image")
            return

        # Create overlay
        overlay = cv2.addWeighted(bbox_image, 0.5, segmentationImage, 0.5, 0.0)

        # (Optional) Display windows if you want real-time previews
        # You can comment these out if running headless or uninterested in previews.
        cv2.namedWindow("Original with BBoxes", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original with BBoxes", 640, 360)
        cv2.imshow("Original with BBoxes", bbox_image)

        cv2.namedWindow("Semantic Segmentation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Semantic Segmentation", 640, 360)
        cv2.imshow("Semantic Segmentation", segmentationImage)

        cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Overlay", 640, 360)
        cv2.imshow("Overlay", overlay)

        # Save visualization files
        cv2.imwrite(os.path.join(args.save_dir, "image", f"{filename}.png"), bbox_image)
        cv2.imwrite(os.path.join(args.save_dir, "SegmentationAndBBox", f"{filename}.png"), overlay)

    except Exception as e:
        logging.error(f"Error in process_visualization: {str(e)}\n{traceback.format_exc()}")

def process_lidar(message, args, filename):
    """Process and save LiDAR data if available."""
    lidar_str = message.get("LiDAR", "")
    if lidar_str:
        points_data = np.frombuffer(base64.b64decode(lidar_str), np.float32)
        # shape: Nx4 => remove intensity channel => Nx3
        points3d = np.delete(points_data.reshape((-1, 4)), 3, 1)

        # Create color gradient based on height
        z_norm = (points3d[:, 0] - points3d[:, 0].min()) / (points3d[:, 0].max() - points3d[:, 0].min() + 1e-6)
        colors = np.zeros((points3d.shape[0], 3))
        colors[:, 0] = z_norm
        colors[:, 2] = 1 - z_norm

        # Create and save point cloud
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(points3d)
        point_cloud.colors = open3d.utility.Vector3dVector(colors)
        lidar_path = os.path.join(args.save_dir, "LiDAR", f"{filename}.ply")
        open3d.io.write_point_cloud(lidar_path, point_cloud)

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
# Capture + Self-Driving Loop
###############################################################################

def capture_and_drive(
    client,
    args,
    run_count,
    model,
    frames_to_capture=100
):
    """
    Combined function to:
     - Receive frames from DeepGTAV.
     - Run a "self-driving" model to produce throttle, brake, steering commands.
     - Save bounding boxes, segmentation, metadata, etc.
    """
    try:
        print(f"\nStarting capture and self-driving loop with {frames_to_capture} frames...")

        frame_indices = []

        with tqdm(total=frames_to_capture, desc="Capturing frames") as pbar:
            for count in range(1, frames_to_capture + 1):
                try:
                    message = client.recvMessage()
                    if message is None:
                        logging.warning("Received null message from client")
                        pbar.update(1)
                        continue

                    # Basic checks
                    if "frame" not in message or message["frame"] is None:
                        logging.warning("No frame data in message")
                        pbar.update(1)
                        continue

                    # 1) RUN MODEL (Vehicle Control)
                    # Convert image to (320, 160) or your CNN input dimension if desired
                    # For demonstration, let's assume we just use the raw frame with shape (IMG_HEIGHT, IMG_WIDTH)
                    # or you can do: frame_model = frame2numpy(message["frame"], (320, 160))
                    # For now, let's decode the original resolution to do the bounding box overlay
                    full_frame = frame2numpy(message["frame"])
                    
                    # In a real ML scenario, you'd resize or do transformations for the model input
                    commands = model.run(full_frame)  # e.g. [throttle, brake, steering]
                    if len(commands) == 3:
                        client.sendMessage(Commands(*commands))
                    else:
                        logging.warning(f"Model returned unexpected commands: {commands}")

                    # 2) CAPTURE + SAVE bounding boxes, segmentation, etc.
                    if "bbox2d" not in message or not message["bbox2d"]:
                        # Even if no bounding boxes, let's keep capturing frames
                        logging.warning("bbox2d is empty or missing")
                    
                    # Build a filename for saving
                    filename = (
                        f'{int(run_count):04}_'
                        f'{args.weather}_'
                        f'{args.time_hour:02d}{args.time_min:02d}_'
                        f'CAR_{count:06d}'
                    )

                    # Save bounding boxes
                    if "bbox2d" in message and message["bbox2d"]:
                        bboxes = parseBBoxesVisDroneStyle(message["bbox2d"])
                        bbox_image = add_bboxes(
                            full_frame,
                            parseBBox_YoloFormatStringToImage(bboxes)
                        )
                    else:
                        bbox_image = full_frame  # No bboxes, just keep original frame

                    # Save image & bounding box file
                    save_image_and_bbox(args.save_dir, filename, full_frame, 
                                       parseBBoxesVisDroneStyle(message.get("bbox2d", [])))
                    
                    # Save metadata (location, time, camera info, etc.)
                    save_meta_data(
                        args.save_dir, filename,
                        message.get("location", {}),
                        message.get("HeightAboveGround", 0),
                        message.get("CameraPosition", {}),
                        message.get("CameraAngle", {}),
                        message.get("time", {}),
                        args.weather
                    )

                    # Save bounding boxes as JSON
                    bbox_json_path = os.path.join(args.save_dir, "bbox_json", f"{filename}.json")
                    with open(bbox_json_path, 'w') as f:
                        json.dump({"bbox2d": message.get("bbox2d", [])}, f)

                    # Process visualization (overlay segmentation if present)
                    process_visualization(message, args, filename, bbox_image)

                    # (Optional) Process LiDAR if exportLiDAR=True in Dataset
                    # process_lidar(message, args, filename)

                    # 3) Verify and add to frame index
                    if verify_saved_files(args.save_dir, filename):
                        frame_data = {"filename": filename}
                        frame_index = save_frame_index(args.save_dir, frame_data)
                        frame_indices.append(frame_index)
                    else:
                        logging.error(f"Missing files for {filename}, skipping index")

                    # Step progress
                    cv2.waitKey(1)
                    pbar.update(1)

                except Exception as e:
                    logging.error(f"Error in capture loop: {str(e)}\n{traceback.format_exc()}")
        
        print("Capture/self-driving loop finished.")

        # Save a session index for all frames
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
        logging.error(f"Error in capture_and_drive: {str(e)}\n{traceback.format_exc()}")


###############################################################################
# Main
###############################################################################

def main():
    setup_logging()
    logging.info("Starting capture session (Self-Driving with Python commands)")

    parser = argparse.ArgumentParser(description="Self-Driving Car + Data Capture with DeepGTAV")
    parser.add_argument('-l', '--host', default='127.0.0.1', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, type=int, help='The port where DeepGTAV is running')
    parser.add_argument('-s', '--save_dir', 
                        default='C:\\workspace\\exported_data\\SelfDrivingCar_Capture',
                        help='Directory where generated data is saved')

    # Location & environment
    parser.add_argument('--loc_x', type=float, default=100, help='X coordinate of the spawn location')
    parser.add_argument('--loc_y', type=float, default=3, help='Y coordinate of the spawn location')
    parser.add_argument('--weather', type=str, default='CLEAR', 
                        help="Weather type, e.g. 'CLEAR', 'RAIN', 'THUNDER'")
    parser.add_argument('--time_hour', type=int, default=12, help='Hour of the day (0-23)')
    parser.add_argument('--time_min', type=int, default=0, help='Minutes of the day (0-59)')

    # Capture parameters
    parser.add_argument('--frames_to_capture', type=int, default=50, 
                        help='Number of frames to capture in self-driving loop')

    # Camera offsets
    parser.add_argument('--cam_y', type=float, default=4.0, 
                        help='Camera position offset on the Y axis (behind the car)')
    parser.add_argument('--cam_z', type=float, default=1.5, 
                        help='Camera position offset on the Z axis (height above the car)')
    parser.add_argument('--rot_x', type=float, default=0, help='Camera rotation in X (pitch)')
    parser.add_argument('--rot_y', type=float, default=0, help='Camera rotation in Y (roll)')

    args = parser.parse_args()
    args.save_dir = os.path.normpath(args.save_dir)

    # Create needed directories
    setup_directories(args.save_dir)
    
    # Determine run count
    run_count = getRunCount(args.save_dir)

    # Attempt to move the GTA window
    try:
        if not move_gta_window():
            print("Could not find GTA V window, continuing anyway...")
    except Exception as e:
        print(f"Error moving GTA window: {e}")

    # Create a client connection to DeepGTAV
    client = None
    try:
        client = Client(ip=args.host, port=args.port)
        logging.info("DeepGTAV Client connected.")

        # Prepare scenario with manual driving => we send Commands ourselves
        scenario = Scenario(
            drivingMode=-1,  # Accept manual commands from Python
            vehicle="comet2",  # Example car
            location=[args.loc_x, args.loc_y, 0.0]  # Spawn location on ground
        )
        dataset = Dataset(
            location=True,
            time=True,
            exportBBox2D=True,
            segmentationImage=True,
            exportLiDAR=False,  # Enable if you want LiDAR
            maxLidarDist=5000,
            exportStencilImage=True,
            exportLiDARRaycast=False,
            exportDepthBuffer=True
        )

        # Start scenario & environment
        client.sendMessage(Start(scenario=scenario, dataset=dataset))
        client.sendMessage(SetCameraPositionAndRotation(
            y=args.cam_y,
            z=args.cam_z,
            rot_x=args.rot_x,
            rot_y=args.rot_y
        ))
        client.sendMessage(SetClockTime(args.time_hour, args.time_min))
        client.sendMessage(SetWeather(args.weather))

        time.sleep(2)  # Let the game stabilize

        # Start recording
        client.sendMessage(StartRecording())

        # Create our "agent" model
        model = Model()

        # Perform capture & driving for N frames
        capture_and_drive(client, args, run_count, model, frames_to_capture=args.frames_to_capture)

    except KeyboardInterrupt:
        print("\nCapture/self-driving interrupted by user.")
    except Exception as e:
        logging.error(f"Unexpected error in main: {str(e)}\n{traceback.format_exc()}")

    finally:
        # Cleanup
        if client:
            try:
                logging.info("Stopping recording...")
                client.sendMessage(StopRecording())
                time.sleep(0.5)
            except Exception as e:
                logging.error(f"Error stopping recording: {str(e)}")

            try:
                logging.info("Stopping scenario...")
                client.sendMessage(Stop())
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error stopping scenario: {str(e)}")

            # Optionally return to some default location or reset the game state
            try:
                logging.info("Resetting location or scenario if needed...")
                client.sendMessage(GoToLocation(-75.0, -818.0, 0.0))
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error resetting location: {str(e)}")

            try:
                logging.info("Closing DeepGTAV client...")
                client.close()
                time.sleep(0.5)
            except Exception as e:
                logging.error(f"Error closing client: {str(e)}")

        logging.info("Destroying OpenCV windows.")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
