#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.Constants import IMG_WIDTH, IMG_HEIGHT

from deepgtav.messages import Start, Stop, Scenario, Dataset, Commands, frame2numpy, GoToLocation, TeleportToLocation, SetCameraPositionAndRotation
from deepgtav.messages import StartRecording, StopRecording, SetClockTime, SetWeather, CreatePed
from deepgtav.client import Client

from utils.BoundingBoxes import add_bboxes, parseBBox2d_LikePreSIL, parseBBoxesVisDroneStyle, parseBBox_YoloFormatStringToImage
from utils.utils import save_image_and_bbox, save_meta_data, getRunCount, generateNewTargetLocation

import argparse
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from random import uniform
from math import sqrt
import numpy as np
import os
import base64
import open3d
import win32gui
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive before importing pyplot

# Configuration Constants
WEATHER_CONDITIONS = [
    'CLEAR', 'EXTRASUNNY', 'CLOUDS', 'OVERCAST', 
    'RAIN', 'FOGGY', 'THUNDER', 'CLEARING'
]

TIME_PERIODS = [
    (8, 0),   # Morning
    (12, 0),  # Noon
    (17, 0),  # Evening
    (22, 0),  # Night
]

# Define multiple locations with their heights
LOCATIONS = [
    # x, y, base_height, [height_variations]
    (-388, 0, 400, [25, 40, 100]),  # Original location
    (245, -998, 400, [25, 40, 100]),  # Downtown
    (1165, -553, 400, [25, 40, 100]),  # Beach area
]

def move_gta_window():
    # Find GTA V window
    hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
    if hwnd:
        # Move window to (0,0) and optionally set size
        win32gui.SetWindowPos(hwnd, None, 0, 0, 1920, 1080, 0)
        return True
    return False

def setup_directories(base_dir):
    """Create all necessary directories for data storage"""
    directories = [
        'images', 'labels', 'meta_data', 'image', 'depth',
        'StencilImage', 'SegmentationAndBBox', 'LiDAR', 'semantic_vis'
    ]
    for dir_name in directories:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def process_visualization(message, args, filename, bbox_image=None):
    """Handle visualization windows and saving visualization data"""
    if message["segmentationImage"] != None and message["segmentationImage"] != "":
        nparr = np.frombuffer(base64.b64decode(message["segmentationImage"]), np.uint8)
        segmentationImage = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

        # Create and position preview windows
        windows = {
            "Original with BBoxes": bbox_image,
            "Semantic Segmentation": segmentationImage,
            "Overlay": cv2.addWeighted(bbox_image, 0.5, segmentationImage, 0.5, 0.0)
        }

        for idx, (name, img) in enumerate(windows.items()):
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 640, 360)
            cv2.moveWindow(name, 1920 + (50 if idx < 2 else 740), 50 + (410 * (idx % 2)))
            cv2.imshow(name, img)

        # Save visualization files
        cv2.imwrite(os.path.join(args.save_dir, "image", f"{filename}.png"), bbox_image)
        cv2.imwrite(os.path.join(args.save_dir, "SegmentationAndBBox", f"{filename}.png"), windows["Overlay"])

def process_lidar(message, args, filename):
    """Process and save LiDAR data"""
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

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='127.0.0.1', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    parser.add_argument('-s', '--save_dir', default='C:\\workspace\\exported_data\\VisDrone_LiDAR_presentation_5', 
                        help='The directory the generated data is saved to')
    args = parser.parse_args('')  # For running in VSCode
    args.save_dir = os.path.normpath(args.save_dir)

    # Setup directories
    setup_directories(args.save_dir)
    
    # Initialize client and get run count
    client = Client(ip=args.host, port=args.port)
    run_count = getRunCount(args.save_dir)

    try:
        if not move_gta_window():
            print("Could not find GTA V window")
    except Exception as e:
        print(f"Error moving GTA window: {e}")

    try:
        # Iterate through different configurations
        for weather in WEATHER_CONDITIONS:
            for time_hour, time_min in TIME_PERIODS:
                for loc_x, loc_y, base_height, heights in LOCATIONS:
                    for current_height in heights:
                        print(f"\nCapturing configuration:")
                        print(f"Weather: {weather}")
                        print(f"Time: {time_hour:02d}:{time_min:02d}")
                        print(f"Location: ({loc_x}, {loc_y})")
                        print(f"Height: {current_height}m")

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
                            exportLiDAR=True,
                            maxLidarDist=5000,
                            exportStencilImage=True,
                            exportLiDARRaycast=True,
                            exportDepthBuffer=True
                        )

                        # Start scenario and configure environment
                        client.sendMessage(Start(scenario=scenario, dataset=dataset))
                        client.sendMessage(SetCameraPositionAndRotation(z=-20, rot_x=-90))
                        client.sendMessage(SetClockTime(time_hour, time_min))
                        client.sendMessage(SetWeather(weather))

                        # Wait for scene to stabilize
                        time.sleep(2)

                        count = 0
                        capture_frames = 30  # Frames to capture per configuration

                        while count < capture_frames:
                            try:
                                count += 1

                                # Record every 5th frame
                                if count % 5 == 0:
                                    client.sendMessage(StartRecording())
                                elif count % 5 == 1:
                                    client.sendMessage(StopRecording())

                                message = client.recvMessage()
                                if message is None:
                                    continue

                                # Maintain height
                                estimated_ground_height = message["location"][2] - message["HeightAboveGround"]
                                client.sendMessage(GoToLocation(
                                    loc_x, loc_y, 
                                    estimated_ground_height + current_height
                                ))

                                # Process frame if available
                                if message["segmentationImage"] and message["bbox2d"]:
                                    # Generate filename with configuration info
                                    filename = f'{run_count:04}_{weather}_{time_hour:02d}{time_min:02d}_h{current_height:03d}_{count:010}'
                                    
                                    # Process bounding boxes
                                    bboxes = parseBBoxesVisDroneStyle(message["bbox2d"])
                                    frame = frame2numpy(message['frame'])
                                    bbox_image = add_bboxes(frame, parseBBox_YoloFormatStringToImage(bboxes))
                                    
                                    # Save data
                                    save_image_and_bbox(args.save_dir, filename, frame, bboxes)
                                    save_meta_data(
                                        args.save_dir, filename,
                                        message["location"],
                                        message["HeightAboveGround"],
                                        message["CameraPosition"],
                                        message["CameraAngle"],
                                        message["time"],
                                        weather
                                    )

                                    # Handle visualization
                                    process_visualization(message, args, filename, bbox_image)
                                    
                                    # Process LiDAR data
                                    process_lidar(message, args, filename)

                                # Small delay to prevent overwhelming the system
                                cv2.waitKey(1)

                            except Exception as e:
                                print(f"Error processing frame {count}: {str(e)}")
                                continue

    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    finally:
        # Cleanup
        client.sendMessage(Stop())
        client.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



