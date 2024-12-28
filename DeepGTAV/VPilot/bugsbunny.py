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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import win32gui
import win32con

def move_gta_window():
    hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
    if hwnd:
        win32gui.SetWindowPos(
            hwnd, 
            None, 
            0, 
            0, 
            1920, 
            1080, 
            win32con.SWP_SHOWWINDOW | win32con.SWP_NOZORDER
        )
        return True
    return False

def calculate_steering(current_location, target_location):
    """Calculate steering based on current and target locations"""
    dx = target_location[0] - current_location[0]
    dy = target_location[1] - current_location[1]
    
    # Print distance to target for debugging
    distance = sqrt(dx*dx + dy*dy)
    print(f"Distance to target: {distance:.2f} units")
    
    angle = np.arctan2(dy, dx)
    # Convert to [-1, 1] range for steering
    return np.clip(angle / np.pi, -1, 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='127.0.0.1', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    parser.add_argument('-s', '--save_dir', default='C:\\workspace\\exported_data\\BugsBunnyData', help='The directory the generated data is saved to')
    # args = parser.parse_args()

    # TODO for running in VSCode
    args = parser.parse_args('')
    
    args.save_dir = os.path.normpath(args.save_dir)

    client = Client(ip=args.host, port=args.port)
    
    # Modified scenario without camera parameter
    scenario = Scenario(
        drivingMode=-1,  # -1 for manual control
        vehicle="blista",  # Regular car
        weather="CLEAR",
        time=[12, 0],
        location=[245.23306274414062, -998.244140625, 29.205352783203125]
    )
    
    # Dataset with camera parameter
    dataset = Dataset(
        location=True,
        time=True,
        exportBBox2D=True,
        segmentationImage=True,
        camera=2  # 2 = Third person camera
    )    
    
    # Try to move window before starting scenario
    try:
        # Multiple attempts to ensure window stays in position
        for _ in range(3):
            if move_gta_window():
                time.sleep(1)
    except Exception as e:
        print(f"Error moving GTA window: {e}")

    # Now start the scenario
    client.sendMessage(Start(scenario=scenario, dataset=dataset))
    
    # One final attempt to ensure window position after scenario loads
    time.sleep(2)
    move_gta_window()

    # Set up third person camera position
    client.sendMessage(SetCameraPositionAndRotation(
        x=0,  # Horizontal offset from vehicle
        y=-6,  # Distance behind vehicle
        z=2,   # Height above vehicle
        rot_x=10,  # Tilt down slightly
        rot_y=0,   # No side rotation
        rot_z=0    # No roll
    ))

    count = 0
    bbox2d_old = ""
    errors = []


    # SETTINGS

    currentTravelHeight = 40
    x_start, y_start = -388, 0
    x_target, y_target = 1165, -553


    if not os.path.exists(os.path.join(args.save_dir, 'images')):
        os.makedirs(os.path.join(args.save_dir, 'images'))
    if not os.path.exists(os.path.join(args.save_dir, 'labels')):
        os.makedirs(os.path.join(args.save_dir, 'labels'))
    if not os.path.exists(os.path.join(args.save_dir, 'meta_data')):
        os.makedirs(os.path.join(args.save_dir, 'meta_data'))

    if not os.path.exists(os.path.join(args.save_dir, 'image')):
        os.makedirs(os.path.join(args.save_dir, 'image'))
    if not os.path.exists(os.path.join(args.save_dir, 'depth')):
        os.makedirs(os.path.join(args.save_dir, 'depth'))
    if not os.path.exists(os.path.join(args.save_dir, 'StencilImage')):
        os.makedirs(os.path.join(args.save_dir, 'StencilImage'))
    if not os.path.exists(os.path.join(args.save_dir, 'SegmentationAndBBox')):
        os.makedirs(os.path.join(args.save_dir, 'SegmentationAndBBox'))
    if not os.path.exists(os.path.join(args.save_dir, 'LiDAR')):
        os.makedirs(os.path.join(args.save_dir, 'LiDAR'))
    
    
        

    run_count = getRunCount(args.save_dir)


    messages = []
    emptybbox = []
    message = None  # Initialize message variable

    try:
        if not move_gta_window():
            print("Could not find GTA V window")
    except Exception as e:
        print(f"Error moving GTA window: {e}")

    while True:
        try:
            count += 1

            # Get message first
            message = client.recvMessage()  
            if message == None:
                print("No message received")
                continue

            # Now process location and send commands
            if message and "location" in message:
                current_location = message["location"]
                print(f"Current location: {current_location}")
                # Calculate steering towards target
                steering = calculate_steering(
                    [current_location[0], current_location[1]], 
                    [x_target, y_target]
                )
                print(f"Calculated steering: {steering}")
                
                # Send driving commands with maximum throttle and more aggressive steering
                client.sendMessage(Commands(
                    throttle=1.0,     # Full throttle
                    steering=steering * 2.0,  # More aggressive steering
                    brake=0.0         # No braking
                ))
                print("Commands sent")

                # Optional: Stop if we're close to target
                dx = x_target - current_location[0]
                dy = y_target - current_location[1]
                if sqrt(dx*dx + dy*dy) < 5:  # Within 5 units of target
                    print("Target reached!")
                    break
            else:
                print("No location data in message")
                print(f"Message keys: {message.keys() if message else 'None'}")

            # Only record every 10th frame
            if count > 50 and count % 10 == 0:
                client.sendMessage(StartRecording())
            if count > 50 and count % 10 == 1:
                client.sendMessage(StopRecording())

            # Process segmentation and bounding boxes
            if message["segmentationImage"] != None and message["segmentationImage"] != "":
                bboxes = parseBBoxesVisDroneStyle(message["bbox2d"])
                
                filename = f'{run_count:04}' + '_' + f'{count:010}'
                save_image_and_bbox(args.save_dir, filename, frame2numpy(message['frame']), bboxes)
                save_meta_data(args.save_dir, filename, message["location"], 0, message["CameraPosition"], message["CameraAngle"], message["time"], "CLEAR")
                
                bbox_image = add_bboxes(frame2numpy(message['frame'], (IMG_WIDTH,IMG_HEIGHT)), parseBBox_YoloFormatStringToImage(bboxes))
                
                nparr = np.frombuffer(base64.b64decode(message["segmentationImage"]), np.uint8)
                segmentationImage = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

                dst = cv2.addWeighted(bbox_image, 0.5, segmentationImage, 0.5, 0.0)

                filename = f'{run_count:04}' + '_' + f'{count:010}' + ".png"
                cv2.imwrite(os.path.join(args.save_dir, "image", filename), bbox_image)
                cv2.imwrite(os.path.join(args.save_dir, "SegmentationAndBBox", filename), dst)

        except KeyboardInterrupt:
            break
            
    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    # client.close()



