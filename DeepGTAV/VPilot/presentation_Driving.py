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
# import open3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import win32gui
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive before importing pyplot

def move_gta_window():
    # Find GTA V window
    hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
    if hwnd:
        # Move window to (0,0) and optionally set size
        win32gui.SetWindowPos(hwnd, None, 0, 0, 1920, 1080, 0)
        return True
    return False

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
    
    scenario = Scenario(
        drivingMode=[786603, 15.0],
        weather='CLEAR',  # Ensure good visibility
        time=[12, 0],    # Daylight for better detection
        vehicle=None     # Use default vehicle
    )
    
    # Increase detection range and add more dataset parameters
    dataset = Dataset(
        location=True,
        time=True,
        weather=True,
        vehicles=True,    # Explicitly enable vehicle detection
        peds=True,        # Enable pedestrian detection
        trafficSigns=True,
        direction=True,
        reward=True,
        throttle=True,
        brake=True,
        steering=True,
        speed=True,
        yawRate=True,
        exportBBox2D=True,
        segmentationImage=True,
        exportLiDAR=True,
        maxLidarDist=120,
        detection_radius=150  # Increase detection radius
    )
    
    client.sendMessage(Start(scenario=scenario, dataset=dataset))

    # Camera offset in accordance with KITTY (in accordance with DeepGTA-PreSIL)
    client.sendMessage(SetCameraPositionAndRotation(z = 1.065))


    count = 0
    bbox2d_old = ""
    errors = []


    # SETTINGS

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
    if not os.path.exists(os.path.join(args.save_dir, 'SegmentationAndBBox')):
        os.makedirs(os.path.join(args.save_dir, 'SegmentationAndBBox'))
    if not os.path.exists(os.path.join(args.save_dir, 'LiDAR')):
        os.makedirs(os.path.join(args.save_dir, 'LiDAR'))

    
    
        

    run_count = getRunCount(args.save_dir)


    messages = []
    emptybbox = []

    try:
        if not move_gta_window():
            print("Could not find GTA V window")
    except Exception as e:
        print(f"Error moving GTA window: {e}")

    while True:
        try:
            count += 1
            print("count: ", count)

            # Only record every 10th frame
            if count > 50 and count % 10 == 0:
                client.sendMessage(StartRecording())
            if count > 50 and count % 10 == 1:
                client.sendMessage(StopRecording())
                

            # if count == 2:
            #     client.sendMessage(TeleportToLocation(-388, 0, 200))
            #     client.sendMessage(GoToLocation(1165, -553, 40))

            if count == 4:
                client.sendMessage(SetClockTime(12))

            if count == 250:
                client.sendMessage(SetClockTime(0))

            if count == 600:
                client.sendMessage(SetClockTime(19))
            

            message = client.recvMessage()  
            
            # None message from utf-8 decode error
            if message == None:
                continue

            # messages.append(message)

            # Plot Segmentation Image and Bounding Box image overlayed for testing 
            if message["segmentationImage"] != None and message["segmentationImage"] != "":
                bboxes = parseBBoxesVisDroneStyle(message["bbox2d"])
                
                filename = f'{run_count:04}' + '_' + f'{count:010}'
                save_image_and_bbox(args.save_dir, filename, frame2numpy(message['frame']), bboxes)
                save_meta_data(args.save_dir, filename, message["location"], message["HeightAboveGround"], message["CameraPosition"], message["CameraAngle"], message["time"], "CLEAR")
                
                bbox_image = add_bboxes(frame2numpy(message['frame'], (IMG_WIDTH,IMG_HEIGHT)), parseBBox_YoloFormatStringToImage(bboxes))
                
                nparr = np.frombuffer(base64.b64decode(message["segmentationImage"]), np.uint8)
                segmentationImage = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

                dst = cv2.addWeighted(bbox_image, 0.5, segmentationImage, 0.5, 0.0)

                cv2.namedWindow("SegmentationBBox", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("SegmentationBBox", int(1920 * (9/10)), int(1080 * (9/10)))
                cv2.imshow("SegmentationBBox", dst)
                cv2.waitKey(1)

                filename = f'{run_count:04}' + '_' + f'{count:010}' + ".png"
                cv2.imwrite(os.path.join(args.save_dir, "image", filename), bbox_image)
                cv2.imwrite(os.path.join(args.save_dir, "SegmentationAndBBox", filename), dst)


            if message["LiDAR"] != None and message["LiDAR"] != "":
                try:
                    a = np.frombuffer(base64.b64decode(message["LiDAR"]), np.float32)
                    a = a.reshape((-1, 4))

                    unique_entities, inv = np.unique(a[:, 3], return_inverse=True)
                    mapping = {k: v for k, v in zip(unique_entities, range(len(unique_entities)))}
                    vals = np.array([mapping[key] for key in unique_entities])
                    colors = np.array(vals[inv])

                    points3d = np.delete(a, 3, 1)

                    plt.ioff()  # Turn off interactive mode
                    fig = plt.figure(figsize=(80,40))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_xlim([-5, 100])
                    ax.set_ylim([-40, 40])
                    ax.set_zlim([-2, 20])

                    # ax.view_init(70, 180)
                    ax.view_init(0, 180)

                    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([10, 1, 1, 1]))

                    ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2], c=colors, s=1)

                    fig.canvas.draw()
                    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Note: 4 channels for ARGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert ARGB to BGR
                    height, width = fig.canvas.get_width_height()[::-1]
                    img = img[int(height * 0.5):int(height * 0.75), int(width * 0.4):int(width * 0.68)]
                    cv2.namedWindow("LiDAR", cv2.WINDOW_NORMAL)
                    cv2.imshow("LiDAR",img)
                    cv2.resizeWindow("LiDAR", int(1920 * (9/10)), int(1080 * (9/10)))
                    cv2.waitKey(1)

                    cv2.imwrite(os.path.join(args.save_dir, "LiDAR", filename), img)

                    plt.close(fig)  # Close the figure to free memory
                except Exception as e:
                    print(f"Error processing LiDAR data: {e}")
                    continue

            if message["bbox2d"] is None or message["bbox2d"] == "":
                print("Warning: No bounding box data received")
                continue
                
            bboxes = parseBBoxesVisDroneStyle(message["bbox2d"])
            
            # Filter out invalid bounding boxes
            filtered_bboxes = []
            for bbox in bboxes:
                try:
                    x, y, w, h, class_id, conf = map(float, bbox.split(','))
                    
                    # Stricter filtering conditions
                    min_size = 20  # Minimum width/height in pixels
                    max_size_ratio = 0.7  # Maximum box size relative to image
                    min_aspect_ratio = 0.25  # Minimum w/h ratio
                    max_aspect_ratio = 4.0  # Maximum w/h ratio
                    
                    # Calculate aspect ratio
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Size relative to image
                    relative_w = w / IMG_WIDTH
                    relative_h = h / IMG_HEIGHT
                    
                    if (w > min_size and h > min_size and  # Minimum size
                        w < IMG_WIDTH * max_size_ratio and h < IMG_HEIGHT * max_size_ratio and  # Maximum size
                        x >= 0 and y >= 0 and  # Position bounds
                        x + w <= IMG_WIDTH and y + h <= IMG_HEIGHT and  # Position bounds
                        conf > 0.7 and  # Higher confidence threshold
                        min_aspect_ratio < aspect_ratio < max_aspect_ratio and  # Reasonable aspect ratio
                        relative_w < max_size_ratio and relative_h < max_size_ratio):  # Not too large relative to image
                        filtered_bboxes.append(bbox)
                    else:
                        print(f"Filtered out box: size({w:.1f}x{h:.1f}) conf:{conf:.2f} ratio:{aspect_ratio:.2f}")
                except Exception as e:
                    print(f"Error processing bbox: {e}")
                    continue
            
            if not filtered_bboxes:
                print("Warning: No valid bounding boxes after filtering")
                continue
                
            # Use filtered_bboxes instead of bboxes for saving and visualization
            filename = f'{run_count:04}' + '_' + f'{count:010}'
            save_image_and_bbox(args.save_dir, filename, frame2numpy(message['frame']), filtered_bboxes)
            
            # Update visualization code to use filtered_bboxes
            bbox_image = add_bboxes(frame2numpy(message['frame'], (IMG_WIDTH,IMG_HEIGHT)), 
                                  parseBBox_YoloFormatStringToImage(filtered_bboxes))

            
        except KeyboardInterrupt:
            break
            
    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    client.close()



