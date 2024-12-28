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
    parser.add_argument('-s', '--save_dir', default='C:\\workspace\\exported_data\\VisDrone_LiDAR_presentation_3', help='The directory the generated data is saved to')
    # args = parser.parse_args()

    # TODO for running in VSCode
    args = parser.parse_args('')
    
    args.save_dir = os.path.normpath(args.save_dir)

    client = Client(ip=args.host, port=args.port)
    
    scenario = Scenario(drivingMode=0, vehicle="buzzard", location=[245.23306274414062, -998.244140625, 29.205352783203125])
    dataset=Dataset(location=True, time=True, exportBBox2D=True, segmentationImage=True, exportLiDAR=True, maxLidarDist=5000, exportStencilImage=True, exportLiDARRaycast=True, exportDepthBuffer=True)    
    
    client.sendMessage(Start(scenario=scenario, dataset=dataset))


    # Adjustments for recording from UAV perspective
    client.sendMessage(SetCameraPositionAndRotation(z = -20, rot_x = -90))


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
    if not os.path.exists(os.path.join(args.save_dir, 'semantic_vis')):
        os.makedirs(os.path.join(args.save_dir, 'semantic_vis'))
    
    
        

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
            # print("count: ", count)

            # Only record every 10th frame
            if count > 50 and count % 10 == 0:
                client.sendMessage(StartRecording())
            if count > 50 and count % 10 == 1:
                client.sendMessage(StopRecording())
                

            if count == 2:
                client.sendMessage(TeleportToLocation(-388, 0, 400))
                client.sendMessage(GoToLocation(1165, -553, 400))

            if count == 4:
                client.sendMessage(SetClockTime(12))

            if count == 150:
                client.sendMessage(SetClockTime(0))

            if count == 200:
                client.sendMessage(SetClockTime(19))
            

            if count == 250:
                currentTravelHeight = 25

            if count == 300:
                currentTravelHeight = 100

            if count == 380:
                currentTravelHeight = 40


            # if count == 400:
            #     client.sendMessage(SetCameraPositionAndRotation(z = -20, rot_x = -90))

            # if count == 450:
            #     client.sendMessage(SetCameraPositionAndRotation(z = -20, rot_x = -20))



            message = client.recvMessage()  
            
            # None message from utf-8 decode error
            if message == None:
                continue

            # messages.append(message)

            # keep the currentTravelHeight under the wanted one
            # Move a little bit in the desired direction but primarily correct the height
            estimated_ground_height = message["location"][2] - message["HeightAboveGround"]
            if message["HeightAboveGround"] > currentTravelHeight + 3 or message["HeightAboveGround"] < currentTravelHeight - 3:
                direction = np.array([x_target - message["location"][0], y_target - message["location"][1]])
                direction = direction / np.linalg.norm(direction)
                direction = direction * 50
                x_temporary = message["location"][0] + direction[0]
                y_temporary = message["location"][1] + direction[1]
                client.sendMessage(GoToLocation(x_temporary, y_temporary, estimated_ground_height + currentTravelHeight))
                # print("Correcting height")
            else:
                client.sendMessage(GoToLocation(x_target, y_target, estimated_ground_height + currentTravelHeight))

            # print(message["segmentationImage"])
            # Plot Segmentation Image and Bounding Box image overlayed for testing 
            if message["segmentationImage"] != None and message["segmentationImage"] != "":
                bboxes = parseBBoxesVisDroneStyle(message["bbox2d"])
                
                filename = f'{run_count:04}' + '_' + f'{count:010}'
                save_image_and_bbox(args.save_dir, filename, frame2numpy(message['frame']), bboxes)
                save_meta_data(args.save_dir, filename, message["location"], message["HeightAboveGround"], message["CameraPosition"], message["CameraAngle"], message["time"], "CLEAR")
                
                bbox_image = add_bboxes(frame2numpy(message['frame'], (IMG_WIDTH,IMG_HEIGHT)), parseBBox_YoloFormatStringToImage(bboxes))
                
                nparr = np.frombuffer(base64.b64decode(message["segmentationImage"]), np.uint8)
                segmentationImage = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

                # Create preview windows
                cv2.namedWindow("Original with BBoxes", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Semantic Segmentation", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)

                # Resize windows
                cv2.resizeWindow("Original with BBoxes", 640, 360)
                cv2.resizeWindow("Semantic Segmentation", 640, 360)
                cv2.resizeWindow("Overlay", 640, 360)

                # Position windows on second monitor (assuming 1920x1080 primary monitor)
                # Adjust these coordinates based on your monitor setup
                second_monitor_x = 1920  # Starting X coordinate of second monitor
                cv2.moveWindow("Original with BBoxes", second_monitor_x + 50, 50)
                cv2.moveWindow("Semantic Segmentation", second_monitor_x + 50, 460)
                cv2.moveWindow("Overlay", second_monitor_x + 740, 50)

                # Show images
                cv2.imshow("Original with BBoxes", bbox_image)
                cv2.imshow("Semantic Segmentation", segmentationImage)
                
                dst = cv2.addWeighted(bbox_image, 0.5, segmentationImage, 0.5, 0.0)
                cv2.imshow("Overlay", dst)
                
                # Add a small delay to allow windows to update
                cv2.waitKey(1)

                # Continue with existing saving code...
                filename = f'{run_count:04}' + '_' + f'{count:010}' + ".png"
                cv2.imwrite(os.path.join(args.save_dir, "image", filename), bbox_image)
                cv2.imwrite(os.path.join(args.save_dir, "SegmentationAndBBox", filename), dst)

            # print(message["LiDAR"])
            if message["LiDAR"] != None and message["LiDAR"] != "":
                # print(message["LiDAR"])
                a = np.frombuffer(base64.b64decode(message["LiDAR"]), np.float32)
                a = a.reshape((-1, 4))
                points3d = np.delete(a, 3, 1)

                # 获取 Z 坐标并对其进行归一化
                z_min, z_max = points3d[:, 0].min(), points3d[:, 0].max()
                z_norm = (points3d[:, 0] - z_min) / (z_max - z_min)

                # 创建 RGB 颜色数组，基于 Z 坐标的归一化值创建颜色梯度（例如，蓝色到红色）
                colors = np.zeros((points3d.shape[0], 3))
                colors[:, 0] = z_norm  # 红色分量随高度增加
                colors[:, 2] = 1 - z_norm  # 蓝色分量随高度减少

                point_cloud = open3d.geometry.PointCloud()
                point_cloud.points = open3d.utility.Vector3dVector(points3d)
                point_cloud.colors = open3d.utility.Vector3dVector(colors)
                # open3d.visualization.draw_geometries([point_cloud])

                open3d.io.write_point_cloud(os.path.join(args.save_dir, "LiDAR", filename.replace('.png', '.ply')), point_cloud)

                # fig = plt.figure(figsize=(20,15))
                # ax = fig.add_subplot(111, projection='3d')
                # # ax.view_init(30, - 90 - 90 -8)
                # ax.view_init(0, 0)
                # ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2], c=points3d[:,2], s=2)
                # plt.savefig(os.path.join(args.save_dir, "LiDAR", filename))
                # plt.show()
            
            if message["StencilImage"]!=None and message["StencilImage"]!="":
                print("stencilImage")
                cv2.imwrite(os.path.join(args.save_dir, "StencilImage", filename), bbox_image)
            if message["DepthBuffer"]!=None and message["DepthBuffer"]!="":
                print("DepthBuffer")
                a = np.frombuffer(base64.b64decode(message["DepthBuffer"]), np.float32)
                # Calculate dimensions: sqrt(2073600) = 1440, so the image is likely 1440x1440
                a = a.reshape((1440, 1440))
                
                a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
                a = np.uint8(a)
                print('after', a.max(), a.min())
                cv2.imwrite(os.path.join(args.save_dir, "depth", filename), a)
            if message["LiDARRaycast"]!=None and message["LiDARRaycast"]!="":
                print("LiDARRaycast")
            if message["LiDAR"]!=None and message["LiDAR"]!="":
                print("LiDAR")
            # print("stencilImage", message["StencilImage"]!=None)
            # print("DepthBuffer", message["DepthBuffer"]!=None)
            # print("LiDARRaycast", message["LiDARRaycast"]!=None)

            
        except KeyboardInterrupt:
            break
            
    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    # client.close()



