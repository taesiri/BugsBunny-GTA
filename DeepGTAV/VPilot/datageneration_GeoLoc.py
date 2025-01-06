#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.Constants import IMG_WIDTH, IMG_HEIGHT


from deepgtav.messages import *
from deepgtav.client import Client

from utils.BoundingBoxes import add_bboxes, parseBBox2d_LikePreSIL, parseBBoxesVisDroneStyle, parseBBox_YoloFormatStringToImage
from utils.utils import save_image_and_bbox, save_image, save_meta_data, getRunCount, generateNewTargetLocation
from utils.colors import pickRandomColor

import argparse
import time
import cv2

import matplotlib.pyplot as plt

from PIL import Image

from random import uniform
import random

from math import sqrt
import math
import numpy as np

import os
import sys



def gaussin_random_truncted(lower_bound, upper_bound, mean, std_dev):
    number = random.gauss(mean, std_dev)
    number = max(number, lower_bound)
    number = min(number, upper_bound)
    return number


def euler_to_rotation_matrix(pitch, roll, yaw):
    # Convert angles from degrees to radians
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    yaw = np.radians(yaw)
    
    # Rotation matrix around x-axis (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    # Rotation matrix around y-axis (roll)
    Ry = np.array([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)]
    ])
    
    # Rotation matrix around z-axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    return R


def calculate_projection_points(height, rot_x, rot_y, rot_z, temp_x, temp_y, hfov=60, vfov=38.9):

    # rot_x, rot_y, rot_z represents pitch, roll, and yaw in eular system

    # Convert angles from degrees to radians
    hfov_rad = math.radians(hfov)
    vfov_rad = math.radians(vfov)
    rot_x = abs(rot_x + 90)
    tilt_angle_rad = math.radians(rot_x)

    # print(hfov_rad, vfov_rad, tilt_angle_rad)
    
    # Calculate the width and length of the projection on the ground
    W = 2 * height * math.tan(hfov_rad / 2)
    L = 2 * height * math.tan(vfov_rad / 2)
    
    # Calculate the shift in the projection center due to the tilt angle
    D = height * math.tan(tilt_angle_rad)
    
    # Calculate the four corner points
    P1 = (-W / 2, -L / 2 + D)
    P2 = (W / 2, -L / 2 + D)
    P3 = (-W / 2, L / 2 + D)
    P4 = (W / 2, L / 2 + D)
    relative_points = [
        [-W / 2, -L / 2 + D, 0],
        [W / 2, -L / 2 + D, 0],
        [-W / 2, L / 2 + D, 0],
        [W / 2, L / 2 + D, 0]
    ]
    # print(relative_points)

    R = euler_to_rotation_matrix(pitch=rot_x, roll=rot_y, yaw=rot_z)

    actual_points = []
    for point in relative_points:
        rotated_point = R @ np.array(point)
        actual_x = temp_x + rotated_point[0]
        actual_y = temp_y + rotated_point[1]
        actual_points.append(actual_x)
        actual_points.append(actual_y)

    return actual_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    parser.add_argument('-s', '--save_dir', default='C:\\workspace\\exported_data\\GTA-UAV\\Captured\\randcam2_std0_stable_5area', help='The directory the generated data is saved to')
    args = parser.parse_args()

    client = Client(ip=args.host, port=args.port)
    # voltic
    # scenario = Scenario(drivingMode=786603, vehicle="buzzard", location=[245.23306274414062, -998.244140625, 29.205352783203125], spawnedEntitiesDespawnSeconds=200)
    scenario = Scenario(drivingMode=[786603,0], vehicle="voltic", location=[245.23306274414062, -998.244140625, 29.205352783203125], spawnedEntitiesDespawnSeconds=200)
    dataset = Dataset(location=True, time=True, exportBBox2D=True)
    client.sendMessage(Start(scenario=scenario, dataset=dataset))
    message = client.recvMessage()
    
    # f = open('log.txt', 'w')
    # sys.stdout = f

    CAMERA_OFFSET_Z = -5
    CAMERA_OFFSET_ROT_Z = 20
    TRAVEL_HEIGHT = 50
    # TRAVEL_HEIGHT_LIST = [100, 200, 300, 400, 500, 600]
    TRAVEL_HEIGHT_LIST = [20]  # Much lower starting height
    TRAVEL_HEIGHT_ATEMPT = 100

    # Modified camera rotation ranges for low-altitude view
    CAMERA_ROT_X = -45
    CAMERA_ROT_X_L = -60
    CAMERA_ROT_X_R = -30

    CAMERA_ROT_Y = 0    
    CAMERA_ROT_Y_L = -20  # Increased range from -10 to -20
    CAMERA_ROT_Y_R = 20   # Increased range from 10 to 20

    CAMERA_ROT_Z = 0    
    CAMERA_ROT_Z_L = -180
    CAMERA_ROT_Z_R = 180

    STD_DEV = 5
    ERROR_EPS = 10

    rot_x = CAMERA_ROT_X
    rot_y = CAMERA_ROT_Y
    rot_z = CAMERA_ROT_Z + CAMERA_OFFSET_ROT_Z

    step = 100
    # STEP_LIST = [50, 100, 150, 200, 250, 300]
    STEP_LIST = [300]


    # Adjustments for recording
    #  from UAV perspective
    # client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x = uniform(CAMERA_ROT_X_LOW, CAMERA_ROT_X_HIGH)))
    client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x=rot_x, rot_z=rot_z, rot_y=rot_y))
    message = client.recvMessage()
    print('start camera', message['CameraAngle'])


    xAr_min, xAr_max, yAr_min, yAr_max = -3418, 3945, -3370, 7251

    x_step = step
    y_step = step
    x_y_list = [
        # [-2480, 1764, -3349, 1304],
        # [-361, 3171, 2283, 3889],
        # [1572, 3064, 4168, 5311],
        # [-1007, 798, 5761, 6745],
        # [-3316, -2135, -28, 1589]

        # [-3418, 3945, -3370, 3740]

        [-1431, -1000, -877, -500]  # Replace with your desired coordinates [x_start, x_end, y_start, y_end]
    ]
    # x_start, x_end = -1700, 1599
    # y_start, y_end = -2586, 710
    # x_start, x_end = 245, 1000
    # y_start, y_end = -998, 100
    z_loc = 0
    # x_target, y_target = generateNewTargetLocation(xAr_min, xAr_max, yAr_min, yAr_max)

    # run_count = getRunCount(save_dir)
    run_count = 1
    # weather = random.choice(["CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST"])
    count = 28339

    for TRAVEL_HEIGHT, step in zip(TRAVEL_HEIGHT_LIST, STEP_LIST):

        save_dir = f'{args.save_dir}'
        save_dir = os.path.normpath(save_dir)

        if not os.path.exists(os.path.join(save_dir, 'images')):
            os.makedirs(os.path.join(save_dir, 'images'))
        if not os.path.exists(os.path.join(save_dir, 'labels')):
            os.makedirs(os.path.join(save_dir, 'labels'))
        if not os.path.exists(os.path.join(save_dir, 'meta_data')):
            os.makedirs(os.path.join(save_dir, 'meta_data'))

        for i in range(len(x_y_list)):
            x_start, x_end, y_start, y_end = x_y_list[i]
            for x_temp in range(x_start, x_end, x_step):
                for y_temp in range(y_start, y_end, y_step):

                    for f in range(40):  # Increased from 20 to 40 for longer capture
                        if f == 1:
                            weather = "CLEAR"
                            client.sendMessage(SetWeather(weather))
                            message = client.recvMessage()
                        
                        elif f == 2:
                            client.sendMessage(SetClockTime(12))
                            message = client.recvMessage()

                        elif f == 3:
                            client.sendMessage(TeleportToLocation(x_temp, y_temp, TRAVEL_HEIGHT_ATEMPT))
                            message = client.recvMessage()

                            heightAboveGround = message['HeightAboveGround']
                            z_loc = message['location'][2]
                            z_ground = z_loc - heightAboveGround
                            z_loc = z_ground + TRAVEL_HEIGHT - CAMERA_OFFSET_Z
                            z_temp = z_ground + TRAVEL_HEIGHT

                        # First capture sequence
                        elif f in range(7, 20):  # Extended capture window
                            if f == 7:
                                rot_x = gaussin_random_truncted(CAMERA_ROT_X_L, CAMERA_ROT_X_R, CAMERA_ROT_X, STD_DEV)
                                rot_y = gaussin_random_truncted(CAMERA_ROT_Y_L, CAMERA_ROT_Y_R, CAMERA_ROT_Y, STD_DEV)
                                rot_z = random.randint(CAMERA_ROT_Z_L, CAMERA_ROT_Z_R)
                                client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x=rot_x, rot_y=rot_y, rot_z=rot_z))
                            elif f == 9:
                                client.sendMessage(StartRecording())
                                message = client.recvMessage()
                                heightAboveGround_1 = message['HeightAboveGround']
                            elif f == 19:  # Extended recording time
                                client.sendMessage(StopRecording())
                                # ... rest of the capture logic ...

                        # Second capture sequence
                        elif f in range(25, 38):  # Extended capture window
                            if f == 25:
                                rot_x = gaussin_random_truncted(CAMERA_ROT_X_L, CAMERA_ROT_X_R, CAMERA_ROT_X, STD_DEV)
                                rot_y = gaussin_random_truncted(CAMERA_ROT_Y_L, CAMERA_ROT_Y_R, CAMERA_ROT_Y, STD_DEV)
                                rot_z = random.randint(CAMERA_ROT_Z_L, CAMERA_ROT_Z_R)
                                client.sendMessage(SetCameraPositionAndRotation(z=CAMERA_OFFSET_Z, rot_x=rot_x, rot_y=rot_y, rot_z=rot_z))
                            elif f == 27:
                                client.sendMessage(StartRecording())
                                message = client.recvMessage()
                                heightAboveGround_3 = message['HeightAboveGround']
                            elif f == 37:  # Extended recording time
                                client.sendMessage(StopRecording())
                                # ... rest of the capture logic ...

                        else:
                            message = client.recvMessage()
            
    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    client.close()

