#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:13:02 2020

@author: salih
"""
import os
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

os.chdir("../../../")

K1 = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
D1 = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])

K2 = np.array([[284.1828918457031, 0.0, 427.9779052734375], [0.0, 285.0440979003906, 399.5506896972656], [0.0, 0.0, 1.0]])
D2 = np.array([0.0009760634857229888, 0.030147459357976913, -0.02769969031214714, 0.0031066760420799255])

fisheyeWidth = 800
fisheyeHeight = 848


def undistortFisheyeImages(fisheyeImages, K, D):
    
    
    # We calculate the undistorted focal length:
    #
    #         h
    # -----------------
    #  \      |      /
    #    \    | f  /
    #     \   |   /
    #      \ fov /
    #        \|/
    stereo_fov_rad = 90 * (np.pi/180)  # 90 degree desired fov
    stereo_height_px = 1000          # 300x300 pixel stereo output
    stereo_focal_px = stereo_height_px/2 / np.tan(stereo_fov_rad/2)

    # We set the left rotation to identity and the right rotation
    # the rotation between the cameras
    R = np.eye(3)

    # The stereo algorithm needs max_disp extra pixels in order to produce valid
    # disparity on the desired output region. This changes the width, but the
    # center of projection should be on the center of the cropped image
    stereo_width_px = stereo_height_px
    stereo_size = (stereo_width_px, stereo_height_px)
    stereo_cx = (stereo_height_px - 1)/2
    stereo_cy = (stereo_height_px - 1)/2

    # Construct the left and right projection matrices, the only difference is
    # that the right projection matrix should have a shift along the x axis of
    # baseline*focal_length
    P = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                       [0, stereo_focal_px, stereo_cy, 0],
                       [0,               0,         1, 0]])    
    
    undistortedImages = []
    
    nk = K.copy()

    # nk[0,0]=K[0,0]/2
    # nk[1,1]=K[1,1]/2   
    
    for image in fisheyeImages:
            
        undistorted = cv2.fisheye.undistortImage(image, K=K, D=D, Knew=P, new_size=(int(stereo_height_px), int(stereo_width_px)))
        undistortedImages.append(undistorted)

    return undistortedImages

def drawChessboardCornersForImages(images_):
    
    images = np.copy(images_)
    
    newImages = []
    
    CHECKERBOARD = (8,8) 
    
    for image in images:
        
        # ret, corners = cv2.findChessboardCorners(image, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret, corners = cv2.findChessboardCornersSB(image, CHECKERBOARD, cv2.CALIB_CB_ACCURACY+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # corners = cv2.goodFeaturesToTrack(gray_img,25,0.01,10)
       
        if(ret):
                        
            for corner in corners:
                
                cv2.circle(image, (int(corner[0][0]),int(corner[0][1])), 4,  (255,255,255), thickness=-1,)        
        
            newImages.append(image)
                    
    return newImages


fileNames = glob.glob("/home/salih/Documents/Records/4_2020-11-03_21:02/leftFisheye/*" )
         
distorted = []
for i in range(0, len(fileNames), 25):
    image = cv2.imread(fileNames[i])
    distorted.append(image)
    
test = cv2.imread("Calibration/Missions/Mission_2/left_649_1604426768746.png")
distorted.append(test)
undistorted = undistortFisheyeImages(distorted, K1, D1)

dotted = drawChessboardCornersForImages(undistorted)

try:
    os.mkdir("./temp")
    
except:
    print("temp already exists")
    
for i,img in enumerate(dotted):
    cv2.imwrite("./temp/"+str(i)+".png", img)
    
    
print(len(distorted))
print(len(undistorted))
print(len(dotted))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    