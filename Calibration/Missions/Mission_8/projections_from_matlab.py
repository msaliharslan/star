#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 00:01:55 2021

@author: salih
"""
import os
import numpy as np
import cv2
import glob
import re
from scipy.spatial.transform import Rotation as rot
from ast import literal_eval

if(os.getcwd().split('/')[-1] != 'star'):
    os.chdir("../../../")
    
def str_to_array(arr):
    arr = re.sub(r"([^[])\s+([^]])", r"\1, \2", arr)
    arr = np.array(literal_eval(arr))
    return arr
    

extrinsics_file = glob.glob("Calibration/Final/*")
extrinsics_file.sort()
extrinsics_file = open(extrinsics_file[-1], "r")
extrinsics_all = extrinsics_file.read()
extrinsics_file.close()
del extrinsics_file

entries = re.split(":+", extrinsics_all)
entries = [entry.split('*') for entry in entries]

T_LF2C = entries[1][0]
T_LF2C = str_to_array(T_LF2C)

R_LF2C = entries[2][0]
R_LF2C = str_to_array(R_LF2C)

F_LF2C = entries[3][0]
F_LF2C = str_to_array(F_LF2C)

T_RF2C = entries[4][0]
T_RF2C = str_to_array(T_RF2C)

R_RF2C = entries[5][0]
R_RF2C = str_to_array(R_RF2C)

F_RF2C = entries[6][0]
F_RF2C = str_to_array(F_RF2C)

T_LF2RF = entries[7][0]
T_LF2RF = str_to_array(T_LF2RF)

R_LF2RF = entries[8][0]
R_LF2RF = str_to_array(R_LF2RF)

F_LF2RF = entries[9][0]
F_LF2RF = str_to_array(F_LF2RF)

R_C2LF = np.linalg.inv(R_LF2C)
T_C2LF = -np.dot(R_C2LF, T_LF2C)

R_C2RF = np.linalg.inv(R_RF2C)
T_C2RF = -np.dot(R_C2RF, T_RF2C)


T_D2C = np.array([0.014851336367428, 0.0004623454588, 0.000593442469835]) # from depth to RGB(infrared 1)

R_D2C = np.array([-0.003368464997038, -0.000677574775182, -0.006368808448315, -0.999973833560944]) # from depth to RGB(infrared 1)
R_D2C = rot.from_quat(R_D2C).as_matrix()


KDep = np.array([[634.013671875, 0.0, 635.5408325195312], [0.0, 634.013671875, 351.9051208496094], [0.0, 0.0, 1.0]]) #1280 x 720
# KDep = np.array([[420.0340576171875, 0.0, 421.04580154418943], [0.0, 422.67578125, 234.6034138997396], [0.0, 0.0, 1.0]]) # 848 480
KDep_inv = np.linalg.inv(KDep)
KRgb = np.array([[611.6753646850586, 0.0, 423.94055938720703], [0.0, 615.7430826822916, 246.8198445638021], [0.0, 0.0, 1.0]]) # 848 x 480

# FISHEYE
KL = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
DLeft = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])

KR = np.array([[284.1828918457031, 0.0, 427.9779052734375], [0.0, 285.0440979003906, 399.5506896972656], [0.0, 0.0, 1.0]])
DRight = np.array([0.0009760634857229888, 0.030147459357976913, -0.02769969031214714, 0.0031066760420799255])

KLeft = np.array([[500. ,   0. , 499.5],
                  [  0. , 500. , 499.5],
                  [  0. ,   0. ,   1. ]])

KRight = np.array([[500. ,   0. , 499.5],
                   [  0. , 500. , 499.5],
                   [  0.,    0. ,   1. ]])
# read inputs
folderName = "SnapShots/t265_d435i/1_2020-11-03_20:56"
fileNamesDepth = glob.glob(folderName + "/depth/*" )
fileNamesRgb = glob.glob(folderName + "/rgb/*")
fileNamesLeft = glob.glob(folderName + "/leftFisheye_undistorted/*")
fileNamesRight = glob.glob(folderName + "/rightFisheye_undistorted/*")


fileNamesDepth.sort()
fileNamesRgb.sort()
fileNamesLeft.sort()
fileNamesRight.sort()

imgDep = cv2.imread(fileNamesDepth[1], cv2.IMREAD_UNCHANGED)
imgRgb = cv2.imread(fileNamesRgb[1], cv2.IMREAD_UNCHANGED)
imgLeft = cv2.imread(fileNamesLeft[1], cv2.IMREAD_UNCHANGED)
imgRight = cv2.imread(fileNamesRight[1] , cv2.IMREAD_UNCHANGED)

[height_D, width_D] = imgDep.shape
[height_C, width_C, _] = imgRgb.shape
[height_L, width_L] = imgLeft.shape

# display images
# figure
# subplot(1,3,1), imagesc(D), title('depth'), axis('image')
# subplot(1,3,2), imshow(C), title('color camera')
# subplot(1,3,3), imshow(L), title('undistorted left fisheye')

# find world coordinates on depth camera, D
shape = imgDep.shape
objp = np.ones((shape[0] * shape[1], 3), np.uint32)
objp[:, :2] = np.mgrid[ 0:shape[1],0:shape[0]].T.reshape(-1, 2)
objp = objp.T
worldP_depth = np.dot(KDep_inv, objp)
worldP_depth = worldP_depth.T
worldP_depth = worldP_depth.reshape((shape[0], shape[1], 3) )
imgDep = np.expand_dims(imgDep, axis=2)
worldP_depth = worldP_depth * imgDep * 1e-3


"""
x = ones(height_D, 1) * (1 : width_D); # image plane coordinates
y = (1 : height_D)' * ones(1, width_D);
z = D(:)'; # measured depth

pointIND_D = [x(:)' + 1; y(:)']; # depth image index for each point

points_D = [pointIND_D - 1; ones(1, height_D * width_D)]; # normalized coordinates (0-based)
points_D = K_D \ points_D; # back projection from normalized coordinates

for i = 1 : 3
    points_D(i, :) = points_D(i, :) .* z; # scaling with depth
end

measured_D = (D(:) ~= 0)'; # measurement available flag 3D point cloud

points_D = points_D(:, measured_D); # measured 3D points in depth camera cordinate system
pointIND_D = pointIND_D(:, measured_D); # image index [x, y] for 3D points on depth camera

# project point cloud onto color camera and find color camera image index for each point 
# pointIMG_C : color camera image of [x, y, z] points
# visibleD_C : visibility flag for depth on color camera image
# pointIND_C : color camera index of points_D
# pointVisible_C : visibility flag of points_D on color camera
[pointIMG_C, pointIND_C, pointVisible_C] = PointCloudProjection(points_D, R_D2C, t_D2C, K_C, width_C, height_C);

# back project color image onto depth camera plane
[imgC_D, measuredC_D] = ImageBackProjection(C, pointIND_C(:, pointVisible_C), pointIND_D(:, pointVisible_C), width_D, height_D);


# project point cloud onto left camera and find left camera image index for each point 
[pointIMG_L, pointIND_L, pointVisible_L] = PointCloudProjection(points_D, R_D2L, t_D2L, K_L, width_L, height_L);

# back project left image onto depth camera plane
[imgL_D, measuredL_D] = ImageBackProjection(L, pointIND_L(:, pointVisible_L), pointIND_D(:, pointVisible_L), width_D, height_D);


# prepare images for display/record
imgD = uint8((D - min(D(:))) / (max(D(:)) - min(D(:))) * 255);

imgD_C = pointIMG_C(:, :, 3);
imgD_C = uint8((imgD_C - min(imgD_C(:))) / (max(imgD_C(:)) - min(imgD_C(:))) * 255);

imgC_D = uint8(imgC_D);

imgD_L = pointIMG_L(:, :, 3);
imgD_L = uint8((imgD_L - min(imgD_L(:))) / (max(imgD_L(:)) - min(imgD_L(:))) * 255);

imgL_D = uint8(imgL_D);

toc
# figure, imshow(imgD), title('Depth Image on D')
# figure, imshow(imgC_D), title('Color Image on D')
# figure, imshow(imgL_D), title('Left Image on D')
# 
# figure, imshow(C), title('Color Image on C')
# figure, imshow(imgD_C), title('Depth Image on C')
# 
# figure, imshow(L), title('Left Image on L')
# figure, imshow(imgD_L), title('Depth Image on L')


# imwrite(imgD, 'sampleDepthDepth.png');
# imwrite(imgC_D, 'sampleDepthRgb.png');
# imwrite(imgL_D, 'sampleDepthLeft.png');
# 
# imwrite(imgD_C, 'sampleRgbDepth.png');
# 
# imwrite(imgD_L, 'sampleLeftDepth.png');

"""



