# -*- coding: utf-8 -*-


######################################################
############### Step 1 ###############################
######################################################
import re
from ast import literal_eval
import numpy as np
import os
from scipy.spatial.transform import Rotation as rot
import cv2
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



if(os.getcwd().split('/')[-1] != 'star'):
    os.chdir("../../../")
    
def str_to_array(arr):
    arr = re.sub(r"([^[])\s+([^]])", r"\1, \2", arr)
    arr = np.array(literal_eval(arr))
    return arr
    
def normalizeImageAndSave_toTemp(image_, name, colorMap=None):
    image = np.copy(image_)
    image = image / np.max(image) * (2**16-1)
    image = (image/256).astype(np.uint8) 
    if(colorMap != None):
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    cv2.imwrite("./temp/"+ name +".png", image)
    



extrinsics_file = open("Calibration/Missions/Mission_5/extrinsics.txt", "r")
extrinsics_all = extrinsics_file.read()
extrinsics_file.close()
del extrinsics_file

entries = re.split(":+", extrinsics_all)
entries = [entry.split('*') for entry in entries]

KL = np.array([[500. ,   0. , 499.5],
       [  0. , 500. , 499.5],
       [  0. ,   0. ,   1. ]])

KR = np.array([[500. ,   0. , 499.5],
 [  0. , 500. , 499.5],
 [  0.,    0. ,   1. ]])

T_LF2C = entries[1][0]
T_LF2C = str_to_array(T_LF2C)

R_LF2C = entries[2][0]
R_LF2C = str_to_array(R_LF2C)

T_RF2C = entries[3][0]
T_RF2C = str_to_array(T_RF2C)

R_RF2C = entries[4][0]
R_RF2C = str_to_array(R_RF2C)

T_LF2RF = entries[5][0]
T_LF2RF = str_to_array(T_LF2RF)

R_LF2RF = entries[6][0]
R_LF2RF = str_to_array(R_LF2RF)


##################################
##################################
R_C2LF = np.linalg.inv(R_LF2C)
T_C2LF = -np.dot(R_C2LF, T_LF2C)

# CHECKING
T = np.dot(R_RF2C, np.dot(R_LF2RF, T_C2LF) + T_LF2RF) + T_RF2C
R = np.dot(np.dot(R_C2LF, R_LF2RF), R_RF2C)

#################
T4 = np.array([0.014851336367428, 0.0004623454588, 0.000593442469835]) # from RGB to depth(infrared 1)

R4 = np.array([-0.003368464997038, -0.000677574775182, -0.006368808448315, -0.999973833560944]) # from RGB to depth(infrared 1)
R4 = rot.from_quat(R4).as_matrix()
# R4 = np.linalg.inv(R4)
# T4 = np.dot(R4, -T4)

KDep = np.array([[634.013671875, 0.0, 635.5408325195312], [0.0, 634.013671875, 351.9051208496094], [0.0, 0.0, 1.0]])
KDep_inv = np.linalg.inv(KDep)
KRgb = np.array([[923.2835693359375, 0.0, 639.9102783203125], [0.0, 923.6146240234375, 370.2297668457031], [0.0, 0.0, 1.0]])

folderName = "SnapShots/t265_d435i/1_2020-11-03_20:56"
fileNamesDepth = glob.glob(folderName + "/depth/*" )
fileNamesRgb = glob.glob(folderName + "/rgb/*")
fileNamesLeft = glob.glob(folderName + "/leftFisheye_undistorted/*")

fileNamesDepth.sort()
fileNamesRgb.sort()
fileNamesLeft.sort()

imgDep = cv2.imread(fileNamesDepth[1], cv2.IMREAD_UNCHANGED)
imgRgb = cv2.imread(fileNamesRgb[1])
imgLeft = cv2.imread(fileNamesLeft[1])


shape = imgDep.shape
objp = np.ones((shape[0] * shape[1], 3), np.uint32)
objp[:, :2] = np.mgrid[ 0:shape[1],0:shape[0]].T.reshape(-1, 2)
objp = objp.T
worldP_depth = np.dot(KDep_inv, objp)
worldP_depth = worldP_depth.T
worldP_depth = worldP_depth.reshape((shape[0], shape[1], 3) )
imgDep = np.expand_dims(imgDep, axis=2)
worldP_depth = worldP_depth * imgDep * 1e-3

# printPoints = worldP_depth.reshape(shape[0]*shape[1],3).T
# fig = plt.figure()
# ax = Axes3D(fig)

# numberOfPoints = 10000
# randomSampleIndexes = np.random.choice(printPoints.shape[1], numberOfPoints)
# printPoints = printPoints[:, randomSampleIndexes]
# ax.scatter(printPoints[0,:numberOfPoints], printPoints[1,:numberOfPoints], printPoints[2,:numberOfPoints])
# plt.show()


worldP_color = np.dot(R4, worldP_depth.reshape(shape[0]*shape[1], 3).T) + np.expand_dims(T4, 1)
worldP_left = np.dot(R_C2LF, worldP_color) + T_C2LF



imgP_color = np.dot(KRgb, worldP_color)
imgP_color /= imgP_color[2,:]
imgP_color = np.array(np.round(imgP_color), dtype=np.int32)

imgP_left = np.dot(KL, worldP_left)
imgP_left /= imgP_left[2,:]
imgP_left = np.array(np.round(imgP_left), dtype=np.int32)

# fig = plt.figure()
# ax = Axes3D(fig)

# numberOfPoints = 5000
# randomSampleIndexes = np.random.choice(worldP_color.shape[1], numberOfPoints)
# printPoints = worldP_color[:, randomSampleIndexes]
# ax.scatter(printPoints[0,:numberOfPoints], printPoints[1,:numberOfPoints], printPoints[2,:numberOfPoints])
# plt.show()

mappedD2C = np.zeros(imgRgb.shape[:2])
mappedC2D = np.zeros(imgRgb.shape).astype(np.uint8)


for i,pt in enumerate(imgP_color.T) :
    pt = [pt[1], pt[0], 1]
    
    x_depth = objp[0][i]
    y_depth = objp[1][i]
        
    
    if(pt[0] < 720 and pt[1] < 1280 and pt[0] >= 0 and pt[1] >= 0):
        
        depth = worldP_color[2, i]*1000
        
        
        if(imgDep[y_depth, x_depth] > 0 and (mappedD2C[pt[0], pt[1]] == 0 or (depth < mappedD2C[pt[0], pt[1]] and depth > 0 ))):
            mappedD2C[pt[0], pt[1]] = depth
            mappedC2D[y_depth, x_depth, :] = imgRgb[pt[0], pt[1], :]
            
       
        
def mapAtoB(worldP_a, imgP_a, img_a,   worldP_b, imgP_b, img_b):
    
    A_ImagePointsAndCorresponds = np.zeros((img_a.shape[0], img_a.shape[1], 2)).astype(np.uint16)
    B_ImagePointsAndCorresponds = np.zeros((img_b.shape[0], img_b.shape[1], 2)).astype(np.uint16)
    
    mappedImage = np.zeros((img_b.shape[0], img_b.shape[1], img_a.shape[2])).astype(img_a.dtype)
    
    for i,pt in enumerate(imgP_b.T) :
        pt = [pt[1], pt[0], 1]
    
        
        if(pt[0] < 1000 and pt[1] < 1000 and pt[0] >= 0 and pt[1] >= 0):
            depth = worldP_b[2, i] * 1000
                
            if(depth != 0 and (B_ImagePointsAndCorresponds[pt[0], pt[1]][1] == 0 or B_ImagePointsAndCorresponds[pt[0], pt[1]][1] > depth)):
                B_ImagePointsAndCorresponds[pt[0], pt[1]] = [i, depth];
    

    for i,pt in enumerate(imgP_a.T) :
        pt = [pt[1], pt[0], 1]
        
        if(pt[0] < 1000 and pt[1] < 1000 and pt[0] >= 0 and pt[1] >= 0):
            depth = worldP_a[2, i] * 1000
                
            if(depth != 0 and (A_ImagePointsAndCorresponds[pt[0], pt[1]][1] == 0 or A_ImagePointsAndCorresponds[pt[0], pt[1]][1] > depth)):
                A_ImagePointsAndCorresponds[pt[0], pt[1]] = [i, depth];
    

    for row in range(B_ImagePointsAndCorresponds.shape[0]):
        for col in range(B_ImagePointsAndCorresponds.shape[1]):
            
            row_ = A_ImagePointsAndCorresponds[B_ImagePointsAndCorresponds[row, col][0]][0]
            col_ = A_ImagePointsAndCorresponds[B_ImagePointsAndCorresponds[row, col][0]][1]
            
            mappedImage[row, col] = img_a[row_, col_]

    return mappedImage


mappedImage = mapAtoB(worldP_left, imgP_left, imgLeft,   worldP_color, imgP_color, imgRgb)

a = input("input la")

mappedD2L = np.zeros(imgLeft.shape[:2]).astype(np.uint16)
mappedL2D = np.zeros(imgDep.shape[:2]).astype(np.uint8)

leftFisheyeImagePointsAndCorresponds =  np.zeros((imgLeft.shape[0], imgLeft.shape[1], 3)).astype(np.uint16)

for i,pt in enumerate(imgP_left.T) :
    pt = [pt[1], pt[0], 1]
    
    x_depth = objp[0][i]
    y_depth = objp[1][i]
    
    
    
    if(pt[0] < 1000 and pt[1] < 1000 and pt[0] >= 0 and pt[1] >= 0):
        depth = worldP_left[2, i] * 1000
        if(mappedD2L[pt[0], pt[1]] == 0 or (depth < mappedD2L[pt[0], pt[1]] and depth > 0)):
            mappedD2L[pt[0], pt[1]] = depth    
            
        if(depth != 0 and (leftFisheyeImagePointsAndCorresponds[pt[0], pt[1]][2] == 0 or leftFisheyeImagePointsAndCorresponds[pt[0], pt[1]][2] > depth)):
            leftFisheyeImagePointsAndCorresponds[pt[0], pt[1]] = [y_depth, x_depth, depth];

for row in range(leftFisheyeImagePointsAndCorresponds.shape[0]):
    for col in range(leftFisheyeImagePointsAndCorresponds.shape[1]):
        if(leftFisheyeImagePointsAndCorresponds[row, col][2] != 0):
            xDepth = leftFisheyeImagePointsAndCorresponds[row, col][1]
            yDepth = leftFisheyeImagePointsAndCorresponds[row, col][0]
            mappedL2D[yDepth, xDepth] = np.mean(imgLeft[row, col])

        
#normalizeImageAndSave_toTemp(mappedD2C, "generated_depth_color", cv2.COLORMAP_JET)    








# surr = 5

# imgPoint_depth = (600,303,1) # x:593, y:353
# depth = imgDep[imgPoint_depth[1],imgPoint_depth[0]][0]

# imgDep[imgPoint_depth[1]-surr: imgPoint_depth[1]+surr, imgPoint_depth[0]-surr: imgPoint_depth[0]+surr] = 100 - imgDep[imgPoint_depth[1]-surr: imgPoint_depth[1]+surr, imgPoint_depth[0]-surr: imgPoint_depth[0]+surr]
# plt.figure()
# plt.imshow(imgDep)


# worldPoint_depth = np.dot(KDep_inv, imgPoint_depth)
# worldPoint_depth /= worldPoint_depth[2]
# worldPoint_depth *= depth
# worldPoint_depth /= 1000

# worldPoint_rgb = np.dot(R4, worldPoint_depth) + T4
# imgPoint_rgb = np.dot(KRgb, worldPoint_rgb)
# imgPoint_rgb /= imgPoint_rgb[2]
# imgPoint_rgb = np.round(imgPoint_rgb)
# imgPoint_rgb = np.array(imgPoint_rgb, dtype=np.int)
# imgRgb[imgPoint_rgb[1]-surr: imgPoint_rgb[1]+surr, imgPoint_rgb[0]-surr: imgPoint_rgb[0]+surr] = 255 - imgRgb[imgPoint_rgb[1]-surr: imgPoint_rgb[1]+surr, imgPoint_rgb[0]-surr: imgPoint_rgb[0]+surr]
# plt.figure()
# plt.imshow(imgRgb)

# worldPoint_rgb = np.expand_dims(worldPoint_rgb,1)

# worldPoint_left = np.dot(R_C2LF, worldPoint_rgb) + T_C2LF
# imgPoint_left = np.dot(KL, worldPoint_left)
# imgPoint_left /= imgPoint_left[2]
# imgPoint_left = np.array(imgPoint_left, dtype=np.int)

# handPickedLeftImage[imgPoint_left[1][0]-surr : imgPoint_left[1][0]+surr, imgPoint_left[0][0]-surr:imgPoint_left[0][0]+surr] = 255 - handPickedLeftImage[imgPoint_left[0][0]-surr : imgPoint_left[0][0]+surr, imgPoint_left[1][0]-surr:imgPoint_left[1][0]+surr]
# plt.figure()
# plt.imshow(handPickedLeftImage)












