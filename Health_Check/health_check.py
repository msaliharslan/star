#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 22:32:19 2021

@author: salih
"""
import os
import numpy as np
import cv2    
import glob
import re
from ast import literal_eval

if(os.getcwd().split('/')[-1] != 'star'):
    os.chdir("../")
    
def undistortFisheyeImages(fisheyeImages, K, D):
    """

    Parameters
    ----------
    fisheyeImages : Image(numpy array)
    K : Camera Matrix numpy array
    D : Kannala-Brandt distortion parameters numpy array

    Returns
    -------
    undistortedImages : Undistorted images and new camera matrix


    """
    # https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/t265_stereo.py
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
    stereo_height_px = 1000          # 1000x1000 pixel stereo output
    stereo_focal_px = stereo_height_px/2 / np.tan(stereo_fov_rad/2)


    # The stereo algorithm needs max_disp extra pixels in order to produce valid
    # disparity on the desired output region. This changes the width, but the
    # center of projection should be on the center of the cropped image
    stereo_width_px = stereo_height_px
    stereo_cx = (stereo_height_px - 1)/2
    stereo_cy = (stereo_height_px - 1)/2

    # Construct the left and right projection matrices, the only difference is
    # that the right projection matrix should have a shift along the x axis of
    # baseline*focal_length
    P = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                       [0, stereo_focal_px, stereo_cy, 0],
                       [0,               0,         1, 0]])    
    
    undistortedImages = []
    
    # nk = K.copy()
    # nk[0,0]=K[0,0]/2
    # nk[1,1]=K[1,1]/2   
    
    for image in fisheyeImages:
            
        undistorted = cv2.fisheye.undistortImage(image, K=K, D=D, Knew=P, new_size=(int(stereo_height_px), int(stereo_width_px)))
        undistortedImages.append(undistorted)

    return undistortedImages, P[:,:-1]

def matchCornerOrder_v2(corners1, corners2):
    """
    Parameters
    ----------
    corners1 : TYPE
        DESCRIPTION.
    corners2 : TYPE
        DESCRIPTION.

    Returns
    -------
    corners1 : TYPE
        DESCRIPTION.
    corners2 : TYPE
        DESCRIPTION.

    """
    cntr = 0
    corners1 = corners1.reshape(8, 8, 1, 2)
    corners2 = corners2.reshape(8, 8, 1, 2)
    xc1 = np.mean(corners1[:,:,:,0])
    yc1 = np.mean(corners1[:,:,:,1])
    xc2 = np.mean(corners2[:,:,:,0])
    yc2 = np.mean(corners2[:,:,:,1])
    
    v1 = np.array((corners1[0,0,:,0] - xc1, corners1[0,0,:,1] - yc1)) 
    v2 = np.array((corners2[0,0,:,0] - xc2, corners2[0,0,:,1] - yc2))
    theta = np.arccos(np.dot(v1.T, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) ) )
    while(np.abs(theta) > np.pi/9):
        #something
        corners2 = np.rot90(corners2, 1, (0, 1))
        v2 = np.array((corners2[0,0,:,0] - xc2, corners2[0,0,:,1] - yc2))
        theta = np.arccos(np.dot(v1.T, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) ) )
        cntr +=1
        if(cntr == 4):
            corners2 = np.flip(corners2, 0)
            print("FLIIIIIIP!!!")
        if(cntr==8):
            print("Cannot match corner order")
            break
        
    corners1 = corners1.reshape(64, 1, 2)
    corners2 = corners2.reshape(64, 1, 2)
    return corners1, corners2

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,_ = img1.shape
    np.random.seed(0)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        pt1 = tuple(map(tuple, pt1))[0]
        pt2 = tuple(map(tuple, pt2))[0]
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,pt1,5,color,-1)
    return img1,img2

def str_to_array(arr):
    """
    
    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.

    Returns
    -------
    arr : TYPE
        DESCRIPTION.

    """
    arr = re.sub(r"([^[])\s+([^]])", r"\1, \2", arr)
    arr = np.array(literal_eval(arr))
    return arr

def calculateFundamentalMatrix(K1, K2, R, T):

    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3,1)))))
    P2 = np.dot(K2, np.hstack((R, T)))

    #calculating epipole'

    e_hat = np.dot(K2, T)

    #reference vision book page 581
    e_hat_crosser = np.array([  [ 0, -e_hat[2], e_hat[1]], \
                                [e_hat[2], 0, -e_hat[0]], \
                                [-e_hat[1], e_hat[0], 0]    ], dtype=np.float32)


    F = np.dot(e_hat_crosser, np.dot(P2, np.linalg.pinv(P1)))   

    return F 

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

# CHECKING
T = np.dot(R_RF2C, np.dot(R_LF2RF, T_C2LF) + T_LF2RF) + T_RF2C
assert np.linalg.norm(T) < 1e-2

R = np.dot(np.dot(R_C2LF, R_LF2RF), R_RF2C)
I = np.identity(3, dtype = R.dtype)
n = np.linalg.norm(I - R)
assert n < 2 * 1e-2

# FISHEYE
KL = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
DL = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])

KR = np.array([[284.1828918457031, 0.0, 427.9779052734375], [0.0, 285.0440979003906, 399.5506896972656], [0.0, 0.0, 1.0]])
DR = np.array([0.0009760634857229888, 0.030147459357976913, -0.02769969031214714, 0.0031066760420799255])

fisheyeWidth = 800
fisheyeHeight = 848

# RGB and undistorted fisheyes
# KRgb = np.array([[923.2835693359375, 0.0, 639.9102783203125], [0.0, 923.6146240234375, 370.2297668457031], [0.0, 0.0, 1.0]]) # 1280 x 720
KRgb = np.array([[611.6753646850586, 0.0, 423.94055938720703], [0.0, 615.7430826822916, 246.8198445638021], [0.0, 0.0, 1.0]]) # 848 x 480
D = np.array([0, 0, 0, 0, 0], dtype=np.float32)

# CHECKER board
boardWidth = 8
boardHeight = 8
CHECKERBOARD = (boardHeight, boardWidth)
squareSize = 0.033 # 3.3 cm


folderName = glob.glob("Records/Health_Check/*")
folderName.sort()
folderName = folderName[-1]

folderNames = glob.glob(folderName + "/*")
folderNames.sort()

levels = len(folderNames)
numSamples = 3
err_threshold = 0.6
health_check = True

fileNamesRgb = []
fileNamesLeft = []
fileNamesRight = []

for folderName in folderNames:
    
    fileNamesRgb.append( sorted(glob.glob(folderName + "/rgb/*")) )
    fileNamesLeft.append( sorted(glob.glob(folderName + "/leftFisheye/*") ))
    fileNamesRight.append( sorted(glob.glob(folderName + "/rightFisheye/*")) )

try:
    os.mkdir("Health_Check/" + fileNamesLeft[0][0].split('/')[2])
except:
    print("Save folder exists")
    
sampledRgb = []
sampledLeft = []
sampledRight = []


for i in range(levels):
    indexRgb = np.random.choice( range(len(fileNamesRgb[i])), numSamples )
    indexFisheyeLeft = []
    indexFisheyeRight = []
    for index in indexRgb:
        time = int(fileNamesRgb[i][index].split(".")[0].split("_")[-1])
        
        closestTime = 1e20
        closestIndex = None
        for j, name in enumerate(fileNamesLeft[i]):
            timeLeft = int(name.split(".")[0].split("_")[-1])
            if(abs(timeLeft - time) < closestTime):
                closestTime = abs(timeLeft - time)
                closestIndex = j
        indexFisheyeLeft.append(closestIndex)
        
        closestTime = 1e20
        closestIndex = None
        for j, name in enumerate(fileNamesRight[i]):
            timeRight = int(name.split(".")[0].split("_")[-1])
            if(abs(timeRight - time) < closestTime):
                closestTime = abs(timeRight - time)
                closestIndex = j
        indexFisheyeRight.append(closestIndex)
        
    temp=[]
    for index in indexFisheyeLeft   : temp.append(cv2.imread(fileNamesLeft[i][index], cv2.IMREAD_UNCHANGED))
    temp, KLeft = undistortFisheyeImages(temp, KL, DL)
    sampledLeft.append(temp)
    
    temp=[]
    for index in indexFisheyeRight  : temp.append(cv2.imread(fileNamesRight[i][index], cv2.IMREAD_UNCHANGED))
    temp, KRight = undistortFisheyeImages(temp, KR, DR)
    sampledRight.append(temp)
   
    temp=[]
    for index in indexRgb           : temp.append(cv2.imread(fileNamesRgb[i][index], cv2.IMREAD_GRAYSCALE))
    sampledRgb.append(temp)

# F_LF2RF = calculateFundamentalMatrix(KLeft, KRight, R_LF2RF, T_LF2RF)
# F_LF2C = calculateFundamentalMatrix(KLeft, KRgb, R_LF2C, T_LF2C)
# F_RF2C = calculateFundamentalMatrix(KRight, KRgb, R_RF2C, T_RF2C)

for j in range(levels):
    imagePointsLeft = []
    imagePointsRight = []
    imagePointsRgb = [] 
    imgsRgbWP = []
    imgsLWP = []
    imgsRWP = []
    for i in range(len(sampledLeft[j])): 

        imageLeft = sampledLeft[j][i]
        imageRgb = sampledRgb[j][i]
        imageRight = sampledRight[j][i]
    
        retL, cornersL = cv2.findChessboardCornersSB(imageLeft, CHECKERBOARD, cv2.CALIB_CB_ACCURACY+cv2.CALIB_CB_NORMALIZE_IMAGE)
        retR, cornersR = cv2.findChessboardCornersSB(imageRight, CHECKERBOARD, cv2.CALIB_CB_ACCURACY+cv2.CALIB_CB_NORMALIZE_IMAGE)
        retRgb, cornersRgb = cv2.findChessboardCornersSB(imageRgb, CHECKERBOARD, cv2.CALIB_CB_ACCURACY+cv2.CALIB_CB_NORMALIZE_IMAGE)
    
        if(retL and retRgb and retR):
            cv2.cornerSubPix(imageLeft, cornersL, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cv2.cornerSubPix(imageRight, cornersR, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cv2.cornerSubPix(imageRgb, cornersRgb, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cornersL, cornersR = matchCornerOrder_v2(cornersL, cornersR)
            cornersL, cornersRgb = matchCornerOrder_v2(cornersL, cornersRgb)
            
            #fix attempt
            imagePointsLeft.append(cornersL)
            imagePointsRight.append(cornersR)
            imagePointsRgb.append(cornersRgb)
            imgsLWP.append(cv2.cvtColor(imageLeft, cv2.COLOR_GRAY2RGB))
            imgsRWP.append(cv2.cvtColor(imageRight, cv2.COLOR_GRAY2RGB))
            imgsRgbWP.append(cv2.cvtColor(imageRgb, cv2.COLOR_GRAY2RGB))
        else:
            print("Cannot find image points for level %d, image %d" % (j, i) )
            
    N_OK = len(imagePointsLeft)
    objp = np.zeros((8*8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:8].T.reshape(-1, 2)
    objp = objp * squareSize
    objp = np.array([objp]*len(imagePointsLeft), dtype=np.float32)
    objp = np.reshape(objp, (N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 3))
        
    try:    
        os.mkdir("Health_Check/" + fileNamesLeft[0][0].split('/')[2] + "/level_" + str(j) )
    except:
        print("Level folder exists")


    err_file = open("Health_Check/" + fileNamesLeft[0][0].split('/')[2] + "/level_" + str(j) +"/errors.txt", 'w')
    for i in range(len(imagePointsLeft)):
        
        #### Left Fisheye - RGB
        
        lines_C2LF = cv2.computeCorrespondEpilines(imagePointsRgb[i].reshape(-1,1,2), 2, F_LF2C)
        lines_C2LF = lines_C2LF.reshape(-1,3)
        img5,img6 = drawlines(np.copy(imgsLWP[i]), imgsRgbWP[i],lines_C2LF, imagePointsLeft[i], imagePointsRgb[i])
        
        lines_LF2C = cv2.computeCorrespondEpilines(imagePointsLeft[i].reshape(-1,1,2), 1, F_LF2C)
        lines_LF2C = lines_LF2C.reshape(-1,3)
        img3,img4 = drawlines(np.copy(imgsRgbWP[i]), imgsLWP[i], lines_LF2C, imagePointsRgb[i], imagePointsLeft[i])
        
        h1, w1 = img3.shape[:2]
        h2, w2 = img5.shape[:2]
    
        #create empty matrix
        img = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
        
        #combine 2 images
        img[:h1, :w1,:3] = img3
        img[:h2, w1:w1+w2,:3] = img5
        cv2.imwrite("Health_Check/" + fileNamesLeft[0][0].split('/')[2] + "/level_" + str(j) + "/LF_RGB_" + str(i) +".png", img)
        
        
        ####################################
        ## Error calculation
        ######################
        
        err = 0
        
        for pt, l in zip(imagePointsLeft[i], lines_C2LF):
            err += abs(pt[0,0] * l[0] + pt[0,1] * l[1] + l[2])
            
        for pt, l in zip(imagePointsRgb[i], lines_LF2C):
            err += abs(pt[0,0] * l[0] + pt[0,1] * l[1] + l[2])
                   
        err /= lines_C2LF.shape[0] + lines_LF2C.shape[0]
        err_file.write("LF_RGB_Image_" + str(i) + "_error = " + str(err) + '\n' )

        if(err > err_threshold):
            print("Error for level " + str(j) + " image LF-RGB " + str(i) + " is larger than threshold. Error = " + str(err) )
            health_check = False


        #### Left Fisheye - Right Fisheye
        
        lines_RF2LF = cv2.computeCorrespondEpilines(imagePointsRight[i].reshape(-1,1,2), 2, F_LF2RF)
        lines_RF2LF = lines_RF2LF.reshape(-1,3)
        img5,img6 = drawlines(np.copy(imgsLWP[i]), imgsRWP[i],lines_RF2LF, imagePointsLeft[i], imagePointsRight[i])
        
        lines_LF2RF = cv2.computeCorrespondEpilines(imagePointsLeft[i].reshape(-1,1,2), 1, F_LF2RF)
        lines_LF2RF = lines_LF2RF.reshape(-1,3)
        img3,img4 = drawlines(np.copy(imgsRWP[i]), imgsLWP[i], lines_LF2RF, imagePointsRight[i], imagePointsLeft[i])
        
        h1, w1 = img3.shape[:2]
        h2, w2 = img5.shape[:2]
    
        #create empty matrix
        img = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
        
        #combine 2 images
        img[:h1, :w1,:3] = img3
        img[:h2, w1:w1+w2,:3] = img5
        cv2.imwrite("Health_Check/" + fileNamesLeft[0][0].split('/')[2] + "/level_" + str(j) + "/LF_RF_" + str(i) +".png", img)
        
        
        ####################################
        ## Error calculation
        ######################
        
        err = 0
        
        for pt, l in zip(imagePointsLeft[i], lines_RF2LF):
            err += abs(pt[0,0] * l[0] + pt[0,1] * l[1] + l[2])
            
        for pt, l in zip(imagePointsRight[i], lines_LF2RF):
            err += abs(pt[0,0] * l[0] + pt[0,1] * l[1] + l[2])
                   
        err /= lines_C2LF.shape[0] + lines_LF2C.shape[0]
        err_file.write("LF_RF_Image_" + str(i) + "_error = " + str(err) + '\n' )
        
        if(err > err_threshold):
            print("Error for level " + str(j) + " image LF-RF " + str(i) + " is larger than threshold. Error = " + str(err) )
            health_check = False

        
         #### Right Fisheye - RGB
         
        lines_C2RF = cv2.computeCorrespondEpilines(imagePointsRgb[i].reshape(-1,1,2), 2, F_RF2C)
        lines_C2RF = lines_C2RF.reshape(-1,3)
        img5,img6 = drawlines(np.copy(imgsRWP[i]), imgsRgbWP[i],lines_C2RF, imagePointsRight[i], imagePointsRgb[i])
        
        lines_RF2C = cv2.computeCorrespondEpilines(imagePointsRight[i].reshape(-1,1,2), 1, F_RF2C)
        lines_RF2C = lines_RF2C.reshape(-1,3)
        img3,img4 = drawlines(np.copy(imgsRgbWP[i]), imgsRWP[i], lines_RF2C, imagePointsRgb[i], imagePointsRight[i])
        
        h1, w1 = img3.shape[:2]
        h2, w2 = img5.shape[:2]
    
        #create empty matrix
        img = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
        
        #combine 2 images
        img[:h1, :w1,:3] = img3
        img[:h2, w1:w1+w2,:3] = img5
        cv2.imwrite("Health_Check/" + fileNamesLeft[0][0].split('/')[2] + "/level_" + str(j) + "/RF_RGB_" + str(i) +".png", img)
        
        
        ####################################
        ## Error calculation
        ######################
        
        err = 0
        
        for pt, l in zip(imagePointsRight[i], lines_C2RF):
            err += abs(pt[0,0] * l[0] + pt[0,1] * l[1] + l[2])
            
        for pt, l in zip(imagePointsRgb[i], lines_RF2C):
            err += abs(pt[0,0] * l[0] + pt[0,1] * l[1] + l[2])
                   
        err /= lines_C2RF.shape[0] + lines_RF2C.shape[0]
        err_file.write("LF_RF_Image_" + str(i) + "_error = " + str(err) + '\n\n' )
        
        if(err > err_threshold):
            print("Error for level " + str(j) + " image RF-RGB " + str(i) + " is larger than threshold. Error = " + str(err) )
            health_check = False


if health_check :
    print("Health Check test passed!!")



































