# -*- coding: utf-8 -*-


import numpy as np
import cv2
import glob
import os

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

#CALIBRATION
folderName = glob.glob("Records/Calibration/*")
folderName.sort()
folderName = folderName[-1]

calib_cases                = ["flat_rot_0",    \
                              "flat_rot_30",    \
                              "flat_rot_60",    \
                              "right_30_rot_0",    \
                              "right_30_rot_30",    \
                              "right_30_rot_60",    \
                              "right_60_rot_0",    \
                              "right_60_rot_30",    \
                              "right_60_rot_60",    \
                            #   "left_30_rot_0",    \
                            #   "left_30_rot_30",    \
                            #   "left_30_rot_60",    \
                            #   "left_60_rot_0",    \
                            #   "left_60_rot_30",    \
                            #   "left_60_rot_60",    \
                              "top_30_rot_0",    \
                              "top_30_rot_30",    \
                              "top_30_rot_60",    \
                              "top_60_rot_0",    \
                              "top_60_rot_30",    \
                              "top_60_rot_60",    \
                            #   "bottom_30_rot_0",    \
                            #   "bottom_30_rot_30",    \
                            #   "bottom_30_rot_60",    \
                            #   "bottom_60_rot_0",    \
                            #   "bottom_60_rot_30",    \
                            #   "bottom_60_rot_60",    \
                              "top-right_45_rot_0",     \
                              "top-right_45_rot_30",    \
                              "top-right_45_rot_60"                     
                            ]
    
numCases = len(calib_cases)
numSamples = 3

# FISHEYE
KLeft = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
DLeft = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])

KRight = np.array([[284.1828918457031, 0.0, 427.9779052734375], [0.0, 285.0440979003906, 399.5506896972656], [0.0, 0.0, 1.0]])
DRight = np.array([0.0009760634857229888, 0.030147459357976913, -0.02769969031214714, 0.0031066760420799255])

fisheyeWidth = 800
fisheyeHeight = 848

# RGB and undistorted fisheyes
# KRgb = np.array([[923.2835693359375, 0.0, 639.9102783203125], [0.0, 923.6146240234375, 370.2297668457031], [0.0, 0.0, 1.0]]) # 1280 x 720
KRgb = np.array([[611.6753646850586, 0.0, 423.94055938720703], [0.0, 615.7430826822916, 246.8198445638021], [0.0, 0.0, 1.0]])  # 848 x 480
D = np.array([0, 0, 0, 0, 0], dtype=np.float32)

# CHECKER board
boardWidth = 8
boardHeight = 8
CHECKERBOARD = (boardHeight, boardWidth)
squareSize = 0.033 # 3.3 cm


fileNamesRgb = []
fileNamesLeft = []
fileNamesRight = []

for name in calib_cases :
    
    fileNamesRgb.append( sorted(glob.glob(folderName + "/" + name  + "/rgb/*")) )
    fileNamesLeft.append( sorted(glob.glob(folderName + "/" + name  + "/leftFisheye/*") ))
    fileNamesRight.append( sorted(glob.glob(folderName + "/" + name  + "/rightFisheye/*")) )

fileNamesLeft.sort()
fileNamesRight.sort()
fileNamesRgb.sort()

sampledRgb = []
sampledLeft = []
sampledRight = []

for i in range(numCases):
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
        

    for index in indexFisheyeLeft   : sampledLeft.append(cv2.imread(fileNamesLeft[i][index], cv2.IMREAD_UNCHANGED))
    for index in indexFisheyeRight  : sampledRight.append(cv2.imread(fileNamesRight[i][index], cv2.IMREAD_UNCHANGED))
    for index in indexRgb           : sampledRgb.append(cv2.imread(fileNamesRgb[i][index], cv2.IMREAD_GRAYSCALE))



sampledLeft, KLeft = undistortFisheyeImages(sampledLeft, KLeft, DLeft)
sampledRight, KRight = undistortFisheyeImages(sampledRight, KRight, DRight)


imagePointsLeft =  []
imagePointsRight =  []
imagePointsRgb =  [] 

imageCount = len(sampledLeft)
for i in range(imageCount):

    imageLeft = sampledLeft[i]
    imageRgb = sampledRgb[i]
    imageRight = sampledRight[i]

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



N_OK = len(imagePointsLeft)
objp = np.zeros((8*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:8].T.reshape(-1, 2)
objp = objp * squareSize
objp = np.array([objp]*len(imagePointsLeft), dtype=np.float32)
objp = np.reshape(objp, (N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 3))


retval1, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R1, T1, E1, F1 = cv2.stereoCalibrate(objp, imagePointsLeft, imagePointsRgb, KLeft, D, KRgb, D, (0,0), flags=cv2.CALIB_FIX_INTRINSIC)
retval2, cameraMatrix3, distCoeffs3, cameraMatrix4, distCoeffs4, R2, T2, E2, F2 = cv2.stereoCalibrate(objp, imagePointsRight, imagePointsRgb, KRight, D, KRgb, D, (0,0), flags=cv2.CALIB_FIX_INTRINSIC)
retval3, cameraMatrix5, distCoeffs5, cameraMatrix6, distCoeffs6, R3, T3, E3, F3 = cv2.stereoCalibrate(objp, imagePointsLeft, imagePointsRight, KLeft, D, KRight, D, (0,0), flags=cv2.CALIB_FIX_INTRINSIC)


file = open("Calibration/Final/" + fileNamesLeft[0][0].split('/')[2] + ".txt", "w")
file.write("\n*Translation from Left Fisheye to RGB:\n")
file.write(str(T1))
file.write("\n*Rotation from Left Fisheye to RGB:\n")
file.write(str(R1))
file.write("\n*Fundamental from Left Fisheye to RGB:\n")
file.write(str(F1))

file.write("\n*Translation from Right Fisheye to RGB:\n")
file.write(str(T2))
file.write("\n*Rotation from Right Fisheye to RGB:\n")
file.write(str(R2))
file.write("\n*Fundamental from Right Fisheye to RGB:\n")
file.write(str(F2))

file.write("\n*Translation from Left Fisheye to Right Fisheye:\n")
file.write(str(T3))
file.write("\n*Rotation from Left Fisheye to Right Fisheye:\n")
file.write(str(R3))
file.write("\n*Fundamental from Left Fisheye to Right Fisheye:\n")
file.write(str(F3))
file.close()




































