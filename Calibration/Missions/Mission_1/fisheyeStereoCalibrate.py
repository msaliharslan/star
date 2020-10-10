"""
elde var sıfır
"""
import numpy as np
import cv2
import glob
import os


###############
# Common Functions
###############

def calculateFundamentalMatrix(K1, K2,  R,T):

    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3,1)))))
    P2 = np.dot(K2, np.hstack((R, T)))

    #calculating epipole'

    e_hat = np.dot(K2, T)

    #reference vision book page 581
    e_hat_crosser = np.array([  [ 0, -e_hat[2], e_hat[1]], \
                                [e_hat[2], 0, -e_hat[0]], \
                                [-e_hat[1], e_hat[0], 0]    ])


    F = np.dot(e_hat_crosser, np.dot(P2, np.linalg.pinv(P1)))   

    return F 

##############
# End Functions
##############





###############
# Common Variables
###############

K1 = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
D1 = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])

K2 = np.array([[284.1828918457031, 0.0, 427.9779052734375], [0.0, 285.0440979003906, 399.5506896972656], [0.0, 0.0, 1.0]])
D2 = np.array([0.0009760634857229888, 0.030147459357976913, -0.02769969031214714, 0.0031066760420799255])
 
###############
# End Common Variables
###############




###############
# 1 Main
###############


boardWidth = 9
boardHeight = 9
CHECKERBOARD = (boardHeight, boardWidth)
squareSize = 0.03 # 3cm


setNumber = 1
fileNames = glob.glob("../../../SnapShots/t265/syncShots/set" + str(setNumber) + "/*.png" )
imageCount = int(len(fileNames) / 2)

objectPointDefault = []
for i in range(boardHeight): #Note: the order how to fill this objectPointsDefault is not clear, todo: make sure of it
    for j in range(boardWidth):
        objectPointDefault.append([i, j, 0])

objectPointss =  [] 
imagePoints1 =  []
imagePoints2 =  [] 


for i in range(imageCount):

    imageIndex, cameraIndex = fileNames[i].split("_")

    imageIndex = int(imageIndex)
    cameraIndex = int(cameraIndex)

    image = cv2.imread(os.path.join("SnapShots/t265/syncShots", fileNames[i]), cv2.IMREAD_UNCHANGED)

    ret, corners = cv2.findChessboardCorners(image, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

    cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), (cv2.CV_TERMCRIT_EPS + cv2.CV_TERMCRIT_ITER, 30, 0.1))



fisheyeWidth = 800
fisheyeHeight = 848
imageSize = (fisheyeWidth, fisheyeHeight)

flags = cv2.fisheye.CALIB_FIX_INTRINSIC
criteria = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 100, 1e-3)

retval, K1, D1, K2, D2, R_1, T_1 = cv2.fisheye.stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, flags = flags)

F_1 = calculateFundamentalMatrix(K1, K2, R_1, T_1)

###############
# End 1 Main
###############





###############
# 2 Main
###############

import transformation

R_2 = transformation.R
T_2 = transformation.T

F_2 = calculateFundamentalMatrix(K1, K2, R_2, T_2)


###############
# End 2 Main
###############










































