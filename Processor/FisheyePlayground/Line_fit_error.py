# -*- coding: utf-8 -*-
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir("../../")


CHECKERBOARD = (8, 8)
fisheyeWidth = 800
fisheyeHeight = 848

K1 = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
D1 = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])

K2 = np.array([[284.1828918457031, 0.0, 427.9779052734375], [0.0, 285.0440979003906, 399.5506896972656], [0.0, 0.0, 1.0]])
D2 = np.array([0.0009760634857229888, 0.030147459357976913, -0.02769969031214714, 0.0031066760420799255])
 
Kleft = np.array([[287.96022483,   0,         429.5041186 ],   \
                  [  0,         287.69152649, 395.0184248 ],   \
                  [  0,           0,           1.        ]])

Dleft = np.array( [-0.0084311, 0.04829486, -0.04519879, 0.00819492 ] )

Kright = np.array( [[287.20410723,   0,          427.68769401],  \
                    [  0,       286.87899807,    400.11660558],  \
                    [  0,            0,            1.        ]] )
    
Dright = np.array( [-0.01576667, 0.06869806, -0.06550856, 0.01515176] )

sampleImg = cv2.imread("fisheyeSnapshots/qualityShots/leftEye/11_Fisheye.png", cv2.IMREAD_UNCHANGED)

fisheye1Frames = []
fisheye2Frames = []

for filename in os.listdir("fisheyeSnapshots/qualityShots/leftEye"):
    img = cv2.imread(os.path.join("fisheyeSnapshots/qualityShots/leftEye",filename), cv2.IMREAD_UNCHANGED)
    if img is not None:
        fisheye1Frames.append(img)


for filename in os.listdir("fisheyeSnapshots/qualityShots/rightEye"):
    img = cv2.imread(os.path.join("fisheyeSnapshots/qualityShots/rightEye",filename), cv2.IMREAD_UNCHANGED)
    if img is not None:
        fisheye2Frames.append(img)
        


def distanceToLine(coefs, point):
    
    return np.abs(coefs[0] * point[0] - point[1] + coefs[1] ) / np.sqrt(coefs[0]**2 + 1)


def findBestFitLinesAndErrors(sampleImg, CHECKERBOARD):
    
    # plt.imshow(sampleImg, cmap='gray'  )
    
    ret, corners = cv2.findChessboardCorners(sampleImg, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)


    if( ret ):
          
        xmin = min( corners[:, 0, 0] ) - 20
        xmax = max( corners[:, 0, 0] ) + 20
        ymin = min( corners[:, 0, 1] ) - 20
        ymax = max( corners[:, 0, 1] ) + 20
        
        # plt.scatter(corners[:,0,0], corners[:,0,1])
        # plt.xlim(xmin, xmax)
        # plt.ylim(ymin, ymax)
        
        
        linesVertical   = np.array([])
        linesHorizontal =  np.array([])
        
        for i in range(CHECKERBOARD[1]):
            line = corners[i * CHECKERBOARD[0]: (i + 1) * CHECKERBOARD[0] : 1, 0, :]
            linesVertical = np.append(linesVertical, line)
        
        for i in range(CHECKERBOARD[0]):
            line = corners[i : : CHECKERBOARD[0], 0, :]
            linesHorizontal = np.append(linesHorizontal, line)
            
        linesVertical   = linesVertical.reshape( CHECKERBOARD[1], CHECKERBOARD[0], 2 )
        linesHorizontal = linesHorizontal.reshape( (CHECKERBOARD[0], CHECKERBOARD[1], 2) )
        
        errorVertical = 0
        errorHorizontal = 0
        coefsVertical = np.array([])
        coefsHorizontal = np.array([])
        
        for line in linesVertical:
            coef = np.polyfit( line[:,0], line[:, 1], 1)
            for point in line:
                errorVertical += distanceToLine(coef, point)
            coefsVertical = np.append(coefsVertical, coef)
            
        for line in linesHorizontal:
            coef = np.polyfit( line[:,0], line[:, 1], 1 )
            for point in line:
                errorHorizontal += distanceToLine(coef, point)
            coefsHorizontal = np.append(coefsHorizontal, coef)
        
        coefsVertical = coefsVertical.reshape( (CHECKERBOARD[1], 2) )
        coefsHorizontal = coefsHorizontal.reshape( (CHECKERBOARD[0], 2) )
        
        ###############################################################################################
        #################### TEST ###########################################
        # x = np.linspace(xmin, xmax, 10000)
        # for coefs in coefsVertical:
        #     y = coefs[0] * x + coefs[1]
        #     plt.plot(x,y, linewidth=2.5)
            
        # for coefs in coefsHorizontal:
        #     y = coefs[0] * x + coefs[1]
        #     plt.plot(x,y)
            
        
        return coefsVertical, coefsHorizontal, errorVertical + errorHorizontal
        
    else:
        print("")
        print("CHECKER BOARD DOES NOT FIT INTO IMAGE")
        
        return None, None, None
    
    



def calculateL2NormBetweenCorners(img1, img2, CHECKERBOARD):
    
    ret1, corners1 = cv2.findChessboardCorners(img1, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret2, corners2 = cv2.findChessboardCorners(img2, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

    if(ret1 and ret2):
        error = 0
        for corner1, corner2 in zip(corners1, corners2):
            error += np.linalg.norm(corner1 - corner2)
            # print(corner1, corner2)
            # print("------------------")
        
        return error
    
    else:
        return None


errDiffLeft          = 0
errDiffRight         = 0
errLineLeft_intel    = 0
errLineLeft_calib    = 0 
errLineRight_intel   = 0
errLineRight_calib   = 0

countDiffleft        = 0
countDiffRight       = 0
countLineLeft_intel  = 0
countLineLeft_calib  = 0
countLineRight_intel = 0
countLineRight_calib = 0

for img1, img2 in zip(fisheye1Frames, fisheye2Frames):
    
    img1_intel = cv2.fisheye.undistortImage(img1, K=K1,    D=D1,    Knew=K1,    new_size=(int(fisheyeHeight), int(fisheyeWidth)))
    img1_calib = cv2.fisheye.undistortImage(img1, K=Kleft, D=Dleft, Knew=Kleft, new_size=(int(fisheyeHeight), int(fisheyeWidth)))
    
    img2_intel = cv2.fisheye.undistortImage(img2, K=K2,     D=D2,     Knew=K2,     new_size=(int(fisheyeHeight), int(fisheyeWidth)))
    img2_calib = cv2.fisheye.undistortImage(img2, K=Kright, D=Dright, Knew=Kright, new_size=(int(fisheyeHeight), int(fisheyeWidth)))
    
    error = calculateL2NormBetweenCorners(img1_intel, img1_calib, CHECKERBOARD )
    if (error):
        errDiffLeft  += error
        countDiffleft += 1
        
    error = calculateL2NormBetweenCorners(img2_intel, img2_calib, CHECKERBOARD )
    if (error):
        errDiffRight += error
        countDiffRight += 1
    
    _, _, error = findBestFitLinesAndErrors(img1_intel, CHECKERBOARD)
    if (error):
        errLineLeft_intel += error
        countLineLeft_intel += 1
    
    _, _, error = findBestFitLinesAndErrors(img1_calib, CHECKERBOARD)
    if (error):
        errLineLeft_calib += error
        countLineLeft_calib +=1
        
    _, _, error = findBestFitLinesAndErrors(img2_intel, CHECKERBOARD)
    if (error):
        errLineRight_intel += error
        countLineRight_intel += 1
    
    _, _, error = findBestFitLinesAndErrors(img2_calib, CHECKERBOARD)
    if (error):
        errLineRight_calib += error
        countLineRight_calib += 1
        
errDiffLeftAvg        = errDiffLeft        / ( countDiffleft        *  CHECKERBOARD[0] * CHECKERBOARD[1]  )
errDiffRightAvg       = errDiffRight       / ( countDiffRight       *  CHECKERBOARD[0] * CHECKERBOARD[1]  )
errLineLeft_intelAvg  = errLineLeft_intel  / ( countLineLeft_intel  * (CHECKERBOARD[0] + CHECKERBOARD[1]) )
errLineLeft_calibAvg  = errLineLeft_calib  / ( countLineLeft_calib  * (CHECKERBOARD[0] + CHECKERBOARD[1]) )
errLineRight_intelAvg = errLineRight_intel / ( countLineRight_intel * (CHECKERBOARD[0] + CHECKERBOARD[1]) )
errLineRight_calibAvg = errLineRight_calib / ( countLineRight_calib * (CHECKERBOARD[0] + CHECKERBOARD[1]) )



# ##################################################################################################
# ## following code is to determine the order of the returned points from chess board corners
# ##################################################################################################
# for i in range( corners.shape[0] + 1 ):
#     plt.figure(i, figsize=(7, 7) )
#     for n in range(i):
#         plt.scatter(corners[n, 0, 0], corners[n, 0, 1])
#         plt.xlim(xmin, xmax)
#         plt.ylim(ymin, ymax)
# ##################################################################################################





























