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
 


sampleImg = cv2.imread("/home/salih/Documents/fishEyeSalvation/fishEyeLeftSalvation/40_Fisheye.png", cv2.IMREAD_UNCHANGED)
sampleImg = cv2.fisheye.undistortImage(sampleImg, K=K1, D=D1, Knew=K1, new_size=(int(fisheyeHeight), int(fisheyeWidth)))


def findBestFitLinesAndErrors(sampleImg, CHECKERBOARD):
    
    plt.imshow(sampleImg, cmap='gray'  )
    
    ret, corners = cv2.findChessboardCorners(sampleImg, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

    xmin = min( corners[:, 0, 0] ) - 20
    xmax = max( corners[:, 0, 0] ) + 20
    ymin = min( corners[:, 0, 1] ) - 20
    ymax = max( corners[:, 0, 1] ) + 20
    
    plt.scatter(corners[:,0,0], corners[:,0,1])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
    if( corners.shape[0] == CHECKERBOARD[0] * CHECKERBOARD[1] ):
        
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
        
        residualVertical = 0
        residualHorizontal = 0
        coefsVertical = np.array([])
        coefsHorizontal = np.array([])
        
        for line in linesVertical:
            coef, residual, _, _, _ = np.polyfit( line[:,0], line[:, 1], 1, full=True )
            residualVertical += residual
            coefsVertical = np.append(coefsVertical, coef)
            
        for line in linesHorizontal:
            coef, residual, _, _, _ = np.polyfit( line[:,0], line[:, 1], 1, full=True )
            residualHorizontal += residual
            coefsHorizontal = np.append(coefsHorizontal, coef)
        
        coefsVertical = coefsVertical.reshape( (CHECKERBOARD[1], 2) )
        coefsHorizontal = coefsHorizontal.reshape( (CHECKERBOARD[0], 2) )
        
        ###############################################################################################
        #################### TEST ###########################################
        x = np.linspace(xmin, xmax, 10000)
        for coefs in coefsVertical:
            y = coefs[0] * x + coefs[1]
            plt.plot(x,y, linewidth=2.5)
            
        for coefs in coefsHorizontal:
            y = coefs[0] * x + coefs[1]
            plt.plot(x,y)
        
        
    else:
        print("")
        print("CHECKER BOARD DOES NOT FIT INTO IMAGE")
    
    

findBestFitLinesAndErrors(sampleImg, CHECKERBOARD)

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


































