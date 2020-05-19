import sys
import argparse
import os
import numpy as np
import cv2

os.chdir("..")
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import fetcher

fetcher.insertFocusDataPath("hessBoardFisheye")
fetcher.fetchFisheyeDataT265()

targetFrameIndex = 100

fisheyeWidth = 800
fisheyeHeight = 848

np.random.seed(4)
targetFrameIndexes = np.random.choice(np.arange(1,500),1)

#targetFrameIndexes = [250]


fisheye1Frames = []
fisheye2Frames = []

for targetIndex in targetFrameIndexes:
    
    data1 = np.array((fetcher.fisheye1_t265[targetIndex][1:-1]).split(", "),dtype=np.uint8)
    data1 = data1.reshape((fisheyeWidth,fisheyeHeight))
    
    data2 = np.array((fetcher.fisheye2_t265[targetIndex][1:-1]).split(", "),dtype=np.uint8)
    data2 = data2.reshape((fisheyeWidth,fisheyeHeight))    
    
    fisheye1Frames.append(data1)
    fisheye2Frames.append(data2)


K1 = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
D1 = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])

KguessFisheye1 = np.copy(K1)
DguessFisheye1 = np.copy(D1)

KguessFisheye2 = np.copy(K1)
DguessFisheye2 = np.copy(D1)


def undistortFisheyeImages(fisheyeImages, K, D):
    
    undistortedImages = []
    
    nk = K.copy()
    nk[0,0]=K[0,0]/2
    nk[1,1]=K[1,1]/2
#    nk[0,0]=K[0,0]
#    nk[1,1]=K[1,1]     
    
    for image in fisheyeImages:
        
        print(image.shape)
    
        undistorted = cv2.fisheye.undistortImage(image, K=K, D=D, Knew=nk, new_size=(int(fisheyeHeight), int(fisheyeWidth)))
        undistortedImages.append(undistorted)

    return undistortedImages


def showImages(images):
    
    for i,image in enumerate(images):
        
        cv2.imshow("image index: " + str(i), image)





def drawLineOnImages(images):
    
    newImages = []
    
    for image in images:
                
        edges = cv2.Canny(image,50,150,apertureSize = 3)
        
        lines = cv2.HoughLines(edges,1,np.pi/180,200)
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
            
                cv2.line(image,(x1,y1),(x2,y2),(255,255,255),2)
        
        newImages.append(image)
    
    return newImages

def drawChessboardCornersForImages(images_):
    
    images = np.copy(images_)
    
    newImages = []
    
    CHECKERBOARD = (8,6) 
    
    for image in images:
        
        ret, corners = cv2.findChessboardCorners(image, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
       
        if(ret == True):
            
            for corner in corners:
                
                cv2.circle(image, (int(corner[0][0]),int(corner[0][1])), 4,  (255,255,255), thickness=-1,)        
        
        newImages.append(image)
                    
    return newImages
        


def findFisheyeCalibrationsFromFrames(frames, Kinitial = None):
    
    if(Kinitial is None):
        Kinitial = np.eye(3)
    
        # Checkboard dimensions
    CHECKERBOARD = (8,6)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC  + cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    for frame in frames:
        
    
        ret, corners = cv2.findChessboardCorners(frame, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(frame,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
    ###
    
    # calculate K & D
    N_imm = len(objpoints)
    K = np.copy(Kinitial)
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        frame.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-7))


    return (K,D)


def undistortImageCorners(K, D, corners):
    
    nk = K.copy()
    nk[0,0]=K[0,0]/2
    nk[1,1]=K[1,1]/2

    
    undistortedVectors = cv2.fisheye.undistortPoints(corners, K, D, P=nk)
    
    return undistortedVectors


def distortImageCorners(K, D, corners):
    
    distortedVectors = cv2.fisheye.distortPoints(corners, K, D)
    
    return distortedVectors


def findDistBetweenPoints(point1, point2):
    
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

        
#main

showImages(fisheye1Frames + fisheye2Frames)
showImages(fisheye2Frames)

###############

corneredImages1 = drawChessboardCornersForImages(fisheye1Frames)
corneredImages2 = drawChessboardCornersForImages(fisheye2Frames)

showImages(corneredImages1)
showImages(corneredImages2)

##############

undistortedImages1 = undistortFisheyeImages(fisheye1Frames, K1, D1)
undistortedImages2 = undistortFisheyeImages(fisheye2Frames, K1, D1)
        
showImages(undistortedImages1)
showImages(undistortedImages2)

showImages([undistortedImages1[0]] + [fisheye1Frames[0]])

##############


KguessFisheye1, DguessFisheye1 = findFisheyeCalibrationsFromFrames(fisheye1Frames)
KguessFisheye2, DguessFisheye2 = findFisheyeCalibrationsFromFrames(fisheye2Frames)

undistortedImages1Guess = undistortFisheyeImages(fisheye1Frames, KguessFisheye1, DguessFisheye1)
undistortedImages2Guess = undistortFisheyeImages(fisheye2Frames, KguessFisheye2, DguessFisheye2)

showImages([undistortedImages1Guess[0]])
showImages([undistortedImages2Guess[0]])

showImages([undistortedImages1[0]] + [undistortedImages1Guess[0]] + [undistortedImages2Guess[0]])


##############

showImages(undistortedImages1 + undistortedImages1Guess)
showImages(undistortedImages2 + undistortedImages2Guess)

##############

outputVectorsIntel = undistortImageCorners(K1, D1, fisheye2Frames[0])

outputVectorsGuess1 = undistortImageCorners(KguessFisheye1, DguessFisheye1, fisheye1Frames[0])
outputVectorsGuess2 = undistortImageCorners(KguessFisheye2, DguessFisheye2, fisheye2Frames[0])

outputVectorGuessMean = undistortImageCorners((KguessFisheye1 + KguessFisheye2)/2, (DguessFisheye1+DguessFisheye2)/2, fisheye1Frames[0])

##############

# try to validate undistorImageCorners

randomPixelsFisheye = np.array([[307,277], [670, 254], [403, 647]]).astype(np.float32)
randomPixelsUndistorted = np.array([[360,328], [608, 290], [412, 567]]).astype(np.float32)


distUndistortFirst = findDistBetweenPoints(randomPixelsUndistorted[0], randomPixelsUndistorted[1])
distUndistortSecond = findDistBetweenPoints(randomPixelsUndistorted[1], randomPixelsUndistorted[2])

ratioUndistorted = distUndistortFirst / distUndistortSecond


undistortedCorners = undistortImageCorners(K1, D1, randomPixelsFisheye)

distCornerFirst = findDistBetweenPoints(undistortedCorners[0][0], undistortedCorners[1][0])
distCornerSecond = findDistBetweenPoints(undistortedCorners[1][0], undistortedCorners[2][0])

ratioCorners = distCornerFirst / distCornerSecond


###############
w = fisheye1Frames[0].shape[1]
h = fisheye1Frames[0].shape[0]

corners = np.array([[0,0], [w,0], [w,h], [0,h], [138, 131]]).astype(np.float32)
corners = np.float32(corners[:, np.newaxis, :])
sides = np.array([[0,int(h/2)], [int(w/2), 0], [w,int(h/2)], [int(w/2), h]]).astype(np.float32)

pixelCorners_intel = undistortImageCorners(K1, D1, corners)
pixelCorners_fisheye1 = undistortImageCorners(KguessFisheye1, DguessFisheye1, corners)
pixelSides_intel = undistortImageCorners(K1, D1, sides)
pixelSides_fisheye1 = undistortImageCorners(KguessFisheye1, DguessFisheye1, sides)

#############


circularMap = [(138,131), (119,146), (258,36), (341,8), (519,5), (670, 73), (777, 195), (825, 327), (821, 493), (725, 667), (597, 757), (365, 787), (173, 700), (70,559), (30,375)]
circularMap = np.array(circularMap).astype(np.float32)
circularMap = np.float32(circularMap[:, np.newaxis, :])

rectangularMap = np.linspace((0,0), (w,0), 100)
rectangularMap = np.concatenate( (rectangularMap, np.linspace((w,0), (w,h), 100)) )
rectangularMap = np.concatenate( (rectangularMap, np.linspace((w,h), (0,h), 100)) )
rectangularMap = np.concatenate( (rectangularMap, np.linspace((0,h), (0,0), 100)) )
rectangularMap = np.array([rectangularMap]).astype(np.float32)
#rectangularMap = np.float32(rectangularMap[:, np.newaxis, :])

mappedRectangular = mappedRectangular[0]

mappedRectangular = undistortImageCorners(K1, D1, rectangularMap)

mappedCircular = undistortImageCorners(K1, D1, circularMap)


plt.scatter(mappedRectangular[:,0], mappedRectangular[:,1])
plt.scatter(mappedCircular[:,0,0], mappedCircular[:,0,1])

plt.scatter(rectangularMap[:,0], rectangularMap[:,1])
plt.scatter(circularMap[:,0], circularMap[:,1])




plt.figure()
