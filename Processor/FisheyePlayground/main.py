import sys
import argparse
import os
import numpy as np
import cv2

#os.chdir("..")
#sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import fetcher


readFromCsv = False

fisheyeWidth = 800
fisheyeHeight = 848

fisheye1Frames = []
fisheye2Frames = []


if(readFromCsv):

    fetcher.insertFocusDataPath("hessBoardFisheye")
    fetcher.fetchFisheyeDataT265()
    
    targetFrameIndex = 100
    

    
    np.random.seed(45)
    targetFrameIndexes = np.random.choice(np.arange(1,500),1)
    
    
    #targetFrameIndexes = [250]
    
    
    for targetIndex in targetFrameIndexes:
        
        data1 = np.array((fetcher.fisheye1_t265[targetIndex][1:-1]).split(", "),dtype=np.uint8)
        data1 = data1.reshape((fisheyeWidth,fisheyeHeight))
        
        data2 = np.array((fetcher.fisheye2_t265[targetIndex][1:-1]).split(", "),dtype=np.uint8)
        data2 = data2.reshape((fisheyeWidth,fisheyeHeight))    
        
        fisheye1Frames.append(data1)
        fisheye2Frames.append(data2)
        
        
else:

          
    
    
    for filename in os.listdir("../../fisheyeSnapshots/leftEye"):
        img = cv2.imread(os.path.join("../../fisheyeSnapshots/leftEye",filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            fisheye1Frames.append(img)
    
    
    for filename in os.listdir("../../fisheyeSnapshots/rightEye"):
        img = cv2.imread(os.path.join("../../fisheyeSnapshots/rightEye",filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            fisheye2Frames.append(img)
    
    
    
    
    


K1 = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
D1 = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])

K2 = np.array([[284.1828918457031, 0.0, 427.9779052734375], [0.0, 285.0440979003906, 399.5506896972656], [0.0, 0.0, 1.0]])
D2 = np.array([0.0009760634857229888, 0.030147459357976913, -0.02769969031214714, 0.0031066760420799255])

KguessFisheye1 = np.copy(K1)
DguessFisheye1 = np.copy(D1)

KguessFisheye2 = np.copy(K2)
DguessFisheye2 = np.copy(D2)


def undistortFisheyeImages(fisheyeImages, K, D):
    
    undistortedImages = []
    
    nk = K.copy()

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
    
    CHECKERBOARD = (6,9) 
    
    for image in images:
        
        ret, corners = cv2.findChessboardCorners(image, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
       
        if(ret == True):
                        
            for corner in corners:
                
                cv2.circle(image, (int(corner[0][0]),int(corner[0][1])), 2,  (255,255,255), thickness=-1,)        
        
            newImages.append(image)
                    
    return newImages


        


def findFisheyeCalibrationsFromFrames(frames, Kinitial = None):
    
    if(Kinitial is None):
        Kinitial = np.eye(3)
    
        # Checkboard dimensions
    CHECKERBOARD = (6,9)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC  + cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    for frame in frames:
        
    
        ret, corners = cv2.findChessboardCorners(image, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        # If found, add object points, image points (after refining them)
        if(ret == True):
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

    
    undistortedVectors = cv2.undistortPoints(corners, K, D, P=nk)
    
    return undistortedVectors


def distortImageCorners(K, D, corners):
    
    distortedVectors = cv2.fisheye.distortPoints(corners, K, D)
    
    return distortedVectors


def findDistBetweenPoints(point1, point2):
    
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def drawPointsOnImage(image_, points):
    
    image = np.copy(image_)
    
    for point in points:
        
        cv2.circle(image, (int(point[0][0]),int(point[0][1])), 4,  (255,255,255), thickness=-1,)        
                 
    return image

        
#main
    
def garbage_main():

    showImages(fisheye1Frames[:2] + fisheye2Frames[:2])
    
    showImages(fisheye1Frames)
    showImages(fisheye2Frames)
    
    ###############
    
    corneredImages1 = drawChessboardCornersForImages(fisheye1Frames)
    corneredImages2 = drawChessboardCornersForImages(fisheye2Frames)
    
    showImages(corneredImages1)
    showImages(corneredImages2)
    
    ##############
    
    undistortedImages1 = undistortFisheyeImages(fisheye1Frames, K1, D1)
    undistortedImages2 = undistortFisheyeImages(fisheye2Frames, K2, D2)
            
    showImages(undistortedImages1)
    showImages(undistortedImages2)
    
    showImages([undistortedImages1[0]] + [undistortedImages2[0]])
    
    showImages([undistortedImages1[0]] + [fisheye1Frames[0]])
    
    ##############
    
    
    KguessFisheye1, DguessFisheye1 = findFisheyeCalibrationsFromFrames(fisheye1Frames)
    KguessFisheye2, DguessFisheye2 = findFisheyeCalibrationsFromFrames(fisheye2Frames)
    
    undistortedImages1Guess = undistortFisheyeImages(fisheye1Frames[:2], KguessFisheye1, DguessFisheye1)
    undistortedImages2Guess = undistortFisheyeImages(fisheye2Frames[:2], KguessFisheye2, DguessFisheye2)
    
    showImages([undistortedImages1Guess[0]])
    showImages([undistortedImages2Guess[1]])
    
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
    
    
    
    mappedRectangular_intel = undistortImageCorners(K1, D1, rectangularMap)
    mappedRectangular_intel = mappedRectangular_intel[:,0,:]
    
    mappedRectangular_guess = undistortImageCorners(KguessFisheye2, DguessFisheye2, rectangularMap)
    mappedRectangular_guess = mappedRectangular_guess[:,0,:]
    
    
    
    mappedCircular_intel = undistortImageCorners(K1, D1, circularMap)
    mappedCircular_intel = mappedCircular_intel[:,0,:]
    
    mappedCircular_guess = undistortImageCorners(KguessFisheye2, DguessFisheye2, circularMap)
    mappedCircular_guess = mappedCircular_guess[:,0,:]
    
    plt.figure()
    plt.scatter(mappedRectangular_intel[:,0], mappedRectangular_intel[:,1])
    plt.scatter(mappedRectangular_guess[:,0], mappedRectangular_guess[:,1])
    
    plt.figure()
    plt.scatter(mappedCircular_intel[:,0], mappedCircular_intel[:,1])
    plt.scatter(mappedCircular_guess[:,0], mappedCircular_guess[:,1])
    
    
    plt.scatter(rectangularMap[0,:,0], rectangularMap[0,:,1])
    plt.scatter(circularMap[:,0,0], circularMap[:,0,1])
    
    ############ After a long time passed through final hell
    # Exploration Area
    
    # Objectives : 1 - For the first step we will use Intel intrinsic parameters to achive an inverse function completion
    # between cv2.fisheye.distort and cv2.fisheye.undistort points
    
    # intel k : K1
    # intel d : D1
    
    # Initiate
    
    # we will use rectangular test points
    w = 50
    h = 50
    originX = -w/2
    originY = -h/2
    
    rectangularMap = np.linspace((originX,originY), (originX + w, originY), 100)
    rectangularMap = np.concatenate( (rectangularMap, np.linspace((originX + w, originY), (originX+ w, originY + h), 100)) )
    rectangularMap = np.concatenate( (rectangularMap, np.linspace((originX + w, originY + h), (originX, originY + h), 100)) )
    rectangularMap = np.concatenate( (rectangularMap, np.linspace((originX, originY + h), (originX, originY), 100)) )
    rectangularMap = np.array([rectangularMap]).astype(np.float32)
    
    plt.scatter(rectangularMap[0,:,0], rectangularMap[0,:,1])
    
    #distorting points
    distorted	 = cv2.fisheye.distortPoints(rectangularMap, K1, D1)
    
    plt.scatter(distorted[0,:,0], distorted[0,:,1])
    
    #undistorting points
    
    undistorted = cv2.fisheye.undistortPoints(distorted, K1, D1)
    
    plt.scatter(2*undistorted[0,:,0], 2*undistorted[0,:,1])
    
    # This objective is completed
    
    # Objectives : 2 - We will now compare our calcualated distortion parameters and Intel's given parameters
    
    # we will use rectangular and circular points as test points
    
    #rectangular test points 
    
    w = 800 * 3/4
    h = 848 * 3/4
    cx = 430.92941284
    cy = 394.6651001
    originX = cx - w/2
    originY = cy - h/2
    
    rectangularMap2 = np.linspace((originX,originY), (originX + w, originY), 100)
    rectangularMap2 = np.concatenate( (rectangularMap2, np.linspace((originX + w, originY), (originX+ w, originY + h), 100)) )
    rectangularMap2 = np.concatenate( (rectangularMap2, np.linspace((originX + w, originY + h), (originX, originY + h), 100)) )
    rectangularMap2 = np.concatenate( (rectangularMap2, np.linspace((originX, originY + h), (originX, originY), 100)) )
    rectangularMap2 = np.array([rectangularMap2]).astype(np.float32)
    
    plt.scatter(rectangularMap2[0,:,0], rectangularMap2[0,:,1])
    
    #undistorting points
    
    undistortedRectIntel = cv2.fisheye.undistortPoints(rectangularMap2, K1, D1)
    undistortedRectGuess1 = cv2.fisheye.undistortPoints(rectangularMap2, KguessFisheye1, DguessFisheye1)
    
    plt.scatter(undistortedRectIntel[0,:,0], undistortedRectIntel[0,:,1])
    plt.scatter(undistortedRectGuess1[0,:,0], undistortedRectGuess1[0,:,1])
    
    #circular test points
    
    circularMap = [(138,131), (119,146), (258,36), (341,8), (519,5), (670, 73), (777, 195), (825, 327), (821, 493), (725, 667), (597, 757), (365, 787), (173, 700), (70,559), (30,375)]
    circularMap = np.array(circularMap).astype(np.float32)
    scale = 1
    circularMap [:,0] = (circularMap[:,0] - cx) * scale + cx
    circularMap [:,1] = (circularMap[:,1] - cy) * scale + cy
    
    
    circularMap = np.float32(circularMap[:, np.newaxis, :]) 
    
    plt.scatter(circularMap[:,0,0], circularMap[:,0,1])
    
    undistortedCircIntel = cv2.fisheye.undistortPoints(circularMap, K1, D1)
    plt.scatter(undistortedCircIntel[:,0,0], undistortedCircIntel[:,0,1])
    
    undistortedCircIntel_ = undistortedCircIntel[:,0,:]
    undistortedCircIntel_[:,0] = undistortedCircIntel_[:,0] * K1[0,0] + K1[0,2]
    undistortedCircIntel_[:,1] = undistortedCircIntel_[:,1] * K1[1,1] + K1[1,2]
    
    
    undistortedCircGuess1 = cv2.fisheye.undistortPoints(circularMap, KguessFisheye1, DguessFisheye1)
    plt.scatter(undistortedCircGuess1[:,0,0], undistortedCircGuess1[:,0,1])
    
    undistortedCircGuess1_ = undistortedCircGuess1[:,0,:]
    undistortedCircGuess1_[:,0] = undistortedCircGuess1_[:,0] * KguessFisheye1[0,0] + KguessFisheye1[0,2]
    undistortedCircGuess1_[:,1] = undistortedCircGuess1_[:,1] * KguessFisheye1[1,1] + KguessFisheye1[1,2]
    
    
    plt.scatter(undistortedCircIntel_[:,0], undistortedCircIntel_[:,1])
    plt.scatter(undistortedCircGuess1_[:,0], undistortedCircGuess1_[:,1])
    
    #plot undistorted points on fisheye image
    cv2.imshow("imageWithPoints", drawPointsOnImage(fisheye1Frames[0], circularMap))
    
    #calculate l2 distance between undistortedIntel and undistortedGuess1
    squareSum = np.sqrt(np.sum((undistortedCircIntel_ - undistortedCircGuess1_)**2, axis=1))
    maxDistance = np.max(squareSum)
    
    
    
    
    ##########################################################################
    
    #checkpoint 
    
    ##########################################################################
    
    #we will now try to match between two points of fisheye1 and fisheye2
    
    x1 = np.array([[[511,204]]]).astype(np.float32)
    x2 = np.array([[[237, 209]]]).astype(np.float32)
    
    #  run the Transformations_fisheye.py code here to obtain transformation matrixes
    x2_predict = np.dot(K2 ,np.dot(R2_inv, np.dot(R1, np.dot(np.linalg.inv(K1), x1) ) + T1 - T2))
    
    
    undistortedX1 = cv2.fisheye.undistortPoints(x1, K1, D1)
    undistortedX2 = cv2.fisheye.undistortPoints(x2, K2, D2)
    
    undistortedX1_ = np.append(undistortedX1[0][0], np.array([1]))
    undistortedX2_ = np.append(undistortedX2[0][0], np.array([1]))
    
    undistortedX2_predict =  np.dot(R2_inv, np.dot(R1,  undistortedX1_ ) + T1 - T2)
    print(undistortedX2_predict)
    print(undistortedX2_)
    
    ########################################################################
    
    image_deneme_corners = drawChessboardCornersForImages([image_deneme])
    cv2.imshow("deneme",image_deneme_corners[0])
    
    
    
    


##############################################################################
                            #Active Work Area#
    
    
















