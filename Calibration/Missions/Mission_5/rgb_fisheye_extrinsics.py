import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

if(os.getcwd().split('/')[-1] != 'star'):
    os.chdir("../../../")
    
test = False
record = True

# read images
leftFishImgs = []
rightFishImgs = []
rgbImgs = []

folderNames = glob.glob("SnapShots/t265_d435i/*") 
folderNames.sort()

for folderName in folderNames :

    # read file names
    fileNamesL = glob.glob(folderName + "/leftFisheye/*" )
    fileNamesR = glob.glob(folderName + "/rightFisheye/*")
    fileNamesRgb = glob.glob(folderName + "/rgb/*")
    
    fileNamesL.sort()
    fileNamesR.sort()
    fileNamesRgb.sort()
    
    
    for iml, imr, imrgb in zip(fileNamesL, fileNamesR, fileNamesRgb):
        iml = cv2.imread(iml, cv2.IMREAD_UNCHANGED)
        imr = cv2.imread(imr, cv2.IMREAD_UNCHANGED)
        imrgb = cv2.imread(imrgb, cv2.IMREAD_GRAYSCALE)
        
        leftFishImgs.append(iml)
        rightFishImgs.append(imr)
        rgbImgs.append(imrgb)
    
# undistort fisheye images
K1 = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
D1 = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])

K2 = np.array([[284.1828918457031, 0.0, 427.9779052734375], [0.0, 285.0440979003906, 399.5506896972656], [0.0, 0.0, 1.0]])
D2 = np.array([0.0009760634857229888, 0.030147459357976913, -0.02769969031214714, 0.0031066760420799255])

fisheyeWidth = 800
fisheyeHeight = 848

KRGB = np.array([[923.2835693359375, 0.0, 639.9102783203125], [0.0, 923.6146240234375, 370.2297668457031], [0.0, 0.0, 1.0]])
D = np.array([0, 0, 0, 0, 0], dtype=np.float32)

boardWidth = 8
boardHeight = 8
CHECKERBOARD = (boardHeight, boardWidth)
squareSize = 0.033 # 3.3 cm

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

    return undistortedImages, np.array(P[:,:-1])


leftUndistorted, KL = undistortFisheyeImages(leftFishImgs, K1, D1)
rightUndistorted, KR = undistortFisheyeImages(rightFishImgs, K2, D2)

def matchCornerOrder_v2(corners1, corners2):
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
            print("Sıçtık Again!!")
            break
        
    corners1 = corners1.reshape(64, 1, 2)
    corners2 = corners2.reshape(64, 1, 2)
    return corners1, corners2

def matchCornerOrder(corners1, corners2, img1, img2, checker=(8,8)):
    corners1 = corners1.reshape(8, 8, 1, 2)
    corners2 = corners2.reshape(8, 8, 1, 2)
    mid1 = corners1[4, 4, 0, :]
    mid2 = corners2[4, 4, 0, :]
    # açıyla yönelim hesapla, eşleştir!!
    flags = False
    cntr = 0
    while(True):
        # point 1 camera 1
        leftFlagP1_1 = corners1[0, 0,0,0] < mid1[0] #if smaller first point on the left
        upFlagP1_1   = corners1[0, 0,0,1] < mid1[1] #if smaller first point on the up
        
        # point 2 camera 1
        leftFlagP2_1 = corners1[7, 0,0,0] < mid1[0] #if smaller second point on the left
        upFlagP2_1   = corners1[7, 0,0,1] < mid1[1] #if smaller second point on the up
        
        # point 1 camera 2
        leftFlagP1_2 = corners2[0, 0,0,0] < mid2[0] #if smaller first point on the left
        upFlagP1_2   = corners2[0, 0,0,1] < mid2[1] #if smaller first point on the up
        
        # point 2 camera 2
        leftFlagP2_2 = corners2[7, 0,0,0] < mid2[0] #if smaller second point on the left
        upFlagP2_2   = corners2[7, 0,0,1] < mid2[1] #if smaller second point on the up
        
        flags = leftFlagP1_1 == leftFlagP1_2 and upFlagP1_1 == upFlagP1_2 and leftFlagP2_1 == leftFlagP2_2 and upFlagP2_1 == upFlagP2_2
        if(flags):
            break
        corners2 = np.rot90(corners2, 1, (0, 1))
        cntr += 1
        
        if(cntr == 4):
            corners2 = np.flip(corners2, 0)
        
        if (cntr > 2) :
            img1_ = np.copy(img1)
            img2_ = np.copy(img2)
            for i, (corner1, corner2) in enumerate(zip(corners1.reshape(64, 1, 2), corners2.reshape(64, 1, 2))):
                cv2.circle(img1_, (int(corner1[0][0]),int(corner1[0][1])), i,  (255,255,255), thickness=1)
                cv2.circle(img2_, (int(corner2[0][0]),int(corner2[0][1])), i,  (255,255,255), thickness=1)
                
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2] 
            img = np.zeros((max(h1, h2), w1+w2), np.uint8)
            #combine 2 images
            img[:h1, :w1] = img1_
            img[:h2, w1:w1+w2] = img2_

            cv2.imwrite("./temp/"+"Georgia"+ str(cntr) +".png", img)    
                
        if(cntr == 8):
            print("Sıçtık!!")
            break
        
    corners1 = corners1.reshape(64, 1, 2)
    corners2 = corners2.reshape(64, 1, 2)
    return corners1, corners2
        

imagePointsLeft =  []
imagePointsRight =  []
imagePointsRgb =  [] 
imgsRgbWP = []
imgsLWP = []
imgsRWP = []

counter = 0
imageCount = len(leftUndistorted)
for i in range(imageCount):

    imageLeft = leftUndistorted[i]
    imageRgb = rgbImgs[i]
    imageRight = rightUndistorted[i]

    retL, cornersL = cv2.findChessboardCornersSB(imageLeft, CHECKERBOARD, cv2.CALIB_CB_ACCURACY+cv2.CALIB_CB_NORMALIZE_IMAGE)
    retR, cornersR = cv2.findChessboardCornersSB(imageRight, CHECKERBOARD, cv2.CALIB_CB_ACCURACY+cv2.CALIB_CB_NORMALIZE_IMAGE)
    retRgb, cornersRgb = cv2.findChessboardCornersSB(imageRgb, CHECKERBOARD, cv2.CALIB_CB_ACCURACY+cv2.CALIB_CB_NORMALIZE_IMAGE)

    if(retL and retRgb and retR):
        cv2.cornerSubPix(imageLeft, cornersL, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.cornerSubPix(imageRight, cornersR, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.cornerSubPix(imageRgb, cornersRgb, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cornersL, cornersR = matchCornerOrder_v2(cornersL, cornersR)
        cornersL, cornersRgb = matchCornerOrder_v2(cornersL, cornersRgb)
        imagePointsLeft.append(cornersL)
        imagePointsRight.append(cornersR)
        imagePointsRgb.append(cornersRgb)
        imgsLWP.append(cv2.cvtColor(imageLeft, cv2.COLOR_GRAY2RGB))
        imgsRWP.append(cv2.cvtColor(imageRight, cv2.COLOR_GRAY2RGB))
        imgsRgbWP.append(cv2.cvtColor(imageRgb, cv2.COLOR_GRAY2RGB))


sep = len(imagePointsLeft) // 3
imagePointsLeft_train = imagePointsLeft[sep:]
imagePointsLeft_validate = imagePointsLeft[:sep]
imagePointsRight_train = imagePointsRight[sep:]
imagePointsRight_validate = imagePointsRight[:sep]
imagePointsRgb_train = imagePointsRgb[sep:]
imagePointsRgb_validate = imagePointsRgb[:sep]

imgsLWP_train = imgsLWP[sep:]
imgsLWP_validate = imgsLWP[:sep]
imgsRWP_train = imgsRWP[sep:]
imgsRWP_validate = imgsRWP[:sep]
imgsRgbWP_train = imgsRgbWP[sep:]
imgsRgbWP_validate = imgsRgbWP[:sep]

N_OK = len(imagePointsLeft_train)
objp = np.zeros((8*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:8].T.reshape(-1, 2)
objp = objp * squareSize
objp = np.array([objp]*len(imagePointsLeft_train), dtype=np.float32)
objp = np.reshape(objp, (N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 3))

# retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objp, imagePointsRgb, (720, 1280), KRGB, D )
retval1, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R1, T1, E1, F1 = cv2.stereoCalibrate(objp, imagePointsLeft_train, imagePointsRgb_train, KL, D, KRGB, D, (0,0), flags=cv2.CALIB_FIX_INTRINSIC)
retval2, cameraMatrix3, distCoeffs3, cameraMatrix4, distCoeffs4, R2, T2, E2, F2 = cv2.stereoCalibrate(objp, imagePointsRight_train, imagePointsRgb_train, KR, D, KRGB, D, (0,0), flags=cv2.CALIB_FIX_INTRINSIC)
retval3, cameraMatrix5, distCoeffs5, cameraMatrix6, distCoeffs6, R3, T3, E3, F3 = cv2.stereoCalibrate(objp, imagePointsLeft_train, imagePointsRight_train, KL, D, KR, D, (0,0), flags=cv2.CALIB_FIX_INTRINSIC)


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,_ = img1.shape
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    i = 1
    np.random.seed(0)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        pt1 = tuple(map(tuple, pt1))[0]
        pt2 = tuple(map(tuple, pt2))[0]
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,pt1,5,color,-1)
        # img2 = cv2.circle(img2,pt2,5,color,-1)
        # img1 = cv2.circle(img1,pt1,i,(255,255,255), thickness=1)
        # img2 = cv2.circle(img2,pt2,i,(255,255,255), thickness=1)
        # i += 1
    return img1,img2

# VALIDATINON SET ERROR!! DO CALCULATE
if(test):
    errs = []
    for i in range(sep):
        lines1 = cv2.computeCorrespondEpilines(imagePointsRgb_validate[i].reshape(-1,1,2), 2, F1)
        # lines1 = cv2.computeCorrespondEpilines(imagePointsRight[i].reshape(-1,1,2), 2, F)
        lines1 = lines1.reshape(-1,3)
        # img5,img6 = drawlines(imgsLWP[i], imgsRWP[i],lines1, imagePointsLeft[i], imagePointsRight[i])
        img5,img6 = drawlines(imgsLWP_validate[i], imgsRgbWP_validate[i],lines1, imagePointsLeft_validate[i], imagePointsRgb_validate[i])
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(imagePointsLeft_validate[i].reshape(-1,1,2), 1,F1)
        lines2 = lines2.reshape(-1,3)
        # img3,img4 = drawlines(imgsRWP[i], imgsLWP[i],lines2, imagePointsRight[i], imagePointsLeft[i])
        img3,img4 = drawlines(imgsRgbWP_validate[i], imgsLWP_validate[i],lines2, imagePointsRgb_validate[i], imagePointsLeft_validate[i])
        # plt.subplot(121),plt.imshow(img5)
        # plt.subplot(122),plt.imshow(img3)
        # plt.show()
        
        # img3 = cv2.resize(img3, img5.shape)
        # img = cv2.hconcat([img5, img3])
        h1, w1 = img3.shape[:2]
        h2, w2 = img5.shape[:2]
    
        #create empty matrix
        img = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
    
        #combine 2 images
        img[:h1, :w1,:3] = img3
        img[:h2, w1:w1+w2,:3] = img5
        cv2.imwrite("./temp/"+"George_Bush"+ str(i) +".png", img)
        
        
        ####################################
        ## Error calculation
        ######################
        
        err = 0
        
        for pt, l in zip(imagePointsLeft[i], lines1):
            err += abs(pt[0,0] * l[0] + pt[0,1] * l[1] + l[2])
        
        # for pt, l in zip(imagePointsRight[i], lines2):
        #     err += abs(pt[0,0] * l[0] + pt[0,1] * l[1] + l[2])
            
        for pt, l in zip(imagePointsRgb[i], lines2):
            err += abs(pt[0,0] * l[0] + pt[0,1] * l[1] + l[2])
            
            
        err /= lines1.shape[0] + lines2.shape[0]
        errs.append(err)
    
    
    
            
                
        
            
        
if(record):
    # 1 rgb -> left
    # 2 rgb -> right
    # 3 right -> left
    file = open("Calibration/Missions/Mission_5/extrinsics.txt", "w")
    file.write("\n*Translation from RGB to Left Fisheye: \n")
    file.write(str(T1))
    file.write("\n*Rotation from RGB to Left Fisheye: \n")
    file.write(str(R1))
    
    file.write("\n*Translation from RGB to Right Fisheye: \n")
    file.write(str(T2))
    file.write("\n*Rotation from RGB to Right Fisheye: \n")
    file.write(str(R2))
    
    file.write("\n*Translation from Right to Left Fisheye: \n")
    file.write(str(T3))
    file.write("\n*Rotation from Right to Left Fisheye: \n")
    file.write(str(R3))
    file.close()



import re
from ast import literal_eval

import numpy as np


a = "[[ 0.99998026  0.00576925  0.00249038] \
[-0.00577328  0.99998203  0.00161431] \
[-0.00248102 -0.00162865  0.9999956 ]]"


# a = """[[[ 0 1][ 2 3]]]"""
a = re.sub(r"([^[])\s+([^]])", r"\1, \2", a)
a = np.array(literal_eval(a))




