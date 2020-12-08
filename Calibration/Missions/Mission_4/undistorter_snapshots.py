# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob
import os


if(os.getcwd().split('/')[-1] != 'star'):
    os.chdir("../../../")
    
    
K1 = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
D1 = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])

K2 = np.array([[284.1828918457031, 0.0, 427.9779052734375], [0.0, 285.0440979003906, 399.5506896972656], [0.0, 0.0, 1.0]])
D2 = np.array([0.0009760634857229888, 0.030147459357976913, -0.02769969031214714, 0.0031066760420799255])

fisheyeWidth = 800
fisheyeHeight = 848    


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




folderNames = glob.glob("SnapShots/t265_d435i/*") 
folderNames.sort()


for folderName in folderNames :

    # read file names
    fileNamesL = glob.glob(folderName + "/leftFisheye/*" )
    fileNamesR = glob.glob(folderName + "/rightFisheye/*")
    
    fileNamesL.sort()
    fileNamesR.sort()
    
    os.mkdir(folderName+"/leftFisheye_undistorted")
    os.mkdir(folderName+"/rightFisheye_undistorted")
    
    for imL, imR in zip(fileNamesL, fileNamesR):
        imL_ = cv2.imread(imL, cv2.IMREAD_UNCHANGED)
        imR_ = cv2.imread(imR, cv2.IMREAD_UNCHANGED)
        
        imL_undistorted, KL = undistortFisheyeImages([imL_], K1, D1)
        imR_undistorted, KR = undistortFisheyeImages([imR_], K2, D2)
        
        imL_undistorted = imL_undistorted[0]
        imR_undistorted = imR_undistorted[0]
        
        cv2.imwrite(folderName + "/leftFisheye_undistorted/" + imL.split("/")[-1] + ".png", imL_undistorted)
        cv2.imwrite(folderName + "/rightFisheye_undistorted/" + imR.split("/")[-1] + ".png", imR_undistorted)
        
    
file = open("Calibration/Missions/Mission_4/intrinsics.txt", "w")
file.write("\n*KL:\n")
file.write(str(KL))
file.write("\n*KR:\n")
file.write(str(KR))
file.close()








