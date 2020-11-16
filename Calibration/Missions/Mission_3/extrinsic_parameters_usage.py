# -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

if(os.getcwd().split('/')[-1] != 'star'):
    os.chdir("../../../")
    
import Calibration.Missions.Mission_1.transformation as trans
# from Calibration.Missions.Mission_1.fisheyeStereoCalibrate import calculateFundamentalMatrix
import Calibration.Missions.Mission_2.photo_quality_control as cont

def calculateFundamentalMatrix(K1, K2,  R,T):

    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3,1)))))
    P2 = np.dot(K2, np.hstack((R, T.reshape(3,1)) ) )

    #calculating epipole'

    e_hat = np.dot(K2, T)

    #reference vision book page 581
    e_hat_crosser = np.array([  [ 0, -e_hat[2], e_hat[1]], \
                                [e_hat[2], 0, -e_hat[0]], \
                                [-e_hat[1], e_hat[0], 0]    ])


    F = np.dot(e_hat_crosser, np.dot(P2, np.linalg.pinv(P1)))   

    return F 


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,_ = img1.shape
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        pt1 = tuple(map(tuple, pt1))[0]
        pt2 = tuple(map(tuple, pt2))[0]
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,pt1,5,color,-1)
        img2 = cv2.circle(img2,pt2,5,color,-1)
    return img1,img2


F = calculateFundamentalMatrix(cont.KL, cont.KR, trans.R, trans.T)

errs = []
for i in range(len(cont.undistortedL)):
    lines1 = cv2.computeCorrespondEpilines(cont.cornersR[i].reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(cont.undistortedL[i],cont.undistortedR[i],lines1,cont.cornersL[i], cont.cornersR[i])
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(cont.cornersL[i].reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(cont.undistortedR[i],cont.undistortedL[i],lines2,cont.cornersR[i], cont.cornersL[i])
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()
    
    
    img = cv2.hconcat([img5, img3])
    cv2.imwrite("./temp/"+"George_Bush"+ str(i) +".png", img)
    
    
    ####################################
    ## Error calculation
    ######################
    
    err = 0
    
    for pt, l in zip(cont.cornersL[i], lines1):
        err += abs(pt[0,0] * l[0] + pt[0,1] * l[1] + l[2])
    
    for pt, l in zip(cont.cornersR[i], lines2):
        err += abs(pt[0,0] * l[0] + pt[0,1] * l[1] + l[2])
        
        
    err /= lines1.shape[0] + lines2.shape[0]
    errs.append(err)





























