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

targetFrameIndex = 10

fisheyeWidth = 800
fisheyeHeight = 848


data = np.array((fetcher.fisheye1_t265[targetFrameIndex][1:-1]).split(", "),dtype=np.uint8)
data = data.reshape((fisheyeWidth,fisheyeHeight))


K1 = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
K2 = np.array([284.1828918457031, 0.0, 427.9779052734375, 0.0, 285.0440979003906, 399.5506896972656, 0.0, 0.0, 1.0])
K2 = K2.reshape((3,3))
D1 = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])
D2 = np.array([0.0009760634857229888, 0.030147459357976913, -0.02769969031214714, 0.0031066760420799255])


undistorted = cv2.fisheye.undistortImage(data, K=K1, D=D1, Knew=K1, new_size=(int(fisheyeWidth), int(fisheyeWidth)))

cv2.imshow("another title", data)
cv2.imshow("another title", img)

cv2.imshow("randomTitle",undistorted)

plt.imshow( undistorted, cmap='gray')
# plt.imshow( data.reshape((fisheyeWidth_a, fisheyeHeight_a)))



# m1type = cv2.CV_32FC1
# (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K1, D1, np.eye(3), np.eye(3), (300,300), m1type)

# godPls =cv2.remap(src = data ,map1 = lm1,map2 = lm2,interpolation = cv2.INTER_LINEAR)

# plt.imshow(godPls)

img = cv2.cvtColor(undistorted,cv2.COLOR_GRAY2BGR)



gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]









edges = cv2.Canny(undistorted,50,150,apertureSize = 3)

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
    
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)






