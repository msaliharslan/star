import sys
import argparse
import os
import numpy as np
import cv2

os.chdir("..")
sys.path.append(os.getcwd())

import matplotlib.pyplot as plot


import fetcher

fetcher.insertFocusDataPath("fisheyeAtakan")
fetcher.fetchFisheyeDataT265()

targetFrameIndex = 200

fisheyeWidth_a = 800
fisheyeHeight_a = 848

fisheyeWidth_b = 848
fisheyeHeight_b = 800

data = np.array((fetcher.fisheye1_t265[targetFrameIndex][1:-1]).split(", "),dtype=np.uint8)
data = data.reshape((fisheyeWidth_b,fisheyeHeight_b))
data = data.reshape((fisheyeWidth_a,fisheyeHeight_a))


K1 = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
K2 = np.array([284.1828918457031, 0.0, 427.9779052734375, 0.0, 285.0440979003906, 399.5506896972656, 0.0, 0.0, 1.0])
K2 = K2.reshape((3,3))
D1 = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])
D2 = np.array([0.0009760634857229888, 0.030147459357976913, -0.02769969031214714, 0.0031066760420799255])


undistorted = cv2.fisheye.undistortImage(data, K=K1, D=D1)

cv2.imshow("title", undistorted.reshape((fisheyeWidth_a, fisheyeHeight_a)))
cv2.imshow("title2", data.reshape((fisheyeWidth_a, fisheyeHeight_a)))



m1type = cv2.CV_32FC1
(lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K1, D1, np.eye(3), np.eye(3), (300,300), m1type)

godPls =cv2.remap(src = data ,map1 = lm1,map2 = lm2,interpolation = cv2.INTER_LINEAR)

cv2.imshow("title", godPls)


print(fetcher.fisheye1_t265.shape)










