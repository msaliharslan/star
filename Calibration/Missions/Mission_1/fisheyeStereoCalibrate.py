"""
elde var sıfır
"""
import numpy as np
import cv2

objectPoints = np.array( [[]] )
imagePoints1 = np.array( [[]] )
imagePoints2 = np.array( [[]] )

K1 = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
D1 = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])

K2 = np.array([[284.1828918457031, 0.0, 427.9779052734375], [0.0, 285.0440979003906, 399.5506896972656], [0.0, 0.0, 1.0]])
D2 = np.array([0.0009760634857229888, 0.030147459357976913, -0.02769969031214714, 0.0031066760420799255])
 
fisheyeWidth = 800
fisheyeHeight = 848
imageSize = (fisheyeWidth, fisheyeHeight)

flags = cv2.fisheye.CALIB_FIX_INTRINSIC
criteria = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 100, 1e-3)

retval, K1, D1, K2, D2, R, T = cv2.fisheye.stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, flags = flags)