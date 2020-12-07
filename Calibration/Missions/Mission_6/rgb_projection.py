# -*- coding: utf-8 -*-


######################################################
############### Step 1 ###############################
######################################################
import re
from ast import literal_eval
import numpy as np
import os
from scipy.spatial.transform import Rotation as rot
import cv2
import glob


if(os.getcwd().split('/')[-1] != 'star'):
    os.chdir("../../../")
    
def str_to_array(arr):
    arr = re.sub(r"([^[])\s+([^]])", r"\1, \2", arr)
    arr = np.array(literal_eval(arr))
    return arr
    
extrinsics_file = open("Calibration/Missions/Mission_5/extrinsics.txt", "r")
extrinsics_all = extrinsics_file.read()
extrinsics_file.close()
del extrinsics_file

entries = re.split(":+", extrinsics_all)
entries = [entry.split('*') for entry in entries]

T_C2LF = entries[1][0]
T_C2LF = str_to_array(T_C2LF)

R_C2LF = entries[2][0]
R_C2LF = str_to_array(R_C2LF)

T_C2RF = entries[3][0]
T_C2RF = str_to_array(T_C2RF)

R_C2RF = entries[4][0]
R_C2RF = str_to_array(R_C2RF)

T_RF2LF = entries[5][0]
T_RF2LF = str_to_array(T_RF2LF)

R_RF2LF = entries[6][0]
R_RF2LF = str_to_array(R_RF2LF)


##################################
##################################
R_LF2RF = np.linalg.inv(R_RF2LF)
T_LF2RF = -1 * (np.dot(R_LF2RF, T_RF2LF))

R_RF2C = np.linalg.inv(R_C2RF)
T_RF2C = -1 * (np.dot(R_RF2C, T_C2RF))

# CHECKING
T = np.dot(R_RF2C, np.dot(R_LF2RF, T_C2LF) + T_LF2RF) + T_RF2C
R = np.dot(np.dot(R_C2LF, R_LF2RF), R_RF2C)

#################
T4 = np.array([0.014851336367428, 0.0004623454588, 0.000593442469835]) # from RGB to depth(infrared 1)

R4 = np.array([-0.003368464997038, -0.000677574775182, -0.006368808448315, -0.999973833560944]) # from RGB to depth(infrared 1)
R4 = rot.from_quat(R4).as_matrix()
R4 = np.linalg.inv(R4)
T4 = np.dot(R4, -T4)

KDep = np.array([[634.013671875, 0.0, 635.5408325195312], [0.0, 634.013671875, 351.9051208496094], [0.0, 0.0, 1.0]])
KDep_inv = np.linalg.inv(KDep)
KRgb = np.array([[923.2835693359375, 0.0, 639.9102783203125], [0.0, 923.6146240234375, 370.2297668457031], [0.0, 0.0, 1.0]])

folderName = "SnapShots/t265_d435i/1_2020-11-03_20:56"
fileNamesDepth = glob.glob(folderName + "/depth/*" )
fileNamesRgb = glob.glob(folderName + "/rgb/*")

fileNamesDepth.sort()
fileNamesRgb.sort()

imgDep = cv2.imread(fileNamesDepth[0], cv2.IMREAD_UNCHANGED)
imgRgb = cv2.imread(fileNamesRgb[0])

shape = imgDep.shape
objp = np.ones((shape[0] * shape[1], 3), np.uint32)
objp[:, :2] = np.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1, 2)
objp = objp.T
worldP = np.dot(KDep_inv, objp)
worldP = worldP.T
worldP = worldP.reshape((shape[0], shape[1], 3) )
imgDep = np.expand_dims(imgDep, axis=2)
worldP = worldP * imgDep * 1e-3

worldP_prime = np.dot(R4, worldP.reshape(shape[0]*shape[1], 3).T) + np.expand_dims(T4, 1)
imgP_prime = np.dot(KRgb, worldP_prime)
imgP_prime /= imgP_prime[2,:]
imgP_prime = np.array(np.round(imgP_prime), dtype=np.int32)

mappedD2C = np.zeros(imgRgb.shape[:2])
for i,pt in enumerate(imgP_prime.T) :
    if(pt[0] < 720 and pt[1] < 1280 and pt[0] >= 0 and pt[1] >= 0):
        depth = worldP_prime[2, i]
        if(mappedD2C[pt[0], pt[1]] == 0 or depth < mappedD2C[pt[0], pt[1]]):
            mappedD2C[pt[0], pt[1]] = depth


# mappedD2C = np.expand_dims(mappedD2C, 2)
# h1, w1 = imgRgb.shape[:2]
# h2, w2 = mappedD2C.shape[:2]

# #create empty matrix
# img = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

# #combine 2 images
# img[:h1, :w1,:3] = imgRgb
# img[:h2, w1:w1+w2,:3] = mappedD2C
cv2.imwrite("./temp/"+"generated_depth" +".png", mappedD2C)

































