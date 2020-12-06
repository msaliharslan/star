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

mappedD2C = np.zeros(imgRgb.shape[:2])

shape = imgDep.shape
for i in range(shape[0]) :
    for j in range(shape[1]):
        if(imgDep[i][j] != 0):
            worldP = np.dot(KDep_inv, (i, j, 1))
            worldP *= imgDep[i][j] * worldP / worldP[2]
            if i == shape[0] // 2 and j == shape[1] // 2:
                print(worldP)



































