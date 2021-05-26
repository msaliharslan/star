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
import matplotlib.pyplot as plt
from PIL import Image

if(os.getcwd().split('/')[-1] != 'star'):
    os.chdir("../../../")
    
def str_to_array(arr):
    arr = re.sub(r"([^[])\s+([^]])", r"\1, \2", arr)
    arr = np.array(literal_eval(arr))
    return arr
    
def normalizeImageAndSave_toTemp(image, name):
    
    # Normalisation
    if(len(image.shape) == 3  and image.shape[-1] == 3): # normalise only rgb
        image = image / np.max(image[~np.isnan(image)]) * (2**16-1)
        image = (image/256).astype(np.uint8) 
    else:
        image = image / np.max(image[~np.isnan(image)]) * (2**16-1)
        image = image.astype(np.uint16)
    # numpy save
    # file = open("./thingsToSubmit/submission4/"+ name +".npy", "wb")
    # np.save(file, image)
    # file.close()
    
    # OpenCV save
    cv2.imwrite("./thingsToSubmit/submission4/"+ name +".png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9] )
    
    # Pillow save
    # toSave = Image.fromarray(image)
    # toSave.save("./thingsToSubmit/submission4/"+ name +".png")


extrinsics_file = open("Calibration/Missions/Mission_5/extrinsics.txt", "r")
extrinsics_all = extrinsics_file.read()
extrinsics_file.close()
del extrinsics_file

entries = re.split(":+", extrinsics_all)
entries = [entry.split('*') for entry in entries]

KL = np.array([[500. ,   0. , 499.5],
       [  0. , 500. , 499.5],
       [  0. ,   0. ,   1. ]])

KR = np.array([[500. ,   0. , 499.5],
 [  0. , 500. , 499.5],
 [  0.,    0. ,   1. ]])



T_LF2C = entries[1][0]
T_LF2C = str_to_array(T_LF2C)

R_LF2C = entries[2][0]
R_LF2C = str_to_array(R_LF2C)

T_RF2C = entries[3][0]
T_RF2C = str_to_array(T_RF2C)

R_RF2C = entries[4][0]
R_RF2C = str_to_array(R_RF2C)

T_LF2RF = entries[5][0]
T_LF2RF = str_to_array(T_LF2RF)

R_LF2RF = entries[6][0]
R_LF2RF = str_to_array(R_LF2RF)


##################################
##################################
R_C2LF = np.linalg.inv(R_LF2C)
T_C2LF = -np.dot(R_C2LF, T_LF2C)

R_C2RF = np.linalg.inv(R_RF2C)
T_C2RF = -np.dot(R_C2RF, T_RF2C)


#################
T4 = np.array([0.014851336367428, 0.0004623454588, 0.000593442469835]) # from depth to RGB(infrared 1)

R4 = np.array([-0.003368464997038, -0.000677574775182, -0.006368808448315, -0.999973833560944]) # from depth to RGB(infrared 1)
R4 = rot.from_quat(R4).as_matrix()


KDep = np.array([[634.013671875, 0.0, 635.5408325195312], [0.0, 634.013671875, 351.9051208496094], [0.0, 0.0, 1.0]])
KDep_inv = np.linalg.inv(KDep)
KRgb = np.array([[923.2835693359375, 0.0, 639.9102783203125], [0.0, 923.6146240234375, 370.2297668457031], [0.0, 0.0, 1.0]])

folderName = "SnapShots/t265_d435i/1_2020-11-03_20:56"
fileNamesDepth = glob.glob(folderName + "/depth/*" )
fileNamesRgb = glob.glob(folderName + "/rgb/*")
fileNamesLeft = glob.glob(folderName + "/leftFisheye_undistorted/*")
fileNamesRight = glob.glob(folderName + "/rightFisheye_undistorted/*")


fileNamesDepth.sort()
fileNamesRgb.sort()
fileNamesLeft.sort()
fileNamesRight.sort()

imgDep = cv2.imread(fileNamesDepth[1], cv2.IMREAD_UNCHANGED)
imgRgb = cv2.imread(fileNamesRgb[1], cv2.IMREAD_UNCHANGED)
imgLeft = cv2.imread(fileNamesLeft[1], cv2.IMREAD_UNCHANGED)
imgRight = cv2.imread(fileNamesRight[1] , cv2.IMREAD_UNCHANGED)


# Up sample images
    
u_left = 1
u_right = 1
u_rgb = 1

imgLeft = cv2.resize(imgLeft, None, fx = u_left, fy = u_left, interpolation =  cv2.INTER_CUBIC )
imgRight = cv2.resize(imgRight, None, fx = u_right, fy = u_right, interpolation =  cv2.INTER_CUBIC )
imgRgb = cv2.resize(imgRgb, None, fx = u_rgb, fy = u_rgb, interpolation =  cv2.INTER_CUBIC )

# Change Camera Matrixes Accordingly


KL*= u_left
KL[2,2] = 1
KR *= u_right
KR[2,2] = 1
KRgb *= u_rgb
KRgb[2,2] = 1

shape = imgDep.shape
objp = np.ones((shape[0] * shape[1], 3), np.uint32)
objp[:, :2] = np.mgrid[ 0:shape[1],0:shape[0]].T.reshape(-1, 2)
objp = objp.T
worldP_depth = np.dot(KDep_inv, objp)
worldP_depth = worldP_depth.T
worldP_depth = worldP_depth.reshape((shape[0], shape[1], 3) )
imgDep = np.expand_dims(imgDep, axis=2)
worldP_depth = worldP_depth * imgDep * 1e-3


worldP_color = np.dot(R4, worldP_depth.reshape(shape[0]*shape[1], 3).T) + np.expand_dims(T4, 1)
worldP_left = np.dot(R_C2LF, worldP_color) + T_C2LF
worldP_right = np.dot(R_C2RF, worldP_color) + T_C2RF

pt_cld_file = open("./thingsToSubmit/submission4/leftPackage/pt_cloud.npy", "wb")
np.save(pt_cld_file, worldP_left)
pt_cld_file.close()

pt_cld_file = open("./thingsToSubmit/submission4/rightPackage/pt_cloud.npy", "wb")
np.save(pt_cld_file, worldP_right)
pt_cld_file.close()

imgP_color = np.dot(KRgb, worldP_color)
imgP_color /= imgP_color[2,:]
imgP_color = np.array(np.round(imgP_color), dtype=np.int32)

imgP_left = np.dot(KL, worldP_left)
imgP_left /= imgP_left[2,:]
imgP_left = np.array(np.round(imgP_left), dtype=np.int32)

imgP_right = np.dot(KR, worldP_right)
imgP_right /= imgP_right[2,:]
imgP_right = np.array(np.round(imgP_right), dtype=np.int32)

# RGB - LEFT - RIGHT - DEPTH     
def downSampleImage(image, downSampleRate):
    if(len(image.shape) == 3 and image.shape[-1] == 3):
        newImage = np.zeros((int(image.shape[0]/downSampleRate), int(image.shape[1]/downSampleRate), image.shape[2]))
        area = np.zeros((int(image.shape[0]/downSampleRate), int(image.shape[1]/downSampleRate)))
        for i in range(downSampleRate):
            for j in range(downSampleRate):    
                newImage += image[j::downSampleRate, i::downSampleRate, :]
                area += np.sum(image[j::downSampleRate, i::downSampleRate, :], 2) > 0
        
        newImage[area>0,:] /= np.expand_dims(area[area>0], 1)
        return newImage.astype(dtype=image.dtype)
    
    else:
        image = np.squeeze(image)
        newImage = np.zeros((int(image.shape[0]/downSampleRate), int(image.shape[1]/downSampleRate)))
        area = np.zeros((int(image.shape[0]/downSampleRate), int(image.shape[1]/downSampleRate)))
        for i in range(downSampleRate):
            for j in range(downSampleRate):    
                newImage += image[j::downSampleRate, i::downSampleRate]
                area += image[j::downSampleRate, i::downSampleRate] > 0
    
        newImage[area>0] /= area[area>0]
        return newImage.astype(dtype=image.dtype)
    
    # if(len(image.shape) == 3 and image.shape[-1] == 3):
    #     newImage = np.zeros((int(image.shape[0]/downSampleRate), int(image.shape[1]/downSampleRate), image.shape[2]))
    #     for i in range(int(image.shape[0] / downSampleRate)):
    #         for j in range(int(image.shape[1] / downSampleRate)):
    #             nonZeroCount = np.sum(np.any(image[i*downSampleRate:(i+1)*downSampleRate,j*downSampleRate:(j+1)*downSampleRate] != 0, axis=2) )
    #             summR = np.sum(image[i*downSampleRate:(i+1)*downSampleRate,j*downSampleRate:(j+1)*downSampleRate, 0])
    #             summG = np.sum(image[i*downSampleRate:(i+1)*downSampleRate,j*downSampleRate:(j+1)*downSampleRate, 1])
    #             summB = np.sum(image[i*downSampleRate:(i+1)*downSampleRate,j*downSampleRate:(j+1)*downSampleRate, 2])
    #             if(nonZeroCount != 0):
    #                 newImage[i,j] = np.array( [summR / nonZeroCount, summG / nonZeroCount, summB / nonZeroCount] )
    
    # else:
    #     newImage = np.zeros((int(image.shape[0]/downSampleRate), int(image.shape[1]/downSampleRate)))

    #     for i in range(int(image.shape[0] / downSampleRate)):
    #         for j in range(int(image.shape[1] / downSampleRate)):
    #             nonZeroCount = np.sum(image[i*downSampleRate:(i+1)*downSampleRate,j*downSampleRate:(j+1)*downSampleRate] != 0)
    #             summ = np.sum(image[i*downSampleRate:(i+1)*downSampleRate,j*downSampleRate:(j+1)*downSampleRate])
    #             if(nonZeroCount != 0):
    #                 newImage[i,j] = summ / nonZeroCount
    
    # return newImage.astype(dtype=image.dtype)
            
            

def generateRgbPackageMatrix(mapMatrix, imgRgb, imgLeft, imgRight):
    
    rgbPackageMatrix = np.zeros((imgRgb.shape[0],imgRgb.shape[1], 10)) * np.nan # r,g,b, fd, depth, fleft, leftIntensity, fright, rightIntensity, point_index
    index_arr = np.arange(0, mapMatrix.shape[0], dtype=np.uint32)
    
    rgbPackageMatrix[:,:,0:3] = imgRgb
  
    # frgb, xrgb, yrgb,   fleft, xleft, yleft,  fright, xright, yright    
      
    frgb = mapMatrix[:,0]
    fleft = mapMatrix[:,4]
    fright = mapMatrix[:,8]
    
    fleft = np.logical_and(frgb, fleft)
    fright = np.logical_and(frgb, fright)
    
    indexRgb = frgb.nonzero()[0]
    indexLeft = fleft.nonzero()[0]
    indexRight = fright.nonzero()[0]
    
    temp = rgbPackageMatrix[mapMatrix[indexRgb,:][:,2], mapMatrix[indexRgb,:][:,1]]
    temp[:,3:5] = np.vstack((np.ones(indexRgb.shape), mapMatrix[indexRgb,:][:,3])).T
    rgbPackageMatrix[mapMatrix[indexRgb,:][:,2], mapMatrix[indexRgb,:][:,1]] = temp
    
    temp = rgbPackageMatrix[mapMatrix[indexLeft,:][:,2], mapMatrix[indexLeft,:][:,1]]
    temp[:,5:7] = np.vstack((np.ones(indexLeft.shape), imgLeft[mapMatrix[indexLeft,:][:,6], mapMatrix[indexLeft,:][:,5]])).T
    rgbPackageMatrix[mapMatrix[indexLeft,:][:,2], mapMatrix[indexLeft,:][:,1]] = temp
    
    temp = rgbPackageMatrix[mapMatrix[indexRight,:][:,2], mapMatrix[indexRight,:][:,1]]
    temp[:,7:9] = np.vstack((np.ones(indexRight.shape), imgRight[mapMatrix[indexRight,:][:,10], mapMatrix[indexRight,:][:,9]])).T
    rgbPackageMatrix[mapMatrix[indexRight,:][:,2], mapMatrix[indexRight,:][:,1]] = temp
    
    rgbPackageMatrix[mapMatrix[indexRgb,:][:,2], mapMatrix[indexRgb,:][:,1], 9] = index_arr[indexRgb] 

    # for mapp in tqdm(mapMatrix):
    #     if(mapp[0]):
            
    #         leftIntensity = 0
    #         rightIntensity = 0
            
    #         if(mapp[4]):
    #             leftIntensity = imgLeft[mapp[6], mapp[5]]
    #         if(mapp[8]):
    #             rightIntensity = imgRight[mapp[10], mapp[9]]
                
    #         rgbPackageMatrix[mapp[2], mapp[1]][3:] = [1, mapp[3], mapp[4], leftIntensity, mapp[8], rightIntensity ]
        
    return rgbPackageMatrix
    
def visualizeRgbPackageMatrix(rgbPackageMatrix): # r,g,b, fd, depth, fleft, leftIntensity, fright, rightIntensity
        
    # First original rgb image
    
    imgRgb = rgbPackageMatrix[:,:,0:3]
    flagRgb = np.ones((imgRgb.shape[0], imgRgb.shape[1], 1), dtype=imgRgb.dtype)     
    

    imgRgb = downSampleImage(imgRgb, u_rgb) # cv2.resize(imgRgb, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )
    flagRgb = downSampleImage(flagRgb, u_rgb) #  cv2.resize(flagRgb, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )                

    normalizeImageAndSave_toTemp(imgRgb, "rgbPackage/packageRgb_rgb")
    normalizeImageAndSave_toTemp(flagRgb, "rgbPackage/packageRgb_rgbFlag")
    
    # Second corresponding depth image
    
    imgDepth = rgbPackageMatrix[:,:,4]
    flagDepth = rgbPackageMatrix[:,:, 3]
    
    imgDepth = downSampleImage(imgDepth, u_rgb) # cv2.resize(imgDepth, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )
    flagDepth = downSampleImage(flagDepth, u_rgb) # cv2.resize(flagDepth, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )
    
        
    normalizeImageAndSave_toTemp(imgDepth, "rgbPackage/packageRgb_depth")
    normalizeImageAndSave_toTemp(flagDepth, "rgbPackage/packageRgb_depthFlag")
    
    
    # Third corresponding left image
    
    imgLeft = rgbPackageMatrix[:,:,6]
    flagLeft = rgbPackageMatrix[:,:,5]
    
    imgLeft = downSampleImage(imgLeft, u_rgb) # cv2.resize(imgLeft, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )
    flagLeft = downSampleImage(flagLeft, u_rgb) # cv2.resize(flagLeft, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )

    
    normalizeImageAndSave_toTemp(imgLeft, "rgbPackage/packageRgb_left")
    normalizeImageAndSave_toTemp(flagLeft, "rgbPackage/packageRgb_leftFlag")
    
    # Fourth corresponding right image
    
    imgRight = rgbPackageMatrix[:, :, 8]
    flagRight = rgbPackageMatrix[:,:,7]
    
    imgRight = downSampleImage(imgRight, u_rgb) # cv2.resize(imgRight, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )
    flagRight = downSampleImage(flagRight, u_rgb) # cv2.resize(flagRight, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )
    
    
    normalizeImageAndSave_toTemp(imgRight, "rgbPackage/packageRgb_right")
    normalizeImageAndSave_toTemp(flagRight, "rgbPackage/packageRgb_rightFlag")
    
        
    
    
    
def generateLeftPackageMatrix(mapMatrix, imgRgb, imgLeft, imgRight):
    
    leftPackageMatrix = np.zeros((imgLeft.shape[0],imgLeft.shape[1], 10)) * np.nan # leftIntensity, fd, depth, frgb, r, g, b, fright, rightIntensity
    index_arr = np.arange(0, mapMatrix.shape[0], dtype=np.uint32)

    leftPackageMatrix[:,:,0] = imgLeft
    
    frgb = mapMatrix[:,0]
    fleft = mapMatrix[:,4]
    fright = mapMatrix[:,8]
    
    frgb = np.logical_and(fleft, frgb)
    fright = np.logical_and(fleft, fright)
    
    indexRgb = frgb.nonzero()[0]
    indexLeft = fleft.nonzero()[0]
    indexRight = fright.nonzero()[0]
    
    # frgb, xrgb, yrgb, rgbDepth,  fleft, xleft, yleft, leftDepth  fright, xright, yright, rightDepth
    
    temp = leftPackageMatrix[mapMatrix[indexLeft,:][:,6], mapMatrix[indexLeft,:][:,5]]
    temp[:,1:3] = np.vstack((np.ones(indexLeft.shape), mapMatrix[indexLeft,:][:,7])).T
    leftPackageMatrix[mapMatrix[indexLeft,:][:,6], mapMatrix[indexLeft,:][:,5]] = temp
    
    temp = leftPackageMatrix[mapMatrix[indexRgb,:][:,6], mapMatrix[indexRgb,:][:,5]]
    temp[:,3:7] = np.concatenate((np.expand_dims(np.ones(indexRgb.shape), 1), imgRgb[mapMatrix[indexRgb,:][:,2], mapMatrix[indexRgb,:][:,1]]), 1)
    leftPackageMatrix[mapMatrix[indexRgb,:][:,6], mapMatrix[indexRgb,:][:,5]] = temp
    
    temp = leftPackageMatrix[mapMatrix[indexRight,:][:,6], mapMatrix[indexRight,:][:,5]]
    temp[:,7:9] = np.vstack((np.ones(indexRight.shape), imgRight[mapMatrix[indexRight,:][:,10], mapMatrix[indexRight,:][:,9]])).T
    leftPackageMatrix[mapMatrix[indexRight,:][:,6], mapMatrix[indexRight,:][:,5]] = temp

    leftPackageMatrix[mapMatrix[indexLeft,:][:,6], mapMatrix[indexLeft,:][:,5], 9] = index_arr[indexLeft]

    # for mapp in tqdm(mapMatrix):
    #     if(mapp[4]):
            
    #         r = 0
    #         g = 0
    #         b = 0
    #         rightIntensity = 0
            
    #         if(mapp[0]):
    #             r = imgRgb[mapp[2], mapp[1]][0]
    #             g = imgRgb[mapp[2], mapp[1]][1]
    #             b = imgRgb[mapp[2], mapp[1]][2]
                
    #         if(mapp[8]):
    #             rightIntensity = imgRight[mapp[10], mapp[9]]
                
    #         leftPackageMatrix[mapp[6], mapp[5]][1:9] = [1, mapp[7], mapp[0], r, g, b, mapp[8], rightIntensity]
            
        
    return leftPackageMatrix


def visualizeLeftPackageMatrix(leftPackageMatrix):
    
    # Save Package matrix itself first
    package_file = open("./thingsToSubmit/submission4/leftPackage/leftPackage.npy", "wb")
    np.save(package_file, leftPackageMatrix)
    package_file.close()
    
    # leftIntensity, fd, depth, frgb, r, g, b, fright, rightIntensity
    
    # First original rgb image
    
    imgRgb = leftPackageMatrix[:,:,4:7]
    flagRgb = leftPackageMatrix[:,:,3]  
    

    
    imgRgb = downSampleImage(imgRgb, u_left) # cv2.resize(imgRgb, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )
    flagRgb = downSampleImage(flagRgb, u_left) # cv2.resize(flagRgb, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )           
        
    
    normalizeImageAndSave_toTemp(imgRgb, "leftPackage/packageLeft_rgb")
    normalizeImageAndSave_toTemp(flagRgb, "leftPackage/packageLeft_rgbFlag")
    
    # Second corresponding depth image
    
    imgDepth = leftPackageMatrix[:,:,2]
    flagDepth = leftPackageMatrix[:,:, 1]
    
    imgDepth = downSampleImage(imgDepth, u_left) # cv2.resize(imgDepth, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )
    flagDepth = downSampleImage(flagDepth, u_left) # cv2.resize(flagDepth, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )             
    
    normalizeImageAndSave_toTemp(imgDepth, "leftPackage/packageLeft_depth")
    normalizeImageAndSave_toTemp(flagDepth, "leftPackage/packageLeft_depthFlag")
    
    
    # Third corresponding left image
    
    imgLeft = leftPackageMatrix[:,:,0]
    flagLeft = np.ones((imgLeft.shape[0], imgLeft.shape[1], 1), dtype=imgLeft.dtype)  
    
    imgLeft = downSampleImage(imgLeft, u_left) # cv2.resize(imgLeft, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )
    flagLeft = downSampleImage(flagLeft, u_left) # cv2.resize(flagLeft, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )   
                   

    normalizeImageAndSave_toTemp(imgLeft, "leftPackage/packageLeft_left")
    normalizeImageAndSave_toTemp(flagLeft, "leftPackage/packageLeft_leftFlag")
    
    # Fourth corresponding right image
    
    imgRight = leftPackageMatrix[:, :, 8]
    flagRight = leftPackageMatrix[:,:,7]
    
    imgRight = downSampleImage(imgRight, u_left) # cv2.resize(imgRight, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )
    flagRight = downSampleImage(flagRight, u_left) # cv2.resize(flagRight, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )      
    
    normalizeImageAndSave_toTemp(imgRight, "leftPackage/packageLeft_right")
    normalizeImageAndSave_toTemp(flagRight, "leftPackage/packageLeft_rightFlag")
    



def generateRightPackageMatrix(mapMatrix, imgRgb, imgLeft, imgRight):
    
    rightPackageMatrix = np.zeros((imgRight.shape[0],imgRight.shape[1], 10)) * np.nan # rightIntensity, fd, depth, frgb, r, g, b, fleft, leftIntensity
    index_arr = np.arange(0, mapMatrix.shape[0], dtype=np.uint32)

    rightPackageMatrix[:,:,0] = imgRight
  
    frgb = mapMatrix[:,0]
    fleft = mapMatrix[:,4]
    fright = mapMatrix[:,8]
    
    frgb = np.logical_and(fright, frgb)
    fleft = np.logical_and(fright, fleft)
    
    indexRgb = frgb.nonzero()[0]
    indexLeft = fleft.nonzero()[0]
    indexRight = fright.nonzero()[0]
    
    # frgb, xrgb, yrgb, rgbDepth,  fleft, xleft, yleft, leftDepth  fright, xright, yright, rightDepth
    
    temp = rightPackageMatrix[mapMatrix[indexRight,:][:,10], mapMatrix[indexRight,:][:,9]]
    temp[:,1:3] = np.vstack((np.ones(indexRight.shape), mapMatrix[indexRight,:][:,11])).T
    rightPackageMatrix[mapMatrix[indexRight,:][:,10], mapMatrix[indexRight,:][:,9]] = temp
    
    temp = rightPackageMatrix[mapMatrix[indexRgb,:][:,10], mapMatrix[indexRgb,:][:,9]]
    temp[:,3:7] = np.concatenate((np.expand_dims(np.ones(indexRgb.shape), 1), imgRgb[mapMatrix[indexRgb,:][:,2], mapMatrix[indexRgb,:][:,1]]), 1)
    rightPackageMatrix[mapMatrix[indexRgb,:][:,10], mapMatrix[indexRgb,:][:,9]] = temp
    
    temp = rightPackageMatrix[mapMatrix[indexLeft,:][:,10], mapMatrix[indexLeft,:][:,9]]
    temp[:,7:9] = np.vstack((np.ones(indexLeft.shape), imgLeft[mapMatrix[indexLeft,:][:,6], mapMatrix[indexLeft,:][:,5]])).T
    rightPackageMatrix[mapMatrix[indexLeft,:][:,10], mapMatrix[indexLeft,:][:,9]] = temp
 
    rightPackageMatrix[mapMatrix[indexRight,:][:,10], mapMatrix[indexRight,:][:,9], 9] = index_arr[indexRight]
                   
    
    # for mapp in tqdm(mapMatrix):
    #     if(mapp[8]):
            
    #         r = 0
    #         g = 0
    #         b = 0
    #         leftIntensity = 0
            
    #         if(mapp[0]):
    #             r = imgRgb[mapp[2], mapp[1]][0]
    #             g = imgRgb[mapp[2], mapp[1]][1]
    #             b = imgRgb[mapp[2], mapp[1]][2]
                
    #         if(mapp[4]):
    #             leftIntensity = imgLeft[mapp[6], mapp[5]]
                
    #         rightPackageMatrix[mapp[10], mapp[9]][1:9] = [1, mapp[11], mapp[0], r, g, b, mapp[4], leftIntensity]
            
        
    return rightPackageMatrix
    
def visualizeRightPackageMatrix(rightPackageMatrix):
    
    # Save Package matrix itself first
    package_file = open("./thingsToSubmit/submission4/rightPackage/rightPackage.npy", "wb")
    np.save(package_file, rightPackageMatrix)
    package_file.close()
    
    # rightIntensity, fd, depth, frgb, r, g, b, fleft, leftIntensity
    
    # First original rgb image
    
    imgRgb = rightPackageMatrix[:,:,4:7]
    flagRgb = rightPackageMatrix[:,:,3]   

    imgRgb = downSampleImage(imgRgb, u_right) # cv2.resize(imgRgb, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )
    flagRgb = downSampleImage(flagRgb, u_right) # cv2.resize(flagRgb, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )          
    
    normalizeImageAndSave_toTemp(imgRgb, "rightPackage/packageRight_rgb")
    normalizeImageAndSave_toTemp(flagRgb, "rightPackage/packageRight_rgbFlag")
    
    # Second corresponding depth image
    
    imgDepth = rightPackageMatrix[:,:,2]
    flagDepth = rightPackageMatrix[:,:, 1]
    
    imgDepth = downSampleImage(imgDepth, u_right) # cv2.resize(imgDepth, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )
    flagDepth = downSampleImage(flagDepth, u_right) # cv2.resize(flagDepth, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )              
    
    normalizeImageAndSave_toTemp(imgDepth, "rightPackage/packageRight_depth")
    normalizeImageAndSave_toTemp(flagDepth, "rightPackage/packageRight_depthFlag")
    
    
    # Third corresponding left image
    
    imgLeft = rightPackageMatrix[:,:,8]
    flagLeft = rightPackageMatrix[:,:,7] 
    
    imgLeft = downSampleImage(imgLeft, u_right) # cv2.resize(imgLeft, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )
    flagLeft = downSampleImage(flagLeft, u_right) # cv2.resize(flagLeft, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )     

    normalizeImageAndSave_toTemp(imgLeft, "rightPackage/packageRight_left")
    normalizeImageAndSave_toTemp(flagLeft, "rightPackage/packageRight_leftFlag")
    
    # Fourth corresponding right image
    
    imgRight = rightPackageMatrix[:, :, 0]
    flagRight = np.ones((imgRight.shape[0], imgRight.shape[1], 1), dtype=imgRight.dtype)
    
    imgRight = downSampleImage(imgRight, u_right) # cv2.resize(imgRight, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )
    flagRight = downSampleImage(flagRight, u_right) # cv2.resize(flagRight, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )         
    
    normalizeImageAndSave_toTemp(imgRight, "rightPackage/packageRight_right")
    normalizeImageAndSave_toTemp(flagRight, "rightPackage/packageRight_rightFlag")
    
    
    
    
def generateDepthPackageMatrix(mapMatrix, imgRgb, imgLeft, imgRight, imgDep):
    depthPackageMatrix = np.zeros((imgDep.shape[0],imgDep.shape[1], 9))  # depth, frgb, r,g,b, fLeft, leftIntensity, fRight, rightIntensity
                                                                           
    depthPackageMatrix[:,:,0] = imgDep[:,:,0]
    
    depthPackageMatrix = depthPackageMatrix.reshape((imgDep.shape[0] * imgDep.shape[1], 9))
  

    depthPackageMatrix[:,1:] = np.concatenate((np.expand_dims(mapMatrix[:,0] ,1), imgRgb[mapMatrix[:,2], mapMatrix[:,1]], np.expand_dims(mapMatrix[:,4], 1), np.expand_dims(imgLeft[mapMatrix[:,6], mapMatrix[:,5]], 1), np.expand_dims(mapMatrix[:,8], 1), np.expand_dims(imgRight[mapMatrix[:,10], mapMatrix[:,9]],1)), 1)  
    depthPackageMatrix = depthPackageMatrix.reshape((imgDep.shape[0], imgDep.shape[1], 9))

  
    # for i,mapp in tqdm(enumerate(mapMatrix)):
        
         
    #     r = 0
    #     g = 0
    #     b = 0reshape
    #     leftIntensity = 0
    #     rightIntensity = 0
        
    #     if(mapp[0]):
    #         r = imgRgb[mapp[2], mapp[1]][0]
    #         g = imgRgb[mapp[2], mapp[1]][1]
    #         b = imgRgb[mapp[2], mapp[1]][2]
            
    #     if(mapp[4]):
    #         leftIntensity = imgLeft[mapp[6], mapp[5]]
            
    #     if(mapp[8]):
    #         rightIntensity = imgRight[mapp[10], mapp[9]]    
            
    #     row = i // imgDepth.shape[1]
    #     col = i % imgDepth.shape[1]       
            
    #     depthPackageMatrix[row, col][1:9] = [mapp[0], r,g,b, mapp[4], leftIntensity, mapp[8], rightIntensity]
            
        
    return depthPackageMatrix    


def visualizeDepthPackageMatrix(depthPackageMatrix):
    
    # depth, frgb, r,g,b, fLeft, leftIntensity, fRight, rightIntensity    

    # First original rgb image
    
    imgRgb = depthPackageMatrix[:,:,2:5]
    flagRgb = depthPackageMatrix[:,:,1]          
    
    normalizeImageAndSave_toTemp(imgRgb, "depthPackage/packageDepth_rgb")
    normalizeImageAndSave_toTemp(flagRgb, "depthPackage/packageDepth_rgbFlag")
    
    # Second corresponding depth image
    
    imgDepth = depthPackageMatrix[:,:,0]
    flagDepth = imgDepth != 0
    
    normalizeImageAndSave_toTemp(imgDepth, "depthPackage/packageDepth_depth")
    normalizeImageAndSave_toTemp(flagDepth, "depthPackage/packageDepth_depthFlag")
    
    
    # Third corresponding left image
    
    imgLeft = depthPackageMatrix[:,:,6]
    flagLeft = depthPackageMatrix[:,:,5] 

    normalizeImageAndSave_toTemp(imgLeft, "depthPackage/packageDepth_left")
    normalizeImageAndSave_toTemp(flagLeft, "depthPackage/packageDepth_leftFlag")
    
    # Fourth corresponding right image
    
    imgRight = depthPackageMatrix[:, :, 8]
    flagRight = depthPackageMatrix[:, :, 7]
    
    normalizeImageAndSave_toTemp(imgRight, "depthPackage/packageDepth_right")
    normalizeImageAndSave_toTemp(flagRight, "depthPackage/packageDepth_rightFlag")
    
    
    
   
def generateMapMatrix(worldPs_rgb, imgPs_rgb, rgbShape, worldPs_left, imgPs_left, leftShape, worldPs_right, imgPs_right, rightShape):
    worldPs_rgb = worldPs_rgb.T
    worldPs_left = worldPs_left.T
    worldPs_right = worldPs_right.T

    mapMatrix = np.zeros((worldPs_rgb.shape[0], 12), dtype=np.int32) # frgb, xrgb, yrgb, rgbDepth,  fleft, xleft, yleft, leftDepth  fright, xright, yright, rightDepth
    
    flagRgb = np.logical_and(np.logical_and(imgPs_rgb[0,:] >= 0, imgPs_rgb[0,:] < rgbShape[1]), np.logical_and(imgPs_rgb[1,:] >= 0, imgPs_rgb[1,:] < rgbShape[0] ) ) 
    indexRgb = flagRgb.nonzero()[0]
    
    flagLeft = np.logical_and(np.logical_and(imgPs_left[0,:] >= 0, imgPs_left[0,:] < leftShape[1]), np.logical_and(imgPs_left[1,:] >= 0, imgPs_left[1,:] < leftShape[0] ) )
    indexLeft = flagLeft.nonzero()[0]
    
    flagRight = np.logical_and(np.logical_and(imgPs_right[0,:] >= 0, imgPs_right[0,:] < rightShape[1]), np.logical_and(imgPs_right[1,:] >= 0, imgPs_right[1,:] < rightShape[0] ) )
    indexRight = flagRight.nonzero()[0]
    
    # sorted indeces for valid points
    si_wp_rgb = np.argsort(-1 * worldPs_rgb[indexRgb ,2])
    si_wp_left = np.argsort(-1 * worldPs_left[indexLeft, 2])
    si_wp_right = np.argsort(-1 * worldPs_right[indexRight, 2])
    
    
    oc_rm_rgb = np.zeros((rgbShape[0], rgbShape[1]), np.uint32) - 1
    oc_rm_rgb[imgPs_rgb[0:2, indexRgb[si_wp_rgb]][1], imgPs_rgb[0:2, indexRgb[si_wp_rgb]][0]] = indexRgb[si_wp_rgb]
    temp = np.zeros((flagRgb.shape[0] + 1))
    oc_rm_rgb = oc_rm_rgb.reshape((rgbShape[0]*rgbShape[1]) )
    temp[oc_rm_rgb + 1] = 1
    flagRgb = np.logical_and(flagRgb, temp[1:])
    
    oc_rm_left = np.zeros((leftShape[0], leftShape[1]), np.uint32) - 1
    oc_rm_left[imgPs_left[0:2, indexLeft[si_wp_left]][1], imgPs_left[0:2, indexLeft[si_wp_left]][0]] = indexLeft[si_wp_left]
    temp = np.zeros((flagLeft.shape[0] + 1))
    oc_rm_left = oc_rm_left.reshape((leftShape[0]*leftShape[1]) )
    temp[oc_rm_left + 1] = 1
    flagLeft = np.logical_and(flagLeft, temp[1:])
    
    oc_rm_right = np.zeros((rightShape[0], rightShape[1]), np.uint32) - 1
    oc_rm_right[imgPs_right[0:2, indexRight[si_wp_right]][1], imgPs_right[0:2, indexRight[si_wp_right]][0]] = indexRight[si_wp_right]
    temp = np.zeros((flagRight.shape[0] + 1))
    oc_rm_right = oc_rm_right.reshape((rightShape[0]*rightShape[1]) )
    temp[oc_rm_right + 1] = 1
    flagRight = np.logical_and(flagRight, temp[1:])

    # Flags in map matrix
    mapMatrix[:,0] = flagRgb
    mapMatrix[:,4] = flagLeft
    mapMatrix[:,8] = flagRight
    
    
    # Depths in map matrix
    mapMatrix[indexRgb[si_wp_rgb],3] = worldPs_rgb[indexRgb[si_wp_rgb], 2] * 1000
    mapMatrix[indexLeft[si_wp_left],7] = worldPs_left[indexLeft[si_wp_left], 2] * 1000
    mapMatrix[indexRight[si_wp_right],11] = worldPs_right[indexRight[si_wp_right], 2] * 1000
    
    
    # Coordinates in map matrix
    mapMatrix[indexRgb[si_wp_rgb], 1:3] = imgPs_rgb[0:2, indexRgb[si_wp_rgb]].T
    mapMatrix[indexLeft[si_wp_left], 5:7] = imgPs_left[0:2, indexLeft[si_wp_left]].T
    mapMatrix[indexRight[si_wp_right], 9:11] = imgPs_right[0:2, indexRight[si_wp_right]].T
    
    # rgbImageCorresponds = np.zeros((rgbShape[0], rgbShape[1], 2))-1 # channels are : depth, pointIndex
    # leftImageCorresponds = np.zeros((leftShape[0], leftShape[1], 2))-1 # channels are : depth, pointIndex
    # rightImageCorresponds = np.zeros((rightShape[0], rightShape[1], 2))-1 # channels are : depth, pointIndex
    
    # for i in range(imgPs_rgb.shape[1]):
        
    #     imgP_rgb = imgPs_rgb[:,i]
    #     imgP_left = imgPs_left[:,i]
    #     imgP_right = imgPs_right[:,i]
        
    #     depth_rgb = worldPs_rgb[2][i]
    #     x_rgb,y_rgb,_ = imgP_rgb
    #     if((x_rgb >= 0 and x_rgb < rgbShape[1] ) and (y_rgb >= 0 and y_rgb < rgbShape[0])):
    #         if(rgbImageCorresponds[y_rgb,x_rgb][0] == -1 or (rgbImageCorresponds[y_rgb,x_rgb][0] > depth_rgb and depth_rgb != 0)):
    #             rgbImageCorresponds[y_rgb,x_rgb] = [depth_rgb*1000, i]
                   
    #     depth_left = worldPs_left[2][i]
    #     x_left,y_left,_ = imgP_left
    #     if((x_left >= 0 and x_left < leftShape[1] ) and (y_left >= 0 and y_left < leftShape[0])):        
    #         if(leftImageCorresponds[y_left,x_left][0] == -1 or (leftImageCorresponds[y_left,x_left][0] > depth_left and depth_left != 0)):
    #             leftImageCorresponds[y_left,x_left] = [depth_left*1000, i]
            
    #     depth_right = worldPs_right[2][i]
    #     x_right,y_right,_ = imgP_right
    #     if((x_right >= 0 and x_right < rightShape[1] ) and (y_right >= 0 and y_right < rightShape[0])):        
    #         if(rightImageCorresponds[y_right,x_right][0] == -1 or (rightImageCorresponds[y_right,x_right][0] > depth_right and depth_right != 0)):
    #             rightImageCorresponds[y_right,x_right] = [depth_right*1000, i]     
                
                

        
    # for i in range(rgbImageCorresponds.shape[0]):
    #     for j in range(rgbImageCorresponds.shape[1]):
    #         correspond = rgbImageCorresponds[i,j]
    #         if(correspond[1] != -1):
    #             mapMatrix[ int(correspond[1]) ][0:4] = [1, j, i, correspond[0]]
                
    # for i in range(leftImadepthPackageMatrixgeCorresponds.shape[0]):
    #     for j in range(leftImageCorresponds.shape[1]):
    #         correspond = leftImageCorresponds[i,j]
    #         if(correspond[1] != -1):
    #             mapMatrix[ int(correspond[1]) ][4:8] = [1, j, i, correspond[0]]

    # for i in range(rightImageCorresponds.shape[0]):
    #     for j in range(rightImageCorresponds.shape[1]):
    #         correspond = rightImageCorresponds[i,j]
    #         if(correspond[1] != -1):
    #             mapMatrix[ int(correspond[1]) ][8:12] = [1, j, i, correspond[0]]
        
    return mapMatrix



mapMatrix = generateMapMatrix(worldP_color, imgP_color, imgRgb.shape, worldP_left, imgP_left, imgLeft.shape, worldP_right, imgP_right, imgLeft.shape)
print("Map matrix done")

rgbPackageMatrix = generateRgbPackageMatrix(mapMatrix, imgRgb, imgLeft, imgRight)
visualizeRgbPackageMatrix(rgbPackageMatrix)
print("RGB done")

leftPackageMatrix = generateLeftPackageMatrix(mapMatrix, imgRgb, imgLeft, imgRight)
visualizeLeftPackageMatrix(leftPackageMatrix)
print("Left done")

rightPackageMatrix = generateRightPackageMatrix(mapMatrix, imgRgb, imgLeft, imgRight)
visualizeRightPackageMatrix(rightPackageMatrix)
print("Right done")

depthPackageMatrix = generateDepthPackageMatrix(mapMatrix, imgRgb, imgLeft, imgRight, imgDep)
visualizeDepthPackageMatrix(depthPackageMatrix)
print("Depth done")


## RGB
karsilastir = cv2.imread("../sampleDepthRgb.png", cv2.IMREAD_UNCHANGED)
karsilastir = karsilastir / np.max(karsilastir) * (2**16-1)
karsilastir = (karsilastir/256).astype(np.uint8) 

flagKarsi = cv2.imread("../sampleDepthRgbMeasured.png", cv2.IMREAD_UNCHANGED)

imgRgb = depthPackageMatrix[:,:,2:5]
flagRgb = depthPackageMatrix[:,:,1]


imgRgb = imgRgb / np.max(imgRgb) * (255)
imgRgb = imgRgb.astype(np.uint8) 

imgRgb =  imgRgb * np.expand_dims(flagRgb, 2).astype(np.uint8)

fark = np.abs(karsilastir - imgRgb)

np.sum(np.logical_xor(flagKarsi, flagRgb))

# cv2.imshow("aa", fark) 


## LEFT
karsilastir = cv2.imread("../sampleDepthLeft.png", cv2.IMREAD_UNCHANGED)
karsilastir = karsilastir / np.max(karsilastir) * (2**16-1)
karsilastir = (karsilastir/256).astype(np.uint8) 

flagKarsi = cv2.imread("../sampleDepthLeftMeasured.png", cv2.IMREAD_UNCHANGED)

imgLeft = depthPackageMatrix[:,:,6]
flagLeft = depthPackageMatrix[:,:,5] 


imgLeft = imgLeft / np.max(imgLeft) * (255)
imgLeft = imgLeft.astype(np.uint8) 

imgLeft =  (imgLeft * flagLeft).astype(np.uint8)

fark = np.abs(karsilastir.astype(np.float32) - imgLeft.astype(np.float32)).astype(np.uint8)

np.sum(np.logical_xor(flagKarsi, flagLeft))

# cv2.imshow("ba", fark) 







