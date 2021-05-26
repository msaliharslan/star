# -*- coding: utf-8 -*-


import re
from ast import literal_eval
import numpy as np
import os
from scipy.spatial.transform import Rotation as rot
import cv2
import glob
from PIL import Image
import threading
from tqdm import tqdm


if(os.getcwd().split('/')[-1] != 'star'):
    os.chdir("../")

def str_to_array(arr):
    """
    
    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.

    Returns
    -------
    arr : TYPE
        DESCRIPTION.

    """
    arr = re.sub(r"([^[])\s+([^]])", r"\1, \2", arr)
    arr = np.array(literal_eval(arr))
    return arr

def normalizeImageAndSave_toTemp(image, name):
    
    # Normalisation
    if(len(image.shape) == 3  and image.shape[-1] == 3): # normalise only rgb
        image = image / np.max(image) * (2**16-1)
        image = (image/256).astype(np.uint8) 
    else:
        image = image / np.max(image) * (2**16-1)
        image = image.astype(np.uint16)

    # numpy save
    # file = open( name +".npy", "wb")
    # np.save(file, image)
    # file.close()
    
    # OpenCV save
    cv2.imwrite(name +".png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9] )
    
    # Pillow save
    # toSave = Image.fromarray(image)
    # toSave.save(name +".png")


def undistortFisheyeImages(fisheyeImages, K, D):
    """

    Parameters
    ----------
    fisheyeImages : Image(numpy array)
    K : Camera Matrix numpy array
    D : Kannala-Brandt distortion parameters numpy array

    Returns
    -------
    undistortedImages : Undistorted images and new camera matrix


    """
    # https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/t265_stereo.py
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
    stereo_height_px = 1000          # 1000x1000 pixel stereo output
    stereo_focal_px = stereo_height_px/2 / np.tan(stereo_fov_rad/2)


    # The stereo algorithm needs max_disp extra pixels in order to produce valid
    # disparity on the desired output region. This changes the width, but the
    # center of projection should be on the center of the cropped image
    stereo_width_px = stereo_height_px
    stereo_cx = (stereo_height_px - 1)/2
    stereo_cy = (stereo_height_px - 1)/2

    # Construct the left and right projection matrices, the only difference is
    # that the right projection matrix should have a shift along the x axis of
    # baseline*focal_length
    P = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                       [0, stereo_focal_px, stereo_cy, 0],
                       [0,               0,         1, 0]])    
    
    undistortedImages = []
    
    # nk = K.copy()
    # nk[0,0]=K[0,0]/2
    # nk[1,1]=K[1,1]/2   
    
    for image in fisheyeImages:
            
        undistorted = cv2.fisheye.undistortImage(image, K=K, D=D, Knew=P, new_size=(int(stereo_height_px), int(stereo_width_px)))
        undistortedImages.append(undistorted)

    return undistortedImages, P[:,:-1]

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
            
            
def generateRgbPackageMatrix(mapMatrix, imgRgb, imgLeft, imgRight):
    
    rgbPackageMatrix = np.zeros((imgRgb.shape[0],imgRgb.shape[1], 9)) # r,g,b, fd, depth, fleft, leftIntensity, fright, rightIntensity
    
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

    return rgbPackageMatrix
    
def visualizeRgbPackageMatrix(rgbPackageMatrix, path): # r,g,b, fd, depth, fleft, leftIntensity, fright, rightIntensity

    global u_left, u_right, u_rgb
    
    # First original rgb image
    
    imgRgb = np.copy(rgbPackageMatrix[:,:,0:3])
    flagRgb = np.ones((imgRgb.shape[0], imgRgb.shape[1], 1), dtype=imgRgb.dtype)     

    imgRgb = downSampleImage(imgRgb, u_rgb) # cv2.resize(imgRgb, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )
    flagRgb = downSampleImage(flagRgb, u_rgb) #  cv2.resize(flagRgb, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )                

    normalizeImageAndSave_toTemp(imgRgb, path + "rgbPackage/packageRgb_rgb")
    normalizeImageAndSave_toTemp(flagRgb, path + "rgbPackage/packageRgb_rgbFlag")
    
    # Second corresponding depth image
    
    imgDepth = np.copy(rgbPackageMatrix[:,:,4])
    flagDepth = np.copy(rgbPackageMatrix[:,:, 3])
    
    imgDepth = downSampleImage(imgDepth, u_rgb) # cv2.resize(imgDepth, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )
    flagDepth = downSampleImage(flagDepth, u_rgb) # cv2.resize(flagDepth, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )
    
        
    normalizeImageAndSave_toTemp(imgDepth, path + "rgbPackage/packageRgb_depth")
    normalizeImageAndSave_toTemp(flagDepth, path + "rgbPackage/packageRgb_depthFlag")
    
    
    # Third corresponding left image
    
    imgLeft = rgbPackageMatrix[:,:,6]
    flagLeft = np.copy(rgbPackageMatrix[:,:,5])
    
    imgLeft = downSampleImage(imgLeft, u_rgb) # cv2.resize(imgLeft, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )
    flagLeft = downSampleImage(flagLeft, u_rgb) # cv2.resize(flagLeft, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )

    
    normalizeImageAndSave_toTemp(imgLeft, path + "rgbPackage/packageRgb_left")
    normalizeImageAndSave_toTemp(flagLeft, path + "rgbPackage/packageRgb_leftFlag")
    
    # Fourth corresponding right image
    
    imgRight = rgbPackageMatrix[:, :, 8]
    flagRight = np.copy(rgbPackageMatrix[:,:,7])
    
    imgRight = downSampleImage(imgRight, u_rgb) # cv2.resize(imgRight, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )
    flagRight = downSampleImage(flagRight, u_rgb) # cv2.resize(flagRight, None, fx = 1/u_rgb, fy = 1/u_rgb, interpolation =  cv2.INTER_CUBIC )
    
    
    normalizeImageAndSave_toTemp(imgRight, path + "rgbPackage/packageRgb_right")
    normalizeImageAndSave_toTemp(flagRight, path + "rgbPackage/packageRgb_rightFlag")
    
        
    
    
    
def generateLeftPackageMatrix(mapMatrix, imgRgb, imgLeft, imgRight):
    
    leftPackageMatrix = np.zeros((imgLeft.shape[0],imgLeft.shape[1], 9)) # leftIntensity, fd, depth, frgb, r, g, b, fright, rightIntensity
    
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

    return leftPackageMatrix


def visualizeLeftPackageMatrix(leftPackageMatrix, path):
    
    global u_left, u_right, u_rgb
    
    # leftIntensity, fd, depth, frgb, r, g, b, fright, rightIntensity
    
    # First original rgb image
    
    imgRgb = np.copy(leftPackageMatrix[:,:,4:7])
    flagRgb = np.copy(leftPackageMatrix[:,:,3])      
    
    imgRgb = downSampleImage(imgRgb, u_left) # cv2.resize(imgRgb, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )
    flagRgb = downSampleImage(flagRgb, u_left) # cv2.resize(flagRgb, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )           
        
    
    normalizeImageAndSave_toTemp(imgRgb, path + "leftPackage/packageLeft_rgb")
    normalizeImageAndSave_toTemp(flagRgb, path + "leftPackage/packageLeft_rgbFlag")
    
    # Second corresponding depth image
    
    imgDepth = np.copy(leftPackageMatrix[:,:,2])
    flagDepth = np.copy(leftPackageMatrix[:,:, 1])
    
    imgDepth = downSampleImage(imgDepth, u_left) # cv2.resize(imgDepth, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )
    flagDepth = downSampleImage(flagDepth, u_left) # cv2.resize(flagDepth, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )             
    
    normalizeImageAndSave_toTemp(imgDepth, path + "leftPackage/packageLeft_depth")
    normalizeImageAndSave_toTemp(flagDepth, path + "leftPackage/packageLeft_depthFlag")
    
    
    # Third corresponding left image
    
    imgLeft = leftPackageMatrix[:,:,0]
    flagLeft = np.ones((imgLeft.shape[0], imgLeft.shape[1], 1), dtype=imgLeft.dtype)  
    
    imgLeft = downSampleImage(imgLeft, u_left) # cv2.resize(imgLeft, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )
    flagLeft = downSampleImage(flagLeft, u_left) # cv2.resize(flagLeft, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )   
                   

    normalizeImageAndSave_toTemp(imgLeft, path + "leftPackage/packageLeft_left")
    normalizeImageAndSave_toTemp(flagLeft, path + "leftPackage/packageLeft_leftFlag")
    
    # Fourth corresponding right image
    
    imgRight = leftPackageMatrix[:, :, 8]
    flagRight = np.copy(leftPackageMatrix[:,:,7])
    
    imgRight = downSampleImage(imgRight, u_left) # cv2.resize(imgRight, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )
    flagRight = downSampleImage(flagRight, u_left) # cv2.resize(flagRight, None, fx = 1/u_left, fy = 1/u_left, interpolation =  cv2.INTER_CUBIC )      
    
    normalizeImageAndSave_toTemp(imgRight, path + "leftPackage/packageLeft_right")
    normalizeImageAndSave_toTemp(flagRight, path + "leftPackage/packageLeft_rightFlag")
    



def generateRightPackageMatrix(mapMatrix, imgRgb, imgLeft, imgRight):
    
    rightPackageMatrix = np.zeros((imgRight.shape[0],imgRight.shape[1], 9)) # rightIntensity, fd, depth, frgb, r, g, b, fleft, leftIntensity
    
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
 
    return rightPackageMatrix
    
def visualizeRightPackageMatrix(rightPackageMatrix, path):
    
    global u_left, u_right, u_rgb
    
     # rightIntensity, fd, depth, frgb, r, g, b, fleft, leftIntensity
    
    # First original rgb image
    
    imgRgb = np.copy(rightPackageMatrix[:,:,4:7])
    flagRgb = np.copy(rightPackageMatrix[:,:,3])            

    imgRgb = downSampleImage(imgRgb, u_right) # cv2.resize(imgRgb, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )
    flagRgb = downSampleImage(flagRgb, u_right) # cv2.resize(flagRgb, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )          
    
    normalizeImageAndSave_toTemp(imgRgb, path + "rightPackage/packageRight_rgb")
    normalizeImageAndSave_toTemp(flagRgb, path + "rightPackage/packageRight_rgbFlag")
    
    # Second corresponding depth image
    
    imgDepth = np.copy(rightPackageMatrix[:,:,2])
    flagDepth = np.copy(rightPackageMatrix[:,:, 1])
    
    imgDepth = downSampleImage(imgDepth, u_right) # cv2.resize(imgDepth, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )
    flagDepth = downSampleImage(flagDepth, u_right) # cv2.resize(flagDepth, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )              
    
    normalizeImageAndSave_toTemp(imgDepth, path + "rightPackage/packageRight_depth")
    normalizeImageAndSave_toTemp(flagDepth, path + "rightPackage/packageRight_depthFlag")
    
    
    # Third corresponding left image
    
    imgLeft = rightPackageMatrix[:,:,8]
    flagLeft = rightPackageMatrix[:,:,7] 
    
    imgLeft = downSampleImage(imgLeft, u_right) # cv2.resize(imgLeft, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )
    flagLeft = downSampleImage(flagLeft, u_right) # cv2.resize(flagLeft, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )     

    normalizeImageAndSave_toTemp(imgLeft, path + "rightPackage/packageRight_left")
    normalizeImageAndSave_toTemp(flagLeft, path + "rightPackage/packageRight_leftFlag")
    
    # Fourth corresponding right image
    
    imgRight = rightPackageMatrix[:, :, 0]
    flagRight = np.ones((imgRight.shape[0], imgRight.shape[1], 1), dtype=imgRight.dtype)
    
    imgRight = downSampleImage(imgRight, u_right) # cv2.resize(imgRight, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )
    flagRight = downSampleImage(flagRight, u_right) # cv2.resize(flagRight, None, fx = 1/u_right, fy = 1/u_right, interpolation =  cv2.INTER_CUBIC )         
    
    normalizeImageAndSave_toTemp(imgRight, path + "rightPackage/packageRight_right")
    normalizeImageAndSave_toTemp(flagRight, path + "rightPackage/packageRight_rightFlag")
    
    
    
    
def generateDepthPackageMatrix(mapMatrix, imgRgb, imgLeft, imgRight, imgDep):
   
    depthPackageMatrix = np.zeros((imgDep.shape[0],imgDep.shape[1], 9))  # depth, frgb, r,g,b, fLeft, leftIntensity, fRight, rightIntensity
                                                                           
    depthPackageMatrix[:,:,0] = imgDep[:,:,0]
    
    depthPackageMatrix = depthPackageMatrix.reshape((imgDep.shape[0] * imgDep.shape[1], 9))
  
    depthPackageMatrix[:,1:] = np.concatenate((np.expand_dims(mapMatrix[:,0] ,1), imgRgb[mapMatrix[:,2], mapMatrix[:,1]], np.expand_dims(mapMatrix[:,4], 1), np.expand_dims(imgLeft[mapMatrix[:,6], mapMatrix[:,5]], 1), np.expand_dims(mapMatrix[:,8], 1), np.expand_dims(imgRight[mapMatrix[:,10], mapMatrix[:,9]],1)), 1)  
    depthPackageMatrix = depthPackageMatrix.reshape((imgDep.shape[0], imgDep.shape[1], 9))

  
    return depthPackageMatrix    


def visualizeDepthPackageMatrix(depthPackageMatrix, path):
    
    # depth, frgb, r,g,b, fLeft, leftIntensity, fRight, rightIntensity    

    # First original rgb image
    
    imgRgb = np.copy(depthPackageMatrix[:,:,2:5])
    flagRgb = np.copy(depthPackageMatrix[:,:,1])                
    
    normalizeImageAndSave_toTemp(imgRgb, path + "depthPackage/packageDepth_rgb")
    normalizeImageAndSave_toTemp(flagRgb, path + "depthPackage/packageDepth_rgbFlag")
    
    # Second corresponding depth image
    
    imgDepth = np.copy(depthPackageMatrix[:,:,0])
    flagDepth = imgDepth != 0
    
    normalizeImageAndSave_toTemp(imgDepth, path + "depthPackage/packageDepth_depth")
    normalizeImageAndSave_toTemp(flagDepth, path + "depthPackage/packageDepth_depthFlag")
    
    
    # Third corresponding left image
    
    imgLeft = depthPackageMatrix[:,:,6]
    flagLeft = depthPackageMatrix[:,:,5] 

    normalizeImageAndSave_toTemp(imgLeft, path + "depthPackage/packageDepth_left")
    normalizeImageAndSave_toTemp(flagLeft, path + "depthPackage/packageDepth_leftFlag")
    
    # Fourth corresponding right image
    
    imgRight = depthPackageMatrix[:, :, 8]
    flagRight = depthPackageMatrix[:, :, 7]
    
    normalizeImageAndSave_toTemp(imgRight, path + "depthPackage/packageDepth_right")
    normalizeImageAndSave_toTemp(flagRight, path + "depthPackage/packageDepth_rightFlag")
    
    
    
   
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
        
    return mapMatrix


extrinsics_file = glob.glob("Calibration/Final/*")
extrinsics_file.sort()
extrinsics_file = open(extrinsics_file[-1], "r")
extrinsics_all = extrinsics_file.read()
extrinsics_file.close()
del extrinsics_file

entries = re.split(":+", extrinsics_all)
entries = [entry.split('*') for entry in entries]

T_LF2C = entries[1][0]
T_LF2C = str_to_array(T_LF2C)

R_LF2C = entries[2][0]
R_LF2C = str_to_array(R_LF2C)

F_LF2C = entries[3][0]
F_LF2C = str_to_array(F_LF2C)

T_RF2C = entries[4][0]
T_RF2C = str_to_array(T_RF2C)

R_RF2C = entries[5][0]
R_RF2C = str_to_array(R_RF2C)

F_RF2C = entries[6][0]
F_RF2C = str_to_array(F_RF2C)

T_LF2RF = entries[7][0]
T_LF2RF = str_to_array(T_LF2RF)

R_LF2RF = entries[8][0]
R_LF2RF = str_to_array(R_LF2RF)

F_LF2RF = entries[9][0]
F_LF2RF = str_to_array(F_LF2RF)

R_C2LF = np.linalg.inv(R_LF2C)
T_C2LF = -np.dot(R_C2LF, T_LF2C)

R_C2RF = np.linalg.inv(R_RF2C)
T_C2RF = -np.dot(R_C2RF, T_RF2C)

R_RF2LF = np.linalg.inv(R_LF2RF)
T_RF2LF = -np.dot(R_RF2LF, T_LF2RF)


T_D2C = np.array([0.014851336367428, 0.0004623454588, 0.000593442469835]) # from depth to RGB(infrared 1)

R_D2C = np.array([-0.003368464997038, -0.000677574775182, -0.006368808448315, -0.999973833560944]) # from depth to RGB(infrared 1)
R_D2C = rot.from_quat(R_D2C).as_matrix()


KDep = np.array([[634.013671875, 0.0, 635.5408325195312], [0.0, 634.013671875, 351.9051208496094], [0.0, 0.0, 1.0]]) #1280 x 720
# KDep = np.array([[420.0340576171875, 0.0, 421.04580154418943], [0.0, 422.67578125, 234.6034138997396], [0.0, 0.0, 1.0]]) # 848 480
KDep_inv = np.linalg.inv(KDep)
KRgb = np.array([[611.6753646850586, 0.0, 423.94055938720703], [0.0, 615.7430826822916, 246.8198445638021], [0.0, 0.0, 1.0]]) # 848 x 480

# FISHEYE
KL = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
DLeft = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])

KR = np.array([[284.1828918457031, 0.0, 427.9779052734375], [0.0, 285.0440979003906, 399.5506896972656], [0.0, 0.0, 1.0]])
DRight = np.array([0.0009760634857229888, 0.030147459357976913, -0.02769969031214714, 0.0031066760420799255])

KLeft = np.array([[500. ,   0. , 499.5],
                  [  0. , 500. , 499.5],
                  [  0. ,   0. ,   1. ]])

KRight = np.array([[500. ,   0. , 499.5],
                   [  0. , 500. , 499.5],
                   [  0.,    0. ,   1. ]])

u_left = 1
u_right = 1
u_rgb = 1
    
def main():
    
    global KLeft, KRight, KRgb, u_left, u_right, u_rgb
    u_left = 3
    u_right = 3
    u_rgb = 4
    
    KLeft*= u_left
    KLeft[2,2] = 1
    KRight *= u_right
    KRight[2,2] = 1
    KRgb *= u_rgb
    KRgb[2,2] = 1
    
    package_list = open("Packager/package_list.txt", "r")
    for line in tqdm(package_list):
        scene_name, index = line.split(' ')
        folderName = glob.glob("Records/Scene/" + scene_name + "/*")
        folderName.sort()
        folderName = folderName[-int(index)]
        
        try:
            os.mkdir("Packager/Packages/" + scene_name)
        except:
            print("Scene folder exists")
        
        fileNamesRgb = sorted(glob.glob(folderName + "/rgb/*"), key = lambda fileName : fileName.split('_')[-1]) 
        fileNamesLeft = sorted(glob.glob(folderName + "/leftFisheye/*"), key = lambda fileName : fileName.split('_')[-1] )
        fileNamesRight = sorted(glob.glob(folderName + "/rightFisheye/*"), key = lambda fileName : fileName.split('_')[-1]) 
        fileNamesDepth = sorted(glob.glob(folderName + "/depth/*"), key = lambda fileName : fileName.split('_')[-1]) 
        
        try:
            os.mkdir("Packager/Packages/" + scene_name + '/' + fileNamesDepth[0].split('/')[3])
        except:
            print("Date folder exists")
        
        rgb_start = 0
        left_start = 0
        right_start = 0
        for depthName in tqdm(fileNamesDepth):
            depth_timestamp = int(depthName.split(".")[0].split("_")[-1])
            recording_date = depthName.split('/')[3]
            frame_num_depth = depthName.split('/')[-1].split('_')[1]
            path = "Packager/Packages/" + scene_name + '/' + recording_date + '/' + frame_num_depth + '/'
            
            try:
                os.mkdir("Packager/Packages/" + scene_name + '/' + recording_date + '/' + frame_num_depth)
                os.mkdir(path +  "/rgbPackage")
                os.mkdir(path +  "/leftPackage")
                os.mkdir(path +  "/rightPackage")
                os.mkdir(path +  "/depthPackage")
            except:
                print("At least one of the package folder exists")
            
            min_diff = 1e20
            for i in range(rgb_start, len(fileNamesRgb)):
                rgb_timestamp = int(fileNamesRgb[i].split(".")[0].split("_")[-1])
                if(abs(rgb_timestamp - depth_timestamp) < min_diff):
                    min_diff = abs(rgb_timestamp - depth_timestamp)
                else:
                    rgb_start = i
                    break
    
            min_diff = 1e20
            for i in range(left_start, len(fileNamesLeft)):
                left_timestamp = int(fileNamesLeft[i].split(".")[0].split("_")[-1])
                if(abs(left_timestamp - depth_timestamp) < min_diff):
                    min_diff = abs(left_timestamp - depth_timestamp)
                else:
                    left_start = i
                    break
                    
            min_diff = 1e20
            for i in range(right_start, len(fileNamesRight)):
                right_timestamp = int(fileNamesRight[i].split(".")[0].split("_")[-1])
                if(abs(right_timestamp - depth_timestamp) < min_diff):
                    min_diff = abs(right_timestamp - depth_timestamp)
                else:
                    right_start = i
                    break
        
            img_depth = cv2.imread(depthName, cv2.IMREAD_UNCHANGED)
            img_rgb = cv2.imread(fileNamesRgb[rgb_start - 1], cv2.IMREAD_UNCHANGED)
            img_left = cv2.imread(fileNamesLeft[left_start - 1], cv2.IMREAD_UNCHANGED)
            img_right = cv2.imread(fileNamesRight[right_start - 1], cv2.IMREAD_UNCHANGED)
            
            # Undistort
            img_left, _ = undistortFisheyeImages([img_left], KL, DLeft)
            img_right, _ = undistortFisheyeImages([img_right], KR, DRight)
            
            
            img_left = cv2.resize(img_left[0], None, fx = u_left, fy = u_left, interpolation =  cv2.INTER_CUBIC )
            img_right = cv2.resize(img_right[0], None, fx = u_right, fy = u_right, interpolation =  cv2.INTER_CUBIC )
            img_rgb = cv2.resize(img_rgb, None, fx = u_rgb, fy = u_rgb, interpolation =  cv2.INTER_CUBIC )
        
            shape = img_depth.shape
            objp = np.ones((shape[0] * shape[1], 3), np.uint32)
            objp[:, :2] = np.mgrid[ 0:shape[1],0:shape[0]].T.reshape(-1, 2)
            objp = objp.T
            worldP_depth = np.dot(KDep_inv, objp)
            worldP_depth = worldP_depth.T
            worldP_depth = worldP_depth.reshape((shape[0], shape[1], 3) )
            img_depth = np.expand_dims(img_depth, axis=2)
            worldP_depth = worldP_depth * img_depth * 1e-3
            
            worldP_color = np.dot(R_D2C, worldP_depth.reshape(shape[0]*shape[1], 3).T) + np.expand_dims(T_D2C, 1)
            worldP_left = np.dot(R_C2LF, worldP_color) + T_C2LF
            worldP_right = np.dot(R_C2RF, worldP_color) + T_C2RF
            
            
            imgP_color = np.dot(KRgb, worldP_color)
            imgP_color /= imgP_color[2,:]
            imgP_color = np.array(np.round(imgP_color), dtype=np.int32)
            
            imgP_left = np.dot(KLeft, worldP_left)
            imgP_left /= imgP_left[2,:]
            imgP_left = np.array(np.round(imgP_left), dtype=np.int32)
            
            imgP_right = np.dot(KRight, worldP_right)
            imgP_right /= imgP_right[2,:]
            imgP_right = np.array(np.round(imgP_right), dtype=np.int32)
        
            mapMatrix = generateMapMatrix(worldP_color, imgP_color, img_rgb.shape, worldP_left, imgP_left, img_left.shape, worldP_right, imgP_right, img_right.shape)
            
            rgbPackageMatrix = generateRgbPackageMatrix(mapMatrix, img_rgb, img_left, img_right)
            visualizeRgbPackageMatrix(rgbPackageMatrix, path )
            
            leftPackageMatrix = generateLeftPackageMatrix(mapMatrix, img_rgb, img_left, img_right)
            visualizeLeftPackageMatrix(leftPackageMatrix, path)
        
            rightPackageMatrix = generateRightPackageMatrix(mapMatrix, img_rgb, img_left, img_right)
            visualizeRightPackageMatrix(rightPackageMatrix, path)
        
            depthPackageMatrix = generateDepthPackageMatrix(mapMatrix, img_rgb, img_left, img_right, img_depth)
            visualizeDepthPackageMatrix(depthPackageMatrix, path)
    
    
    package_list.close()

if __name__ == "__main__":
    main()









































