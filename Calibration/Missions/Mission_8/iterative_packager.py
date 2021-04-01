# -*- coding: utf-8 -*-

import re
from ast import literal_eval
import numpy as np
import os
from scipy.spatial.transform import Rotation as rot
import cv2
import glob
from matplotlib import pyplot as plt

while(os.getcwd().split('/')[-1] != 'star'):
    os.chdir("../")
    
from Packager import packager as pack
from Superpixel import hoca

filled_leftPM, filled_rightPM = None, None

def map_img_2_img(depth_img, K1, T, R, K2, target_shape, img = None):
    
    if img is None:
        img = depth_img 
    else:
        assert depth_img.shape[0:2] == img.shape[0:2]
        
    K1_inv = np.linalg.inv(K1)
    
    # Generate img index points
    shape = depth_img.shape
    objp = np.ones((shape[0] * shape[1], 3), np.uint32)
    objp[:, :2] = np.mgrid[ 0:shape[1],0:shape[0]].T.reshape(-1, 2)
    objp = objp.T
    
    source_shape = img.shape
    img = img.reshape( (shape[0] * shape[1], -1) )

    # Generate 3D point cloud
    worldP = np.dot(K1_inv, objp)
    worldP = worldP.T
    worldP = worldP.reshape((shape[0], shape[1], 3) )
    
    if len(depth_img.shape) != 3:
        depth_img = np.expand_dims(depth_img, axis=2)
        
    worldP = worldP * depth_img * 1e-3

    if len(T.shape) == 1:
        T = np.expand_dims(T, 1)
    worldP_mapped = np.dot(R, worldP.reshape(shape[0]*shape[1], 3).T) + T
    
    
    imgP_mapped = np.dot(K2, worldP_mapped)
    imgP_mapped /= imgP_mapped[2,:]
    imgP_mapped = np.array(np.round(imgP_mapped), dtype=np.int32)
    
    flag_mapped = np.logical_and(np.logical_and(imgP_mapped[0,:] >= 0, imgP_mapped[0,:] < target_shape[1]), np.logical_and(imgP_mapped[1,:] >= 0, imgP_mapped[1,:] < target_shape[0] ) ) 
    index_mapped = flag_mapped.nonzero()[0]

    # sorted indeces for valid points
    si_wp_mapped = np.argsort(-1 * worldP_mapped.T[index_mapped ,2])
    
    # oc_rm_mapped = np.zeros((shape[0], shape[1]), np.int32) - 1
    # oc_rm_mapped[imgP_mapped[0:2, index_mapped[si_wp_mapped]][1], imgP_mapped[0:2, index_mapped[si_wp_mapped]][0]] = index_mapped[si_wp_mapped]
    # temp = np.zeros((flag_mapped.shape[0] + 1))
    # oc_rm_mapped = oc_rm_mapped.reshape((shape[0]*shape[1]) )
    # temp[oc_rm_mapped + 1] = 1
    # flag_mapped = np.logical_and(flag_mapped, temp[1:])
    
    if len(source_shape) == 3:
        img_mapped = np.zeros((target_shape[0], target_shape[1], source_shape[2]), img.dtype)
    else:
        img_mapped = np.zeros((target_shape[0], target_shape[1]), img.dtype)
        
    img_mapped[imgP_mapped.T[index_mapped[si_wp_mapped], 1], imgP_mapped.T[index_mapped[si_wp_mapped], 0]] = img[index_mapped[si_wp_mapped]][0]
    print(img[index_mapped[si_wp_mapped]][0].shape)
    print(img_mapped[imgP_mapped.T[index_mapped[si_wp_mapped], 0], imgP_mapped.T[index_mapped[si_wp_mapped], 1]].shape)
    
    return img_mapped
    if len(source_shape) == 3:
        return img_mapped.reshape((target_shape[0], target_shape[1], source_shape[2]))
    else:
        return img_mapped.reshape(target_shape)


def main():
    
    global filled_leftPM, filled_rightPM
    
    # read Package matrices first
    package_file = open("./thingsToSubmit/submission4/leftPackage/leftPackage.npy", "rb")
    leftPackageMatrix = np.load(package_file)
    package_file.close()

    package_file = open("./thingsToSubmit/submission4/rightPackage/rightPackage.npy", "rb")
    rightPackageMatrix = np.load(package_file)
    package_file.close()
    
    filled_leftPM, filled_rightPM = hoca.fill_depth_using_superpixels( (leftPackageMatrix, rightPackageMatrix) )
    
    filled_right_mapped = map_img_2_img(filled_leftPM, pack.KLeft, pack.T_LF2RF, pack.R_LF2RF, pack.KRight, filled_rightPM.shape)
    
    plt.figure(dpi=300)
    plt.imshow(filled_right_mapped, cmap='binary')
    plt.title("Left mapped to Right")
    
    plt.figure(dpi=300)
    plt.imshow(filled_rightPM, cmap='binary')
    plt.title("Right after filling")
    
    plt.figure(dpi=300)
    plt.imshow(filled_leftPM, cmap='binary')
    plt.title("Left after filling")
    
    """
        - Check for consensus between left and right depth images after filling
        - keep matching pixel values and discard the rest
        - restore original depth data by using old flag and old depthimages
        - update package matrix
        - repeat first, second and third steps
    """

if __name__ == "__main__":
    main()