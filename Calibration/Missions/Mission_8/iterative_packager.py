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
        img_mapped = np.zeros((target_shape[0], target_shape[1], source_shape[2]), img.dtype) * np.nan
        img_mapped[imgP_mapped.T[index_mapped[si_wp_mapped], 1], imgP_mapped.T[index_mapped[si_wp_mapped], 0]] = img[index_mapped[si_wp_mapped]]

    else:
        img_mapped = np.zeros((target_shape[0], target_shape[1]), img.dtype) * np.nan
        img_mapped[imgP_mapped.T[index_mapped[si_wp_mapped], 1], imgP_mapped.T[index_mapped[si_wp_mapped], 0]] = img[index_mapped[si_wp_mapped]][:, 0]
    
    # print(np.sum(~np.isnan(img_mapped)))
    # print(index_mapped.shape[0])
    # print(np.sum(~np.isnan(img_mapped)) / index_mapped.shape[0])
    
    return img_mapped


def main():
    
    global filled_leftPM, filled_rightPM
    
    # read Package matrices first
    package_file = open("./thingsToSubmit/submission4/leftPackage/leftPackage.npy", "rb")
    leftPackageMatrix = np.load(package_file)
    package_file.close()

    package_file = open("./thingsToSubmit/submission4/rightPackage/rightPackage.npy", "rb")
    rightPackageMatrix = np.load(package_file)
    package_file.close()
    
    pt_cld_file = open("./thingsToSubmit/submission4/leftPackage/pt_cloud.npy", "rb")
    left_pt_cloud = np.load(pt_cld_file)
    pt_cld_file.close()
    
    pt_cld_file = open("./thingsToSubmit/submission4/rightPackage/pt_cloud.npy", "rb")
    right_pt_cloud = np.load(pt_cld_file)
    pt_cld_file.close()
    

    
    left_measured_flag = leftPackageMatrix[:,:,1][0:992, 0:992]
    left_measured = leftPackageMatrix[:,:,2][0:992, 0:992]
    left_measured[left_measured_flag == 0] = np.nan
    right_measured_flag = rightPackageMatrix[:,:,1][0:992, 0:992]
    right_measured = rightPackageMatrix[:,:,2][0:992, 0:992]
    right_measured[right_measured_flag == 0] = np.nan

    plt.figure()
    plt.imshow(left_measured[375:600, 890:990])

    
    ################## First iteration
    filled_leftPM, filled_rightPM = hoca.fill_depth_using_superpixels( (leftPackageMatrix, rightPackageMatrix), (left_pt_cloud, right_pt_cloud), 'plane'  )
    
    upsample_rate = 1
    KLeft = pack.KLeft * upsample_rate
    KLeft[2,2] = 1
    KRight = pack.KRight * upsample_rate
    KRight[2,2] = 1
    
    # filled_leftPM = cv2.resize(filled_leftPM, None, fx = upsample_rate, fy = upsample_rate, interpolation =  cv2.INTER_CUBIC )
    # filled_rightPM = cv2.resize(filled_rightPM, None, fx = upsample_rate, fy = upsample_rate, interpolation =  cv2.INTER_CUBIC )

    
    filled_right_mapped = map_img_2_img(filled_leftPM, KLeft, pack.T_LF2RF, pack.R_LF2RF, KRight, filled_rightPM.shape)
    filled_left_mapped = map_img_2_img(filled_rightPM, KRight, pack.T_RF2LF, pack.R_RF2LF, KLeft, filled_leftPM.shape)
    
    # filled_right_mapped = pack.downSampleImage(filled_right_mapped, upsample_rate)
    # filled_left_mapped = pack.downSampleImage(filled_left_mapped, upsample_rate)
    
    threshold = 30
    diff_on_right = np.abs( filled_rightPM - filled_right_mapped )
    consensus_on_right = diff_on_right < threshold
    filled_rightPM[~consensus_on_right] = np.nan
    filled_rightPM[right_measured_flag.nonzero()] = right_measured[right_measured_flag.nonzero()]
    
    diff_on_left = np.abs( filled_leftPM - filled_left_mapped )
    consensus_on_left = diff_on_left < threshold
    # filled_leftPM[~consensus_on_left] = np.nan
    # filled_leftPM[left_measured_flag.nonzero()] = left_measured[left_measured_flag.nonzero()]
    
    # plt.figure()
    # plt.title("Histogram of difference on left frame")
    # plt.hist(diff_on_left[np.isnan(left_measured)].flatten(), density=True, bins=range(0, int(np.max(diff_on_left[~np.isnan(diff_on_left)])), 5) )
    # num_new_points = np.sum(diff_on_left[np.isnan(left_measured)].flatten().shape)
    # plt.text(100, 0.04, "Number of new points = " + str(num_new_points) )
   
    # plt.figure()
    # plt.title("Histogram of difference on left frame including measured points")
    # plt.hist(diff_on_left.flatten(), density=True, bins=range(0, int(np.max(diff_on_left[~np.isnan(diff_on_left)])) ) )
   
    
    # plt.figure()
    # plt.imshow(filled_rightPM, cmap='binary')
    # plt.title("Right filled")
    # plt.figure()
    # plt.imshow(filled_leftPM, cmap='binary')
    # plt.title("Left filled")

    
    
    
    return # TODO
    ############ Second iteration
    leftPackageMatrix[0:992,0:992,2] = filled_leftPM
    leftPackageMatrix[0:992,0:992,1] = ~np.isnan(filled_leftPM)
    rightPackageMatrix[0:992,0:992,2] = filled_rightPM
    rightPackageMatrix[0:992,0:992,1] = ~np.isnan(filled_rightPM)

    filled_leftPM, filled_rightPM = hoca.fill_depth_using_superpixels( (leftPackageMatrix, rightPackageMatrix), (left_pt_cloud, right_pt_cloud) )
    
    # filled_leftPM = cv2.resize(filled_leftPM, None, fx = upsample_rate, fy = upsample_rate, interpolation =  cv2.INTER_CUBIC )
    # filled_rightPM = cv2.resize(filled_rightPM, None, fx = upsample_rate, fy = upsample_rate, interpolation =  cv2.INTER_CUBIC )
    
    filled_right_mapped = map_img_2_img(filled_leftPM, KLeft, pack.T_LF2RF, pack.R_LF2RF, KRight, filled_rightPM.shape)
    filled_left_mapped = map_img_2_img(filled_rightPM, KRight, pack.T_RF2LF, pack.R_RF2LF, KLeft, filled_leftPM.shape)
    
    # filled_right_mapped = pack.downSampleImage(filled_right_mapped, upsample_rate)
    # filled_left_mapped = pack.downSampleImage(filled_left_mapped, upsample_rate)
    
    print("#################################################################")
    diff_on_left = filled_leftPM.copy()
    diff_on_left[left_measured_flag != 0] = np.nan
    
    diff_on_right = filled_rightPM.copy()
    diff_on_right[right_measured_flag != 0] = np.nan
    
    diff_right_mapped = map_img_2_img(diff_on_left, KLeft, pack.T_LF2RF, pack.R_LF2RF, KRight, filled_rightPM.shape)
    diff_left_mapped = map_img_2_img(diff_on_right, KRight, pack.T_RF2LF, pack.R_RF2LF, KLeft, filled_leftPM.shape)
    
    threshold = 30
    consensus_on_right = np.abs( filled_rightPM - diff_right_mapped ) < threshold
    
    consensus_rate_A = np.sum(consensus_on_right) / np.sum(~np.isnan(diff_right_mapped))
    new_rate         = np.sum(np.isnan(filled_rightPM[~np.isnan(diff_right_mapped)])) / np.sum(~np.isnan(diff_right_mapped))
    consensus_rate_A /= 1 - new_rate
    print("Consensus rate = %" + str(100 * consensus_rate_A ))
    print("Consensus rate = %" + str(100 * new_rate ))
    
    print("#################################################################")
     
    threshold = 30
    consensus_on_right = np.abs( filled_rightPM - filled_right_mapped ) < threshold
    filled_rightPM[~consensus_on_right] = np.nan
    filled_rightPM[right_measured_flag.nonzero()] = right_measured[right_measured_flag.nonzero()]
    
    consensus_on_left = np.abs( filled_leftPM - filled_left_mapped ) < threshold
    filled_leftPM[~consensus_on_left] = np.nan
    filled_leftPM[left_measured_flag.nonzero()] = left_measured[left_measured_flag.nonzero()]

    """ 
    plt.figure(dpi=300, figsize=(1.8,1.8))
    plt.imshow(filled_right_mapped, cmap='jet')
    plt.title("Left mapped to Right")
    
    plt.figure(dpi=300, figsize=(1.8,1.8))
    plt.imshow(filled_left_mapped, cmap='jet')
    plt.title("Right mapped to Left")
    
    plt.figure(dpi=300, figsize=(1.8,1.8))
    plt.imshow(filled_rightPM, cmap='jet')
    plt.title("Right after filling")
    
    plt.figure(dpi=300, figsize=(1.8,1.8))
    plt.imshow(filled_leftPM, cmap='jet')
    plt.title("Left after filling")
    """
    
    
    
    """
        - Check for consensus between left and right depth images after filling
        - keep matching pixel values and discard the rest
        - restore original depth data by using old flag and old depthimages
        - update package matrix
        - repeat first, second and third steps
    """

if __name__ == "__main__":
    main()