#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 00:14:03 2021

@author: salih
"""
import numpy as np
import os
from matplotlib import pyplot as plt

while(os.getcwd().split('/')[-1] != 'star'):
    os.chdir("../")
    
from Calibration.Missions.Mission_8.iterative_packager import map_img_2_img
from Packager import packager as pack

def diamond(n):
    a = np.arange(n)
    b = np.minimum(a,a[::-1])
    return (b[:,None]+b)>=(n-1)//2


def calculate_oh_my_goodness(src_dep, prj_dep, src_inten, prj_inten):
    """
    occupied by a point nearer than candidate depth, depth is valid but occluded use a high error value like 255
    occupied by a point further than candidate depth, depth is invalid
    not occupied, take the intensity difference

    """

    kotu = 255
    diff_threshold = src_dep * 0.04 # mm
    inten_threshold = 5
    if np.isnan(src_dep):
        if(np.abs(src_inten - prj_inten) < inten_threshold):
            return np.abs(src_inten - prj_inten) / inten_threshold * 255
        return kotu
    
    if(src_dep - prj_dep > diff_threshold):
        return np.inf
    elif (prj_dep - src_dep > diff_threshold):
        return kotu
    else:
        return np.abs(src_dep - prj_dep) / diff_threshold * 255  
    
def calculate_neighbourhood_depth(nghbrhood_depths):
    """
    array bekliyor

    """
    valids = nghbrhood_depths[~np.isnan(nghbrhood_depths)]
    
    if(np.sum(valids.shape) <= 0 ):
        return np.nan
    
    return np.mean(valids)
    
def calculate_neighbourhood_intensity(nghbrhood): # actually np.mean ;)
    """
    array bekliyor

    """
    return np.mean(nghbrhood)

def find_target_neighbourhood(point, target_array):
    
    point = point[:,0]
    y_low = point[0] - 1
    y_high = point[0] + 2
    x_low = point[1] - 1
    x_high = point[1] + 2
    
    x_low = np.max((x_low, 0))
    y_low = np.max((y_low, 0))
    x_high = np.min((x_high, target_array.shape[1]))
    y_high = np.min((y_high, target_array.shape[0]))
    
    N = np.full( (y_high - y_low, x_high - x_low), True )
    return target_array[y_low: y_high, x_low: x_high][N]

    
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


left = leftPackageMatrix[:,:,0][0:992, 0:992]
left_measured_flag = leftPackageMatrix[:,:,1][0:992, 0:992]
left_measured = leftPackageMatrix[:,:,2][0:992, 0:992]
left_measured[left_measured_flag == 0] = np.nan

rigt = rightPackageMatrix[:,:,0][0:992, 0:992]
right_measured_flag = rightPackageMatrix[:,:,1][0:992, 0:992]
right_measured = rightPackageMatrix[:,:,2][0:992, 0:992]
right_measured[right_measured_flag == 0] = np.nan

upsample_rate = 1
KLeft = pack.KLeft * upsample_rate
KLeft[2,2] = 1
KRight = pack.KRight * upsample_rate
KRight[2,2] = 1
KL_inv = np.linalg.inv(KLeft)

# Left mapped to right section
# left_mapped_2_right = map_img_2_img(left_measured, KLeft, pack.T_LF2RF, pack.R_LF2RF, KRight, right_measured.shape)
E_left = np.full(left_measured.shape, np.inf)
E_left[left_measured_flag == 0] = np.nan

width = left.shape[1]
height = left.shape[0]

for i in range(width):
    for j in range(height):
        if(np.isnan(left_measured[i, j])):
            continue
        point = np.array([[i , j, 1]]).T
        point = np.dot(KL_inv, point)
        point *= left_measured[i, j]
        point = np.dot(pack.R_LF2RF, point) + pack.T_LF2RF
        point = np.dot(KRight, point)
        point /= point[2]
        point = np.round(point).astype(np.int32)
        if not np.logical_and(np.logical_and(point[0] >= 0, point[0] < right_measured.shape[1]), np.logical_and(point[1] >= 0, point[1] < right_measured.shape[0] ) ): 
            continue    
        N = find_target_neighbourhood(point, right_measured)
        val = calculate_neighbourhood_depth(N)
        inten = calculate_neighbourhood_intensity(N)
        goodness = calculate_oh_my_goodness(val, left_measured[i ,j], inten, left[i,j])
        E_left[i, j] = goodness


plt.imshow(E_left, cmap='winter')
"""

LD, RD
PRJ = LD -> R
PRJ - RD ?
    >threshold: invalid
    <-threshold: occlude
    not occupied: compare intensity
    within threshold: valid, return qulity between 0-255

"""







































