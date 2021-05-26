#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 00:09:48 2021

@author: salih
"""
import numpy as np
from Superpixel.Superpixel import *
from skimage.color import rgb2lab, lab2rgb
from matplotlib import pyplot as plt
import cv2

KL = np.array([[500. ,   0. , 499.5],
       [  0. , 500. , 499.5],
       [  0. ,   0. ,   1. ]])

KR = np.array([[500. ,   0. , 499.5],
 [  0. , 500. , 499.5],
 [  0.,    0. ,   1. ]])

def fill_depth_using_superpixels(packages, pt_clouds, fill_method): # (leftPackage, RightPackage)
    """
    Input packages: (leftPackage, RightPackage) must be given
    """
    
    sp = Superpixel(tiling = 'iSQUARE', exp_area = 256.0, spectral_cost = 'Bayesian', spatial_cost = 'Bayesian', compactness = 8)
    
    images = ['left', 'right', 'rgb', 'depth']
    num_ch = np.array([1, 1, 3, 1])
    
    depthLeft = None
    depthRight = None
    # prepare images
    for i, package in enumerate(packages):


        h = package.shape[0] // 16 * 16
        w = package.shape[1] // 16 * 16
        
        img_set = np.zeros([h, w, 6], dtype=np.float64)
        
        package = package[ 0:h, 0:w, : ]
        
         # leftIntensity, fd, depth, frgb, b, g, r, fright, rightIntensity
        if i == 0: #leftPackage
                
            left_indices_img = package[:,:,-1][0:992, 0:992]
            pts = pt_clouds[i]
            
            img_set[:,:,0] = package[:,:,0] #left
            
            tmp = package[:,:,8] # right
            flag = package[:,:,7]
            tmp[flag == 0] = np.nan
            img_set[:,:,1] = tmp
            
            tmp = package[:,:,4:7]  # rgb
            tmp[:,:,0], tmp[:,:,2] = tmp[:,:,2], tmp[:,:,0]
            tmp = rgb2lab(tmp)
            flag = package[:,:,3]
            tmp[flag == 0, :] = np.nan
            img_set[:,:,2:5] = tmp
            
            tmp = package[:,:,2] # depth
            flag = package[:,:,1]
            tmp[flag == 0] = np.nan
            img_set[:,:,5] = tmp
        
            sp.extract_superpixels(img_set, main_channel = i)
            
            if(fill_method == 'mean'):
            # fill the holes with SP mean
                output = sp.fill_mean_image()
                            
                output[~np.isnan(img_set)] = img_set[~np.isnan(img_set)]
                
                # convert color channels to rgb
                # output[:,:,2:5] = lab2rgb(output[:,:,2:5])
                depthLeft = output[:,:,5]
            elif(fill_method == 'plane'):
                depthLeft = sp.fill_plane_fitted_superpixel(pts, left_indices_img, KL)
                # median = np.median(depthLeft)
                # depthLeft[depthLeft > 18 * median] = 0
                plt.figure()
                plt.imshow(depthLeft, cmap='jet')
                plt.title("Planes on left image")
                cv2.imwrite("random_left.png", (depthLeft / np.max(depthLeft[~np.isnan(depthLeft)]) * 65535).astype(np.uint16) )
                # plt.figure()
                # depthLeft[~np.isnan(img_set)[:,:,-1]] = img_set[:,:,-1][~np.isnan(img_set)[:,:,-1]]
                # plt.imshow(depthLeft, cmap='binary')
                # plt.title("Depth filled with planes")
                # plt.figure()
                # plt.imshow(img_set[:,:,-1], cmap='binary')
                # plt.show()

            
        else: #rightPackage
            return # TODO remove
            right_indices_img = package[:,:,-1][0:992, 0:992]    
            pts = pt_clouds[i]
            
            img_set[:,:,1] = package[:,:,0] #right
            
            tmp = package[:,:,8] # left
            flag = package[:,:,7]
            tmp[flag == 0] = np.nan
            img_set[:,:,0] = tmp
            
            tmp = package[:,:,4:7]  # rgb
            tmp[:,:,0], tmp[:,:,2] = tmp[:,:,2], tmp[:,:,0]
            tmp = rgb2lab(tmp)
            flag = package[:,:,3]
            tmp[flag == 0, :] = np.nan
            img_set[:,:,2:5] = tmp
            
            tmp = package[:,:,2] # depth
            flag = package[:,:,1]
            tmp[flag == 0] = np.nan
            img_set[:,:,5] = tmp
            
            sp.extract_superpixels(img_set, main_channel = i)

            if(fill_method == 'mean'):
            # fill the holes with SP mean
                output = sp.fill_mean_image()
                            
                output[~np.isnan(img_set)] = img_set[~np.isnan(img_set)]
                
                # convert color channels to rgb
                # output[:,:,2:5] = lab2rgb(output[:,:,2:5])
                depthRight = output[:,:,5]
            elif(fill_method == 'plane'):
                depthRight = sp.fill_plane_fitted_superpixel(pts, right_indices_img, KR)
                median = np.median(depthRight)
                depthRight[depthRight > 18 * median] = 0
                plt.figure()
                plt.imshow(depthRight, cmap='jet')
                plt.title("Planes on right image")
                cv2.imwrite("random_right.png", depthRight)
                # plt.figure()
                # depthRight[~np.isnan(img_set)[:,:,-1]] = img_set[:,:,-1][~np.isnan(img_set)[:,:,-1]]
                # plt.imshow(depthRight, cmap='binary')
                # plt.title("Depth filled with planes")
                # plt.figure()
                # plt.imshow(img_set[:,:,-1], cmap='binary')
    
                plt.show()

    return depthLeft, depthRight


























