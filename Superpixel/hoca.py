#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 00:09:48 2021

@author: salih
"""
import numpy as np
from Superpixel.Superpixel import *
from skimage.color import rgb2lab, lab2rgb

def fill_depth_using_superpixels(packages): # (leftPackage, RightPackage)
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
            # fill the holes with SP mean
            output = sp.fill_mean_image()
            
            output[~np.isnan(img_set)] = img_set[~np.isnan(img_set)]
            
            # convert color channels to rgb
            output[:,:,2:5] = lab2rgb(output[:,:,2:5])
            depthLeft = output[:,:,5]

        else: #rightPackage
            
        # rightIntensity, fd, depth, frgb, r, g, b, fleft, leftIntensity
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
            # fill the holes with SP mean
            output = sp.fill_mean_image()
            
            output[~np.isnan(img_set)] = img_set[~np.isnan(img_set)]
            
            # convert color channels to rgb
            output[:,:,2:5] = lab2rgb(output[:,:,2:5])
            depthRight = output[:,:,5]

    return depthLeft, depthRight


























