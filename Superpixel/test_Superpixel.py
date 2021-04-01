#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:40:25 2021

@author: kutalmisince
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from PIL import Image
import scipy.io as spio
from Superpixel import Superpixel

inp_img = Image.open('img1.bmp')
img_disp = np.asarray(inp_img.crop((0, 0, inp_img.size[0] // 16 * 16, inp_img.size[1] // 16 * 16)))
img_proc = rgb2lab(img_disp)

# initial tiling and loss function alternatives
tiling = ['SQUARE', 'iSQUARE', 'HEX']
spectral = ['Bayesian', 'N']
spatial  = ['Bayesian', 'N']

'''
# test initial tiling
for t in tiling:
    print('running ' + t)
    sp = Superpixel(tiling = t, spectral_cost = 'Bayesian', spatial_cost = 'Bayesian', compactness= 16.0)
    sp.extract_superpixels(img_proc)
                
    out_img = sp.draw_boundaries(img_disp)
    
    plt.figure(dpi=300)
    plt.axis('off')
    plt.imshow(out_img)
    plt.title('final')
    plt.show()
    sp.img_disp = -1
    outDict = {'sp': sp}
                
    spio.savemat('lena_tiling_' + t + '.mat', outDict)

'''
# test initial tiling and loss function alternative together
for t in tiling:
    for spat in spatial:
        for spec in spectral:
            
            print('running ' + t + ', ' + spat + ', ' + spec)
            sp = Superpixel(tiling = t, spectral_cost = spec, spatial_cost = spat, compactness = 8)
            
            sp.extract_superpixels(img_proc)
            
            out_img = sp.draw_boundaries(img_disp, color = [0, 255, 0])
    
            plt.figure(dpi=300)
            plt.axis('off')
            plt.imshow(out_img)
            plt.title('spectral: ' + spec + ', spatial: ' + spat)
            plt.show()
            
            Image.fromarray(out_img).save('img_output.png')
            
            sp.img_disp = -1
            
            outDict = {'sp': sp}
            
            spio.savemat('img_' + t[0] + spec[0] + spat[0] + '.mat', outDict)
            
            
