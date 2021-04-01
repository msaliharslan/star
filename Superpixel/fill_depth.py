#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 15:01:59 2021

@author: kutalmisince
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
from Superpixel import Superpixel
from mpldatacursor import datacursor

main_directory = '/home/salih/Documents/realsense-workspace/star/thingsToSubmit/submission4'

package = 'packageLeft'; main_channel = 0
#package = 'packageRight'; main_channel = 1
#package = 'packageRgb'; main_channel = 2

images = ['left', 'right', 'rgb', 'depth']
num_ch = np.array([1, 1, 3, 1])

# prepare images
img_set = np.array([])

for i, image in enumerate(images):
    input_img = Image.open(main_directory + '/' + package + '/' + package + '_' + image + '.png')
    input_flag = Image.open(main_directory + '/' + package + '/' + package + '_' + image + 'Flag.png')
    
    w = input_img.size[0] // 16 * 16
    h = input_img.size[1] // 16 * 16
    
    if img_set.shape[0] == 0:
        img_set = np.zeros([h, w, 6])
    
    flag = np.asarray(input_flag.crop((0, 0, w, h)))
    
    if image == 'rgb':        
        img = rgb2lab(np.asarray(input_img.crop((0, 0, w, h))))        
    else:
        img = np.expand_dims(np.asarray(input_img.crop((0, 0, w, h))), 2).copy().astype(float)
    
    # fill unavailable measurements with NaN    
    img[flag == 0, :] = np.nan
    
    # add to image set          
    img_set[:,:,np.sum(num_ch[0:i]):np.sum(num_ch[0:i+1])] = img.copy()
        
# extract superpixels    
sp = Superpixel(tiling = 'iSQUARE', exp_area = 256.0, spectral_cost = 'Bayesian', spatial_cost = 'Bayesian', compactness = 8)

sp.extract_superpixels(img_set, img_disp = img_set[:,:,main_channel], main_channel = main_channel)
#img display olmazsa figure oluşturmaz

# fill the holes with SP mean
output = sp.fill_mean_image()

output[~np.isnan(img_set)] = img_set[~np.isnan(img_set)]

# convert color channels to rgb
output[:,:,2:5] = lab2rgb(output[:,:,2:5])


for i, image in enumerate(images):
    print(image)
    plt.figure(dpi=300)
    plt.axis("off")
    plt.imshow(output[:,:,np.sum(num_ch[0:i]):np.sum(num_ch[0:i+1])])
    plt.title(image)
    datacursor(display='single')


plt.figure(dpi=300)
plt.axis("off")
plt.imshow(128+10*(output[:,:,0] - output[:,:,1]))
plt.title('depth before fill')
    


