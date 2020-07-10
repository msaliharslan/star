# -*- coding: utf-8 -*-

import numpy as np
import cv2


def saveImages(folder, images, header = "" ):
    
    for i,image in enumerate(images):
        
        cv2.imwrite(folder + header + "_" + str(i) + ".png", image)