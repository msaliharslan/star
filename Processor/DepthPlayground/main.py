import sys
import argparse
import os
import numpy as np
import cv2
import struct

os.chdir("..")
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import fetcher

fetcher.insertFocusDataPath("pthdeneme")
fetcher.fetchDepthDataD435i()

targetFrameIndex = 10

data_8bit_seperated = np.array((fetcher.depth_d435i[targetFrameIndex][1:-1]).split(", "),dtype=np.uint8).reshape((720, 2560))
data = []
for  row in data_8bit_seperated:
    tmp_row=[]
    for i in range(0, row.size, 2):
        tmp_row.append( row[i] + (row[i+1]<<8) )
    data.append(tmp_row)
    
data = np.array(data, dtype=np.uint16)
src = cv2.resize(data,(680, 360))
src = (src/256).astype(np.uint8)

dst	= cv2.applyColorMap( src, cv2.COLORMAP_RAINBOW)




