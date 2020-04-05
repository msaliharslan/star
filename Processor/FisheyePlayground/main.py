import sys
import argparse
import os
import numpy as np
os.chdir("..")
sys.path.append(os.getcwd())

import matplotlib.pyplot as plot

%matplotlib inline

import fetcher

fetcher.insertFocusDataPath("fisheyeAtakan")
fetcher.fetchFisheyeDataT265()

targetFrameIndex = 410
fisheyeWidth = 800
fisheyeHeight = 848

data = np.array((fetcher.fisheye1_t265[targetFrameIndex][1:-1]).split(", "),dtype=np.int)
data = data.reshape((fisheyeWidth,fisheyeHeight))

plot.imshow(data, cmap="gray")
 
print(data)
print(fetcher.fisheye1_t265.shape)