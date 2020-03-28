#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:19:58 2020

@author: salih
"""
from Evaluators.correlation import *
from Evaluators.delay import *
#from Evaluators.rotationMatrix import *
import fetcher
import matplotlib.pyplot as plt



fetcher.fetchAllData()
coverianceBox,_,_, a = createCoverianceBoxForGyros_Mag(fetcher.gyro_d435i, fetcher.gyro_t265, 400)
plt.plot(range(a[0], a[1]), coverianceBox)
floating_delay = getFloatIndexDelayForGivenCoverianceBox(coverianceBox, a[0])
print(floating_delay)
#R = calculateRotationMatrixFromAccs(90, 100)