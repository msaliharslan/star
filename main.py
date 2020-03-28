#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:19:58 2020

@author: salih
"""

import sys
sys.path.append("Evaluators")
# from Evaluators.correlation import *
# from Evaluators.delay import *
# from Evaluators.rotationMatrix import *
import covariance
import delay
import rotationMatrix
import fetcher
import matplotlib.pyplot as plt
import utility



fetcher.fetchAllData()
coverianceBox,_,_, a = covariance.createCovarianceBoxForGyros_Mag(fetcher.gyro_d435i, fetcher.gyro_t265, 400)
plt.plot(range(a[0], a[1]), coverianceBox)
floating_delay = delay.getFloatIndexDelayForGivenCovarianceBox(coverianceBox, a[0])
print(floating_delay)
R = rotationMatrix.calculateRotationMatrixFromGyros(90, 100)

print(utility.isRotationMatrix(R))

Rn = utility.orthogonalize(R)
print(utility.isRotationMatrix(Rn))