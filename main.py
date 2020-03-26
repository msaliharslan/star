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
corelationBox,_,_, a = createCorrelationBoxForAccs_Mag(fetcher.acc_d435i, fetcher.acc_t265, 100)
plt.plot(range(a[0], a[1]), corelationBox)
floating_delay = getFloatIndexDelayForGivenCorrelationBox(corelationBox, a[0])

#R = calculateRotationMatrixFromAccs(90, 100)