#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 00:22:52 2020

@author: salih
"""

import numpy as np
from scipy.spatial.transform import Rotation as rot

T1 = np.array([0.031929191201925, 0.000113989575766, -3.38033132720739E-05]) # from fisheye1 to pose
T2 = np.array([-0.031929302960634, 7.47319791116752E-05, -4.20763171860017E-05]) # from fisheye2 to pose
T3 = np.array([-0.021229315549135, 2.60376800724771E-05, 9.00632439879701E-05]) # from IMU to pose

R1 = np.array([0.99998950958252, 0.003296238137409, -0.00291465758346, 0.001287228194997]) # from fisheye1 to pose
R2 = np.array([0.999997079372406, 0.000338271434885, -0.001731238677166, 0.001662679249421]) # from fisheye2 to pose
R3 = np.array([0, 1, 0, 0]) # from IMU to pose

T4 = np.array([0.014851336367428, 0.0004623454588, 0.000593442469835]) # from RGB to depth(infrared 1)

R4 = np.array([-0.003368464997038, -0.000677574775182, -0.006368808448315, -0.999973833560944]) # from RGB to depth(infrared 1)

R1 = rot.from_quat(R1).as_matrix()
R2 = rot.from_quat(R2).as_matrix()
R3 = rot.from_quat(R3).as_matrix()
R4 = rot.from_quat(R4).as_matrix()

R1_inv = np.linalg.inv(R1)


R = np.dot(R2, R1_inv)
temp = np.dot(R1_inv, np.dot(R2, T2))
T = - np.dot(R, (T1 - temp))



