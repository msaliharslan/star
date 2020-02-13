import pandas
from matplotlib import pyplot as plt
import math
import numpy as np
from helper import *

path1 = "./Intel RealSense D435I/_device_0_sensor_2_Accel_0_imu_data.csv"
path2 = "./Intel RealSense T265/_device_0_sensor_0_Accel_0_imu_data.csv"
path3 = "./Intel RealSense D435I/_device_0_sensor_2_Gyro_0_imu_data.csv"
path4 = "./Intel RealSense T265/_device_0_sensor_0_Gyro_0_imu_data.csv"

acc1 = pandas.read_csv(path1).iloc[:,19:22]
acc2 = pandas.read_csv(path2).iloc[:,19:22]
gyro1 = pandas.read_csv(path3).iloc[:,14:17]
gyro2 = pandas.read_csv(path4).iloc[:,14:17]

mag_acc1 = []
mag_acc2 = []
mag_gyro1 = []
mag_gyro2 = []

for i in range(acc1.shape[0]):
    datum = math.sqrt(acc1.iloc[i,0]**2 + acc1.iloc[i,1]**2 + acc1.iloc[i,2]**2)
    mag_acc1.append(datum)

for i in range(acc2.shape[0]):
    datum = math.sqrt(acc2.iloc[i,0]**2 + acc2.iloc[i,1]**2 + acc2.iloc[i,2]**2)
    mag_acc2.append(datum)

for i in range(gyro1.shape[0]):
    datum = math.sqrt(gyro1.iloc[i,0]**2 + gyro1.iloc[i,1]**2 + gyro1.iloc[i,2]**2)
    mag_gyro1.append(datum)

for i in range(gyro2.shape[0]):
    datum = math.sqrt(gyro2.iloc[i,0]**2 + gyro2.iloc[i,1]**2 + gyro2.iloc[i,2]**2)
    mag_gyro2.append(datum)    
    

coverience_box = []
acc_lim = 50
acc_bias = int((acc1.shape[0] + acc2.shape[0]) / 4)
gyro_lim = 100
gyro_bias = int((gyro1.shape[0] + gyro2.shape[0]) / 4)

for i in range(-acc_lim,acc_lim + 1):
    sum = 0
    for j in range(acc_lim):
        sum += (mag_acc1[acc_bias + j] - mag_acc2[acc_bias + i +j])**2
    coverience_box.append(sum)
    
coverience_box_2 = []
for i in range(-gyro_lim,gyro_lim + 1):
    sum = 0
    for j in range(gyro_lim):
        sum += (mag_gyro1[gyro_bias + j] - mag_gyro2[gyro_bias + i +j])**2
    coverience_box_2.append(sum)
    
plt.figure(0)
plt.plot(mag_acc1, label="acc1")
plt.plot(mag_acc2, label="acc2")
plt.legend()
plt.figure(1)
plt.plot(range(-acc_lim,acc_lim+1), coverience_box)
plt.figure(2)
plt.plot(mag_gyro1, label="gyro1")
plt.plot(mag_gyro2, label="gyro2")
plt.legend()
plt.figure(3)
plt.plot(range(-gyro_lim,gyro_lim+1), coverience_box_2)



########################################
sample_window = 20
sample_stride = 25
delay_acc = (range(-acc_lim, acc_lim + 1))[np.argmin(coverience_box)]
delay_gyro = (range(-gyro_lim, gyro_lim + 1))[np.argmin(coverience_box_2)]
Rs = []



for sample in range(8, sample_window):
    

    
    
    v1 = unit_vector(gyro1.iloc[sample * sample_stride,:])
    v2 = unit_vector(gyro2.iloc[sample * sample_stride - delay_gyro, :])
    v3 = unit_vector(gyro1.iloc[sample * sample_stride + 10,:])
    v4 = unit_vector(gyro2.iloc[sample * sample_stride + 10 - delay_gyro, :])
    v5 = np.cross(v1, v3)
    v6 = np.cross(v2, v4)
    v7 = np.stack( (v1, v3, v5), -1 )
    v8 = np.stack( (v2, v4, v6), -1 ) 
    R = np.dot( np.linalg.inv(v7), v8)

    print(R)
    print("-----------")    
    
    cross = np.cross(v1, v2)
    angle = angle_between(v1, v2)
    sine = np.sin(angle)
    cose = np.cos(angle)
    
    cross_skew = [ [0, -1 * cross[2], cross[1] ],  \
                    [cross[2], 0, -1 * cross[0] ],  \
                    [-1 * cross[1], cross[0], 0 ] ]
        
         
    R = np.identity(3) + cross_skew + np.linalg.matrix_power(cross_skew, 2) * ( (1-cose) / sine**2 )
    Rs.append(R)
    v3 = np.dot(R, v1)
    #print(v2 - v3)

































