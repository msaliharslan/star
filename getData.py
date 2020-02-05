import pandas
import numpy as np
from struct import *
from matplotlib import pyplot as plt
import math

path1 = "./atakansalih/_device_0_sensor_2_Accel_0_imu_data.csv"
path2 = "./Intel RealSense T265/_device_0_sensor_0_Accel_0_imu_data.csv"

acc1 = pandas.read_csv(path1).iloc[:,19:22]
acc2 = pandas.read_csv(path2).iloc[:,19:22]

mag_acc1 = []
mag_acc2 = []

for i in range(acc1.shape[0]):
    datum = math.sqrt(acc1.iloc[i,0]**2 + acc1.iloc[i,1]**2 + acc1.iloc[i,2]**2)
    mag_acc1.append(datum)

for i in range(acc2.shape[0]):
    datum = math.sqrt(acc2.iloc[i,0]**2 + acc2.iloc[i,1]**2 + acc2.iloc[i,2]**2)
    mag_acc2.append(datum)

coverience_box = []
for i in range(-50,51):
    sum = 0
    for j in range(50):
        sum += (mag_acc1[125 + j] - mag_acc2[125 + i +j])**2
    coverience_box.append(sum)
    
plt.figure(0)
plt.plot(mag_acc1, label="acc1")
plt.plot(mag_acc2, label="acc2")
plt.legend()
plt.figure(1)
plt.plot(range(-50,51), coverience_box)

# depthDataAsString = (pandas.read_csv(path1).iloc[0,12])
# depthData = np.fromstring(depthDataAsString[1:-1], dtype=int, sep=',')
# print(type(depthData))
# print(np.amax(depthData))
# print(depthData.shape)

# depth = []

# formatString = ">1H"

# depthScale = 0.001

# for i in range(int(len(depthData)/2)): 
#     value = (depthData[2*i+1] << 8) + depthData[i*2]
#     depth.append(value)