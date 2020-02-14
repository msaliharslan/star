import pandas
from matplotlib import pyplot as plt
import math
import numpy as np
from helper import *
import scipy.linalg as salih
from syncher import  *

runfile("syncher.py")

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
plt.title("acc magnitudes")
plt.figure(1)
plt.plot(range(-acc_lim,acc_lim+1), coverience_box)
plt.title("acc coverience box")
plt.figure(2)
plt.plot(mag_gyro1, label="gyro1")
plt.plot(mag_gyro2, label="gyro2")
plt.legend()
plt.title("gyro magnitudes")
plt.figure(3)
plt.plot(range(-gyro_lim,gyro_lim+1), coverience_box_2)
plt.title("gyro coverience box")


########################################
gyro_time_diff_const = 1/199 * 1e3
acc_time_diff_const = 16
sample_window = 20
sample_stride = 25
delay_acc = (range(-acc_lim, acc_lim + 1))[np.argmin(coverience_box)]
delay_gyro = (range(-gyro_lim, gyro_lim + 1))[np.argmin(coverience_box_2)]

c_gyro =  coverience_box_2[delay_gyro + gyro_lim]
b_gyro = (coverience_box_2[delay_gyro + gyro_lim + 1] - coverience_box_2[delay_gyro + gyro_lim - 1] ) / 2
a_gyro =  coverience_box_2[delay_gyro + gyro_lim + 1] - b_gyro - c_gyro

c_acc =  coverience_box[delay_acc + acc_lim]
b_acc = (coverience_box[delay_acc + acc_lim + 1] - coverience_box[delay_acc + acc_lim - 1] ) / 2
a_acc =  coverience_box[delay_acc + acc_lim + 1] - b_acc - c_acc

delay_acc_float = -b_acc/(2 * a_acc) + delay_acc
delay_gyro_float = -b_gyro/(2 * a_gyro) + delay_gyro
acc_time_delay = delay_acc_float * acc_time_diff_const
acc_time_delay_2 = (d435_all_extracted_tuples_acc[0][1] - t265_all_extracted_tuples_acc[0][1] * 1e-3) 
gyro_time_delay = delay_gyro_float * gyro_time_diff_const 
gyro_time_delay_2 = (d435_all_extracted_tuples_gyro[0][1] - t265_all_extracted_tuples_gyro[0][1] * 1e-3)

Rs = []
for sample in range(8, sample_window):
    
    v1 = unit_vector(gyro1.iloc[sample * sample_stride,:])
    v2 = unit_vector(gyro1.iloc[sample * sample_stride + 10,:])
    v3 = unit_vector(gyro1.iloc[sample * sample_stride + 20,:])    
    
    v4_0 = unit_vector(gyro2.iloc[sample * sample_stride      + delay_gyro, :])
    v5_0 = unit_vector(gyro2.iloc[sample * sample_stride + 10 + delay_gyro, :])
    v6_0 = unit_vector(gyro2.iloc[sample * sample_stride + 20 + delay_gyro, :])

    v4_1 = unit_vector(gyro2.iloc[sample * sample_stride      + delay_gyro, :])
    v5_1 = unit_vector(gyro2.iloc[sample * sample_stride + 10 + delay_gyro, :])
    v6_1 = unit_vector(gyro2.iloc[sample * sample_stride + 20 + delay_gyro, :])
    
    v4_2 = unit_vector(gyro2.iloc[sample * sample_stride      + delay_gyro, :])
    v5_2 = unit_vector(gyro2.iloc[sample * sample_stride + 10 + delay_gyro, :])
    v6_2 = unit_vector(gyro2.iloc[sample * sample_stride + 20 + delay_gyro, :])
        
    if ( (delay_gyro_float - delay_gyro) > 0):
        v4 = lineFit( [v4_1, v4_2],  delay_gyro_float - delay_gyro)
        v5 = lineFit( [v5_1, v5_2],  delay_gyro_float - delay_gyro)
        v6 = lineFit( [v6_1, v6_2],  delay_gyro_float - delay_gyro)
    else:
        v4 = lineFit( [v4_0, v4_1],  delay_gyro_float - delay_gyro + 1)
        v5 = lineFit( [v5_0, v5_1],  delay_gyro_float - delay_gyro + 1)
        v6 = lineFit( [v6_0, v6_1],  delay_gyro_float - delay_gyro + 1)
        

    v7 = np.stack( (v1, v2, v3), -1 )
    v8 = np.stack( (v4, v5, v6), -1 ) 
    R = np.dot( np.linalg.inv(salih.orth(v7)), salih.orth(v8))
    Rs.append(R)
    
    print("**************\n")
    print(R)
    print("\n")
    
    # print("aaa: ", angle_between(v1,v2) * 180 / np.pi)
    # print("-------------------------------")
    # print("bbb: ", angle_between(v4,v5) * 180 / np.pi)
    # print("\n\n")

    
    # cross = np.cross(v1, v2)
    # angle = angle_between(v1, v2)
    # sine = np.sin(angle)
    # cose = np.cos(angle)
    
    # cross_skew = [ [0, -1 * cross[2], cross[1] ],  \
    #                 [cross[2], 0, -1 * cross[0] ],  \
    #                 [-1 * cross[1], cross[0], 0 ] ]
        
         
    # R = np.identity(3) + cross_skew + np.linalg.matrix_power(cross_skew, 2) * ( (1-cose) / sine**2 )
    # Rs.append(R)
    # v3 = np.dot(R, v1)
    # print(v2 - v3)


# initial_selected_acc_index = int(len(d435_all_extracted_tuples_acc) / 2) + +150

# for initial_selected_acc_index in range(len(d435_all_extracted_tuples_acc)-delay_acc -40):
#     best_tof_diff_accVsGyro = 1e100
#     best_gyro_index = -1
            
#     for j in range(len(d435_all_extracted_tuples_gyro)):
        
#         current_tof = d435_all_extracted_tuples_gyro[j][1]
        
#         current_tof_difference = abs(current_tof - d435_all_extracted_tuples_acc[initial_selected_acc_index][1])
#         best_gyro_index = j
        
#         if ((current_tof_difference > best_tof_diff_accVsGyro)):
#             break        
#         else:
#             best_tof_diff_accVsGyro = current_tof_difference
       
#     otherSideAccTOF = t265_all_extracted_tuples_acc[ initial_selected_acc_index + delay_acc ][1]
#     otherSideGyroTOF = t265_all_extracted_tuples_gyro[ best_gyro_index + delay_gyro ][1]
    
#     print("this side tof dif : ", best_tof_diff_accVsGyro)
#     print("other side tof dif: ", otherSideAccTOF - otherSideGyroTOF)        






























