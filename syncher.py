#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:57:28 2020

@author: salih
"""
import pandas


path1 = "./Intel RealSense D435I/_device_0_sensor_2_Gyro_0_imu_metadata.csv"
path2 = "./Intel RealSense T265/_device_0_sensor_0_Gyro_0_imu_metadata.csv"
path3 = "./Intel RealSense D435I/_device_0_sensor_0_Depth_0_image_metadata.csv"
path4 = "./Intel RealSense T265/_device_0_sensor_0_Fisheye_1_image_metadata.csv"
path5 = "./Intel RealSense D435I/_device_0_sensor_2_Accel_0_imu_metadata.csv"
path6 = "./Intel RealSense T265/_device_0_sensor_0_Accel_0_imu_metadata.csv"

d435_gyro_metadata = pandas.read_csv(path1)
d435_depth_metadata = pandas.read_csv(path3)
d435_acc_metadata = pandas.read_csv(path5)
t265_gyro_metadata = pandas.read_csv(path2)
t265_fisheye_metadata = pandas.read_csv(path4)
t265_acc_metadata = pandas.read_csv(path6)

d435_gyro_corres_rts = [] # rosbag timestamps of matched gyro and depth d435
d435_acc_corres_rts = [] # rosbag timestamps of matched acc and depth d435
t265_gyro_corres_rts = [] # rosbag timestamps of matched gyro and fisheye t265
t265_acc_corres_rts = [] # rosbag timestamps of matched acc and fisheye t265
d435_all_extracted_tuples_depth = [] # rosbag timestamps and time of arrivals of d435
d435_all_extracted_tuples_gyro = []  # rosbag timestamps and time of arrivals of d435
d435_all_extracted_tuples_acc = [] # rosbag timestamps and time of arrivals of d435
t265_all_extracted_tuples_fisheye = [] # rosbag timestamps and time of arrivals of t265
t265_all_extracted_tuples_gyro = [] # rosbag timestamps and time of arrivals of t265
t265_all_extracted_tuples_acc = [] # rosbag timestamps and time of arrivals of t265


# d435 depth extracting all time of arrivals with corresponding rosbag timestamps
for i in range(0, d435_depth_metadata.shape[0]):
    if (d435_depth_metadata.iloc[i,1] == '"Time Of Arrival"'):
        current_tof = int(d435_depth_metadata.iloc[i, 2][1:-1])
        current_rts = d435_depth_metadata.iloc[i, 0]
        d435_all_extracted_tuples_depth.append( (current_rts, current_tof) )
        

# d435 gyro extracting all time of arrivals with corresponding rosbag timestamps
for i in range(0, d435_gyro_metadata.shape[0]):
    if (d435_gyro_metadata.iloc[i,1] == '"Time Of Arrival"'):
        current_tof = int(d435_gyro_metadata.iloc[i, 2][1:-1])
        current_rts = d435_gyro_metadata.iloc[i, 0]
        d435_all_extracted_tuples_gyro.append( (current_rts, current_tof) )  
        
# d435 acc extracting all time of arrivals with corresponding rosbag timestamps
for i in range(0, d435_acc_metadata.shape[0]):
    if (d435_acc_metadata.iloc[i,1] == '"Time Of Arrival"'):
        current_tof = int(d435_acc_metadata.iloc[i, 2][1:-1])
        current_rts = d435_acc_metadata.iloc[i, 0]
        d435_all_extracted_tuples_acc.append( (current_rts, current_tof) )  
        
        
# t265 fiheye extracting all time of arrivals with corresponding rosbag timestamps
for i in range(0, t265_fisheye_metadata.shape[0]):
    if (t265_fisheye_metadata.iloc[i,1] == '"Time Of Arrival"'):
        current_tof = int(t265_fisheye_metadata.iloc[i, 2][1:-1])
        current_rts = t265_fisheye_metadata.iloc[i, 0]
        t265_all_extracted_tuples_fisheye.append( (current_rts, current_tof) )
        

# t265 gyro extracting all time of arrivals with corresponding rosbag timestamps
for i in range(0, t265_gyro_metadata.shape[0]):
    if (t265_gyro_metadata.iloc[i,1] == '"Time Of Arrival"'):
        current_tof = int(t265_gyro_metadata.iloc[i, 2][1:-1])
        current_rts = t265_gyro_metadata.iloc[i, 0]
        t265_all_extracted_tuples_gyro.append( (current_rts, current_tof) )  
        
# t265 acc extracting all time of arrivals with corresponding rosbag timestamps
for i in range(0, t265_acc_metadata.shape[0]):
    if (t265_acc_metadata.iloc[i,1] == '"Time Of Arrival"'):
        current_tof = int(t265_acc_metadata.iloc[i, 2][1:-1])
        current_rts = t265_acc_metadata.iloc[i, 0]
        t265_all_extracted_tuples_acc.append( (current_rts, current_tof) ) 
        
# d435 matching gyro and depth frames
last_matched_gyro = -1
last_matched_acc = -1
for i in range(len(d435_all_extracted_tuples_depth)):
        current_tof = d435_all_extracted_tuples_depth[i][1]
        current_rts = d435_all_extracted_tuples_depth[i][0]
        
        best_tof_difference = abs(current_tof - d435_all_extracted_tuples_gyro[last_matched_gyro+1][1])
        for j in range(last_matched_gyro + 2, len(d435_all_extracted_tuples_gyro)):
            current_tof_difference = abs(current_tof - d435_all_extracted_tuples_gyro[j][1])
            if (current_tof_difference > best_tof_difference):
                d435_gyro_corres_rts.append( d435_all_extracted_tuples_gyro[j-1][0] )
                last_matched_gyro = j-1
                break
            else:
                best_tof_difference = current_tof_difference
                
        best_tof_difference = abs(current_tof - d435_all_extracted_tuples_acc[last_matched_acc+1][1])
        for j in range(last_matched_acc + 2, len(d435_all_extracted_tuples_acc)):
            current_tof_difference = abs(current_tof - d435_all_extracted_tuples_acc[j][1])
            if (current_tof_difference > best_tof_difference):
                d435_acc_corres_rts.append( d435_all_extracted_tuples_acc[j-1][0] )
                last_matched_acc = j-1
                break
            else:
                best_tof_difference = current_tof_difference
        
# t265 matching gyro and depth frames
last_matched_gyro = -1
last_matched_acc = -1
for i in range(len(t265_all_extracted_tuples_fisheye)):
        current_tof = t265_all_extracted_tuples_fisheye[i][1]
        current_rts = t265_all_extracted_tuples_fisheye[i][0]
        
        best_tof_difference = abs(current_tof - t265_all_extracted_tuples_gyro[last_matched_gyro+1][1])
        for j in range(last_matched_gyro + 2, len(t265_all_extracted_tuples_gyro)):
            current_tof_difference = abs(current_tof - t265_all_extracted_tuples_gyro[j][1])
            if (current_tof_difference > best_tof_difference):
                t265_gyro_corres_rts.append( t265_all_extracted_tuples_gyro[j-1][0] )
                last_matched_gyro = j-1
                break
            else:
                best_tof_difference = current_tof_difference
        
        best_tof_difference = abs(current_tof - t265_all_extracted_tuples_acc[last_matched_acc+1][1])
        for j in range(last_matched_acc + 2, len(t265_all_extracted_tuples_acc)):
            current_tof_difference = abs(current_tof - t265_all_extracted_tuples_acc[j][1])
            if (current_tof_difference > best_tof_difference):
                t265_acc_corres_rts.append( t265_all_extracted_tuples_acc[j-1][0] )
                last_matched_acc = j-1
                break
            else:
                best_tof_difference = current_tof_difference
            
        
print("çalıştım")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        