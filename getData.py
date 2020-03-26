import pandas
from matplotlib import pyplot as plt
import math
import numpy as np
from utility import *
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
acc_lim = 100
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

tempMinLen = min(len(d435_all_extracted_tuples_gyro), len(t265_all_extracted_tuples_gyro))

for i in range(tempMinLen - 2 * delay_gyro-20):
    
    print(d435_all_extracted_tuples_gyro[i][1] - t265_all_extracted_tuples_gyro[i+delay_gyro][1])


c_gyro =  coverience_box_2[delay_gyro + gyro_lim]
b_gyro = (coverience_box_2[delay_gyro + gyro_lim + 1] - coverience_box_2[delay_gyro + gyro_lim - 1] ) / 2
a_gyro =  coverience_box_2[delay_gyro + gyro_lim + 1] - b_gyro - c_gyro

c_acc =  coverience_box[delay_acc + acc_lim]
b_acc = (coverience_box[delay_acc + acc_lim + 1] - coverience_box[delay_acc + acc_lim - 1] ) / 2
a_acc =  coverience_box[delay_acc + acc_lim + 1] - b_acc - c_acc

delay_acc_float = -b_acc/(2 * a_acc) + delay_acc
delay_gyro_float = -b_gyro/(2 * a_gyro) + delay_gyro
acc_time_delay = delay_acc_float * acc_time_diff_const
acc_time_delay_2 = (d435_all_extracted_tuples_acc[100][1] - t265_all_extracted_tuples_acc[100][1] ) 
gyro_time_delay = delay_gyro_float * gyro_time_diff_const 
gyro_time_delay_2 = (d435_all_extracted_tuples_gyro[100][1] - t265_all_extracted_tuples_gyro[110][1] )

# Rs = []
# for sample in range(8, sample_window):
#     remove unit_vectors might be neccessary
#     v1 = unit_vector(gyro1.iloc[sample * sample_stride,:])
#     v2 = unit_vector(gyro1.iloc[sample * sample_stride + 10,:])
#     v3 = unit_vector(gyro1.iloc[sample * sample_stride + 20,:])    
    
#     v4_0 = unit_vector(gyro2.iloc[sample * sample_stride      + delay_gyro - 1, :])
#     v5_0 = unit_vector(gyro2.iloc[sample * sample_stride + 10 + delay_gyro - 1, :])
#     v6_0 = unit_vector(gyro2.iloc[sample * sample_stride + 20 + delay_gyro - 1, :])

#     v4_1 = unit_vector(gyro2.iloc[sample * sample_stride      + delay_gyro + 0, :])
#     v5_1 = unit_vector(gyro2.iloc[sample * sample_stride + 10 + delay_gyro + 0, :])
#     v6_1 = unit_vector(gyro2.iloc[sample * sample_stride + 20 + delay_gyro + 0, :])
    
#     v4_2 = unit_vector(gyro2.iloc[sample * sample_stride      + delay_gyro + 1, :])
#     v5_2 = unit_vector(gyro2.iloc[sample * sample_stride + 10 + delay_gyro + 1, :])
#     v6_2 = unit_vector(gyro2.iloc[sample * sample_stride + 20 + delay_gyro + 1, :])
        
#     if ( (delay_gyro_float - delay_gyro) > 0):
#         v4 = unit_vector(lineFit( [v4_1, v4_2],  delay_gyro_float - delay_gyro))
#         v5 = unit_vector(lineFit( [v5_1, v5_2],  delay_gyro_float - delay_gyro))
#         v6 = unit_vector(lineFit( [v6_1, v6_2],  delay_gyro_float - delay_gyro))
#     else:
#         v4 = unit_vector(lineFit( [v4_0, v4_1],  delay_gyro_float - delay_gyro + 1))
#         v5 = unit_vector(lineFit( [v5_0, v5_1],  delay_gyro_float - delay_gyro + 1))
#         v6 = unit_vector(lineFit( [v6_0, v6_1],  delay_gyro_float - delay_gyro + 1))
        

#     v7 = np.stack( (v1, v2, v3), -1 )
#     v8 = np.stack( (v4, v5, v6), -1 ) 
#     R = np.dot( np.linalg.inv(salih.orth(v7)), salih.orth(v8))
#     Rs.append(R)

    
    # print("**************\n")
    # print(R)
    # print("\n")
    
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





######################################################
#### SVD using all samples ###########################
#### gyro ######################################
zero = np.array( [0, 0, 0] )
limit = min( len(gyro1), len(gyro2) ) - abs(delay_gyro)
first_iteration = True
matched_vectors_linearfit = []
for i in range( abs(delay_gyro), limit-2):

    v1 = gyro1.iloc[i,:]
    
    v2_0 = gyro2.iloc[i - 1 + delay_gyro, :]
    v2_1 = gyro2.iloc[i + 0 + delay_gyro, :]
    v2_2 = gyro2.iloc[i + 1 + delay_gyro, :]
    
    if ( np.amax(np.isnan( (v1,v2_0, v2_1, v2_2)) ) ):
        continue
     
    if ( (delay_gyro_float - delay_gyro) > 0):
        v2 = lineFit( [v2_1, v2_2],  delay_gyro_float - delay_gyro)
    else:
        v2 = lineFit( [v2_0, v2_1],  delay_gyro_float - delay_gyro + 1)

    
    matched_vectors_linearfit.append( (v1, v2) )
    
    if(first_iteration):
        A = np.stack((v1))
        B = np.stack((v2))
        first_iteration = False

    A = np.vstack((A, v1))
    B = np.vstack((B, v2))

    
    
    # row1 = np.concatenate( (zero, -1 * v2[2] * v1, v2[1] * v1) )
    # row2 = np.concatenate( (v2[2] * v1, zero, -1 * v2[0] * v1) )
    # Ai = np.vstack( (row1, row2) )
    
    # if (first_iteration):
    #     A = np.stack( (Ai) )
    #     first_iteration = False
    
    # A = np.vstack( (A, Ai) )

# R = np.linalg.svd(A)[2].T[:,-1].reshape(3,3)
R = np.linalg.lstsq(A,B)[0]
print("Calculated rotation matrix from acc data: ", isRotationMatrix(R))

########## acc #######################
limit = min( len(acc1), len(acc2) ) - abs(delay_acc)
first_iteration = True
matched_vectors_linearfit_acc = []
for i in range( abs(delay_acc) +2, limit -2):

    v1 = acc1.iloc[i,:]
    
    
    v2_0 = acc2.iloc[i - 1 + delay_acc, :]
    v2_1 = acc2.iloc[i + 0 + delay_acc, :]
    v2_2 = acc2.iloc[i + 1 + delay_acc, :]
    
    if ( np.amax(np.isnan( (v1,v2_0, v2_1, v2_2)) ) ):
        continue
     
    if ( (delay_acc_float - delay_acc) > 0):
        v2 = lineFit( [v2_1, v2_2],  delay_acc_float - delay_acc)
    else:
        v2 = lineFit( [v2_0, v2_1],  delay_acc_float - delay_acc + 1)

    matched_vectors_linearfit_acc.append( (v1, v2) )
    
    if(first_iteration):
        A = np.stack((v1))
        B = np.stack((v2))
        first_iteration = False

    A = np.vstack((A, v1))
    B = np.vstack((B, v2))

    
    
    # row1 = np.concatenate( (zero, -1 * v2[2] * v1, v2[1] * v1) )
    # row2 = np.concatenate( (v2[2] * v1, zero, -1 * v2[0] * v1) )
    # Ai = np.vstack( (row1, row2) )
    
    # if (first_iteration):
    #     A = np.stack( (Ai) )
    #     first_iteration = False
    
    # A = np.vstack( (A, Ai) )

# R = np.linalg.svd(A)[2].T[:,-1].reshape(3,3)
R_acc = np.linalg.lstsq(A,B)[0]
print("Calculated rotation matrix from acc data: ", isRotationMatrix(R_acc))



##########################################################
################ reprojection error #########################
################ gyro #####################################
rpro_err_mag = []
rpro_err_ang = []
rpro_err_x = []
rpro_err_y = []
rpro_err_z = []
for v1, v2 in matched_vectors_linearfit:
    v1 = np.dot(R, np.transpose(v1))
    rpro_err_mag.append(np.linalg.norm(v2 - v1))
    rpro_err_ang.append(angle_between(v1, v2) * 180 / np.pi)
    rpro_err_x.append( v2[0] - v1[0] )
    rpro_err_y.append( v2[1] - v1[1] )
    rpro_err_z.append( v2[2] - v1[2] )



figen_old = plt.figure(4)
figen_old.suptitle('GYRO')
plt.title("Reprojection Error Magnitude")
plt.subplot(221)
plt.plot(range(len(matched_vectors_linearfit)), rpro_err_mag, 'r', label="magnitude", linewidth=1)
plt.legend()
plt.subplot(222)
plt.plot(range(len(matched_vectors_linearfit)), rpro_err_x, 'g', label="x", linewidth=1 )
plt.legend()
plt.subplot(223)
plt.plot(range(len(matched_vectors_linearfit)), rpro_err_y, 'b', label="y", linewidth=1)
plt.legend()
plt.subplot(224)
plt.plot(range(len(matched_vectors_linearfit)), rpro_err_z, 'purple', label="z", linewidth=1)
plt.legend()
plt.figure(5)
plt.plot(range(len(matched_vectors_linearfit)), rpro_err_ang, label="angle", linewidth=1) 
plt.title("Reprojection Error Angle")

################## acc ##########################################
rpro_err_mag = []
rpro_err_ang = []
rpro_err_x = []
rpro_err_y = []
rpro_err_z = []
for v1, v2 in matched_vectors_linearfit_acc:
    v1 = np.dot(R, np.transpose(v1))
    rpro_err_mag.append(np.linalg.norm(v2 - v1))
    rpro_err_ang.append(angle_between(v1, v2) * 180 / np.pi)
    rpro_err_x.append( v2[0] - v1[0] )
    rpro_err_y.append( v2[1] - v1[1] )
    rpro_err_z.append( v2[2] - v1[2] )



figen = plt.figure(6)
figen.suptitle('ACC')
plt.title("Reprojection Error Magnitude")
plt.subplot(221)
plt.plot(range(len(matched_vectors_linearfit_acc)), rpro_err_mag, 'r', label="magnitude", linewidth=1)
plt.legend()
plt.subplot(222)
plt.plot(range(len(matched_vectors_linearfit_acc)), rpro_err_x, 'g', label="x", linewidth=1 )
plt.legend()
plt.subplot(223)
plt.plot(range(len(matched_vectors_linearfit_acc)), rpro_err_y, 'b', label="y", linewidth=1)
plt.legend()
plt.subplot(224)
plt.plot(range(len(matched_vectors_linearfit_acc)), rpro_err_z, 'purple', label="z", linewidth=1)
plt.legend()
plt.figure(7)
plt.plot(range(len(matched_vectors_linearfit_acc)), rpro_err_ang, label="angle", linewidth=1) 
plt.title("Reprojection Error Angle")

######################################################

# windowSize = 100
# shiftMax = 50
# biasIndex = shiftMax + 10

# innerBag = []

# for i in range(-shiftMax, shiftMax+1):
#     inner = 0
#     for j in range(windowSize):
#         projected = np.dot(R, np.transpose(matched_vectors_linearfit[biasIndex + i + j][0]))
#         inner += np.dot(projected, matched_vectors_linearfit[biasIndex + j][1])
    
#     innerBag.append(inner)

# plt.figure(8)
# plt.plot(range(-shiftMax, shiftMax + 1), innerBag)
# plt.title("Cross correlation")
# plt.grid('both')
    
    
##########################################################
######## making rotation matrix orthogonal ##############
Y = np.dot(np.transpose(R), R) - np.identity(3)
Q = R - np.dot( np.dot(R, Y), (0.5*np.identity(3) - 3/8 * Y + 5/16 * np.dot(Y,Y) - 35 / 128 * np.linalg.matrix_power(Y,4)  ) )
    
##### Reprojection error using this matrix #############
rpro_err_mag = []
rpro_err_ang = []
rpro_err_x = []
rpro_err_y = []
rpro_err_z = []
for v1, v2 in matched_vectors_linearfit:
    v1 = np.dot(Q, np.transpose(v1))
    rpro_err_mag.append(np.linalg.norm(v2 - v1))
    rpro_err_ang.append(angle_between(v1, v2) * 180 / np.pi)
    rpro_err_x.append( v2[0] - v1[0] )
    rpro_err_y.append( v2[1] - v1[1] )
    rpro_err_z.append( v2[2] - v1[2] )



figen_old = plt.figure(8)
figen_old.suptitle('Orthogonalized')
plt.title("Reprojection Error Magnitude using orthogonalized matrix")
plt.subplot(221)
plt.plot(range(len(matched_vectors_linearfit)), rpro_err_mag, 'r', label="magnitude", linewidth=1)
plt.legend()
plt.subplot(222)
plt.plot(range(len(matched_vectors_linearfit)), rpro_err_x, 'g', label="x", linewidth=1 )
plt.legend()
plt.subplot(223)
plt.plot(range(len(matched_vectors_linearfit)), rpro_err_y, 'b', label="y", linewidth=1)
plt.legend()
plt.subplot(224)
plt.plot(range(len(matched_vectors_linearfit)), rpro_err_z, 'purple', label="z", linewidth=1)
plt.legend()
plt.figure(9)
plt.plot(range(len(matched_vectors_linearfit)), rpro_err_ang, label="angle", linewidth=1) 
plt.title("Reprojection Error Angle") 

    

    
    
























