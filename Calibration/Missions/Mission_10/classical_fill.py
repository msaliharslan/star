import numpy as np
import os
from matplotlib import pyplot as plt

while(os.getcwd().split('/')[-1] != 'star'):
    os.chdir("../")
    
from Calibration.Missions.Mission_8.iterative_packager import map_img_2_img
from Packager import packager as pack

# read Package matrices first
package_file = open("./thingsToSubmit/submission4/leftPackage/leftPackage.npy", "rb")
leftPackageMatrix = np.load(package_file)
package_file.close()

package_file = open("./thingsToSubmit/submission4/rightPackage/rightPackage.npy", "rb")
rightPackageMatrix = np.load(package_file)
package_file.close()

pt_cld_file = open("./thingsToSubmit/submission4/leftPackage/pt_cloud.npy", "rb")
left_pt_cloud = np.load(pt_cld_file)
pt_cld_file.close()

pt_cld_file = open("./thingsToSubmit/submission4/rightPackage/pt_cloud.npy", "rb")
right_pt_cloud = np.load(pt_cld_file)
pt_cld_file.close()

left = leftPackageMatrix[:,:,0][0:992, 0:992]
left_measured_flag = leftPackageMatrix[:,:,1][0:992, 0:992]
left_measured = leftPackageMatrix[:,:,2][0:992, 0:992]
left_measured[left_measured_flag == 0] = np.nan


right = rightPackageMatrix[:,:,0][0:992, 0:992]
right_measured_flag = rightPackageMatrix[:,:,1][0:992, 0:992]
right_measured = rightPackageMatrix[:,:,2][0:992, 0:992]
right_measured[right_measured_flag == 0] = np.nan

upsample_rate = 1
KLeft = pack.KLeft * upsample_rate
KLeft[2,2] = 1
KRight = pack.KRight * upsample_rate
KRight[2,2] = 1
KL_inv = np.linalg.inv(KLeft)


width = 992
height = 992

# form the coordinate system
x = np.arange(width)
y = np.arange(height)

X, Y = np.meshgrid(x, y)
mask = np.isnan(left_measured_flag)

# define neighborhood
N = np.arange(-4, 5)
nx, ny = np.meshgrid(N, N)

# get the indexes for possible depth values
x_check = np.expand_dims(X[mask], 1) + np.expand_dims(nx.flatten(), 0)
y_check = np.expand_dims(Y[mask], 1) + np.expand_dims(ny.flatten(), 0)

x_check = np.clip(x_check, 0, width-1)
y_check = np.clip(y_check, 0, height-1)

# get possible depth values
z_check = left_measured[y_check, x_check]

# select the valid depth only
check_mask = ~np.isnan(z_check)

# check points in 3D
P = np.stack((x_check[check_mask], y_check[check_mask], np.ones(np.sum(check_mask)))) * z_check[check_mask] * 1e-3


P_3D = np.dot(KL_inv, P)

P_3D = np.dot(pack.R_LF2RF, P_3D) + pack.T_LF2RF
P_3D = np.dot(KRight, P_3D)
P_3D /= P_3D[2, :]
P_3D = np.round(P_3D).astype(np.int32)

P_3D = np.clip(P_3D, 0, 991)

right_corresponding_intensity = right[ P_3D[1, :], P_3D[0, :] ]
right_corresponding_depth = right_measured[ P_3D[1, :], P_3D[0, :] ]

depth_diff = z_check[check_mask] - right_corresponding_depth
intensity_diff = left[y_check, x_check][check_mask] - right_corresponding_intensity

plt.figure(1)
plt.hist( np.abs( depth_diff), bins=np.arange(0, 1000, 20), density=True )
plt.figure(2)
plt.hist( np.abs(intensity_diff ), bins=np.arange(0, 255, 5), density=True )


depth_diff_threshold = 50
further_mask = depth_diff < -1 * depth_diff_threshold
closer_mask = depth_diff > depth_diff_threshold
match_mask = ~np.logical_or(further_mask, closer_mask) 

match_intensity = np.abs(intensity_diff) < 13

combined_match = np.logical_and(match_mask, match_intensity)




index_map =  np.expand_dims(np.arange(0, check_mask.shape[0], 1, dtype=np.uint32), 1) + np.zeros( (1, 81), dtype=np.uint32 )
index_map[check_mask]

E = np.full(check_mask.shape, np.inf)
temp = E[check_mask]
temp[combined_match] = depth_diff[combined_match] * 0.3 / 50 + intensity_diff[combined_match] * 0.7 / 13
E[check_mask] = temp

E_sorted = np.argsort(E, axis=1)

x_fill = x_check[check_mask][combined_match]
y_fill = y_check[check_mask][combined_match]

depth_diff[index_map[check_mask]]































