import pandas
import os


appendAllDefault = "../Records/activeCsv/"

paths = {

    #metadata paths

    #t265
    "path_metadata_acc_t265" : "/_device_0_sensor_0_Accel_0_imu_metadata.csv",
    "path_metadata_gyro_t265" : "/_device_0_sensor_0_Gyro_0_imu_metadata.csv",
    "path_metadata_fisheye_t265" : "/_device_0_sensor_0_Fisheye_1_image_metadata.csv",
    
    #d435i
    "path_metadata_acc_d435i" : "/_device_0_sensor_2_Accel_0_imu_metadata.csv",
    "path_metadata_gyro_d435i" : "/_device_0_sensor_2_Gyro_0_imu_metadata.csv",
    "path_metadata_depth_d435i" : "/_device_0_sensor_0_Depth_0_image_metadata.csv",

    #data paths

    #t265
    "path_data_acc_t265" : "/_device_0_sensor_0_Accel_0_imu_data.csv",
    "path_data_gyro_t265": "/_device_0_sensor_0_Gyro_0_imu_data.csv",
    "path_data_fisheye1_t265" : "/_device_0_sensor_0_Fisheye_1_image_data.csv",
    "path_data_fisheye2_t265" : "/_device_0_sensor_0_Fisheye_2_image_data.csv",

    #d435i
    "path_data_acc_d435i" : "/_device_0_sensor_2_Accel_0_imu_data.csv",
    "path_data_gyro_d435i" : "/_device_0_sensor_2_Gyro_0_imu_data.csv",

}

def insertFocusDataPath( focusDir ):
    for key in paths.keys():
        paths[key] = appendAllDefault + focusDir + paths[key]

#fetched data
#d435i
acc_d435i = None
gyro_d435i = None
metadata_acc_d435i = None
metadata_gyro_d435i = None
metadata_depth_d435i = None
depth_d435i = None
#t265
acc_t265 = None
gyro_t265 = None
metadata_acc_t265 = None
metadata_gyro_t265 = None
metadata_fisheye_t265 = None
fisheye1_t265 = None
fisheye2_t265 = None
#metadata fetching

def fetchAccAndGyroMetadataForD435i():
    global metadata_acc_d435i
    global metadata_gyro_d435i
    metadata_acc_d435i = pandas.read_csv(paths["path_metadata_acc_d435i"])
    metadata_gyro_d435i = pandas.read_csv(paths["path_metadata_gyro_d435i"])

def fetchDepthMetadataForD435i():
    global metadata_depth_d435i
    metadata_depth_d435i = pandas.read_csv(paths["path_metadata_depth_d435i"])

def fetchAllMetadataForD435i():
    fetchAccAndGyroMetadataForD435i()
    fetchDepthMetadataForD435i()

def fetchAccAndGyroMetadataForT265():
    global metadata_acc_t265
    global metadata_gyro_t265
    metadata_acc_t265 = pandas.read_csv(paths["path_metadata_acc_t265"])
    metadata_gyro_t265 = pandas.read_csv(paths["path_metadata_gyro_t265"])

def fetchFisheyeMetadataForT265():
    global metadata_fisheye_t265
    metadata_fisheye_t265 = pandas.read_csv(paths["path_metadata_fisheye_t265"])

def fetchAllMetadataForT265():
    fetchAccAndGyroMetadataForT265()
    fetchFisheyeMetadataForT265()


def fetchAllMetadata():
    fetchAllMetadataForD435i()
    fetchAllMetadataForT265()

#data fetching

def fetchAccAndGyroDataD435i():
    global acc_d435i 
    global gyro_d435i
    acc_d435i = pandas.read_csv(paths["path_data_acc_d435i"]).iloc[:,19:22]
    gyro_d435i = pandas.read_csv(paths["path_data_gyro_d435i"]).iloc[:,14:17]

def fetchDepthDataD435i(): #not implemented yet
    return None

def fetchAllDataD435i(): #not complete since fetchDepthData435i is not complete
    fetchAccAndGyroDataD435i()
    fetchDepthDataD435i()


def fetchAccAndGyroDataT265():
    global acc_t265
    global gyro_t265 
    acc_t265 = pandas.read_csv(paths["path_data_acc_t265"]).iloc[:,19:22]
    gyro_t265 = pandas.read_csv(paths["path_data_gyro_t265"]).iloc[:,14:17]

def fetchFisheyeDataT265(): #not implemented yet
    global fisheye1_t265
    global fisheye2_t265
    fisheye1_t265 = pandas.read_csv(paths["path_data_fisheye1_t265"]).iloc[:,12]
    fisheye2_t265 = pandas.read_csv(paths["path_data_fisheye2_t265"]).iloc[:,12]

def fetchAllDataT265(): #not complete since fetchFisheyeDataT265 is not complete
    fetchAccAndGyroDataT265()
    fetchFisheyeDataT265()
    
def fetchAllData():
    fetchAllDataD435i()
    fetchAllDataT265()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
