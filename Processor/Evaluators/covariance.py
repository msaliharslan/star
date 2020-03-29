#this module does : 
'''

'''

import math
import numpy as np


def createCovarianceBoxForAccs_Mag(acc_d435i, acc_t265, windowLength):

    #input validity checks
    if(min(acc_d435i.shape[0], acc_t265.shape[0]) <= windowLength):
        print("Error(createCorrelationBoxForAccs_Mag): windowLength is too large")
        return

    magAcc_d435i = []
    magAcc_t265 = []


    for i in range(acc_d435i.shape[0]):
        datum = math.sqrt(acc_d435i.iloc[i,0]**2 + acc_d435i.iloc[i,1]**2 + acc_d435i.iloc[i,2]**2)
        magAcc_d435i.append(datum)

    for i in range(acc_t265.shape[0]):
        datum = math.sqrt(acc_t265.iloc[i,0]**2 + acc_t265.iloc[i,1]**2 + acc_t265.iloc[i,2]**2)
        magAcc_t265.append(datum)

    
    magAcc_longer = None
    magAcc_shorter = None
    
    holdStillName = None # this is string, either "acc_d435i" or "acc_t265"

    if(min(acc_d435i.shape[0], acc_t265.shape[0]) == acc_d435i.shape[0]):
        holdStillName = "acc_d435i"
        magAcc_shorter = magAcc_d435i
        magAcc_longer = magAcc_t265
    else:
        holdStillName = "acc_t265"
        magAcc_shorter = magAcc_t265
        magAcc_longer = magAcc_d435i

    
    covarianceBox = []

    startIndex = int(len(magAcc_shorter) / 2)

    if(startIndex >= len(magAcc_shorter) - windowLength):
        startIndex = len(magAcc_shorter) - windowLength - 1
    
    maxSlideRightRange = len(magAcc_longer) - 1 - (startIndex + windowLength)
    maxSlideLeftRange = startIndex - int(len(magAcc_shorter) / 8)
    
    for i in range(-maxSlideLeftRange,maxSlideRightRange):
        covarianceBox.append(np.cov([magAcc_shorter[startIndex:startIndex+windowLength], magAcc_longer[startIndex+i:startIndex+i+windowLength]])[0,1])

    print("Correlationbox for acc is calculated while fixing ", holdStillName)    
            
    return covarianceBox, magAcc_longer, magAcc_shorter, (-maxSlideLeftRange, maxSlideRightRange)


def createCovarianceBoxForGyros_Mag(gyro_d435i, gyro_t265, windowLength):

    #input validity checks
    if(min(gyro_d435i.shape[0], gyro_t265.shape[0]) <= windowLength):
        print("Error(createCorrelationBoxForGyros_Mag): windowLength is too large")
        return

    magGyro_d435i = []
    magGyro_t265 = []


    for i in range(gyro_d435i.shape[0]):
        datum = math.sqrt(gyro_d435i.iloc[i,0]**2 + gyro_d435i.iloc[i,1]**2 + gyro_d435i.iloc[i,2]**2)
        magGyro_d435i.append(datum)

    for i in range(gyro_t265.shape[0]):
        datum = math.sqrt(gyro_t265.iloc[i,0]**2 + gyro_t265.iloc[i,1]**2 + gyro_t265.iloc[i,2]**2)
        magGyro_t265.append(datum)

    magGyro_longer = None
    magGyro_shorter = None

    holdStillName = None # this is string, either "gyro_d435i" or "gyro_t265"

    if(min(gyro_d435i.shape[0], gyro_t265.shape[0]) == gyro_d435i.shape[0]):
        holdStillName = "gyro_d435i"
        magGyro_shorter = magGyro_d435i
        magGyro_longer = magGyro_t265
    else:
        holdStillName = "gyro_t265"
        magGyro_shorter = magGyro_t265
        magGyro_longer = magGyro_d435i

    covarianceBox = []

    startIndex = int(len(magGyro_shorter) / 2)

    if(startIndex >= len(magGyro_shorter) - windowLength):
        startIndex = len(magGyro_shorter) - windowLength - 1
    
    maxSlideRightRange = len(magGyro_longer) - 1 - (startIndex + windowLength)
    maxSlideLeftRange = startIndex - int(len(magGyro_shorter) / 8)
    
    for i in range(-maxSlideLeftRange,maxSlideRightRange):
        covarianceBox.append(np.cov([magGyro_shorter[startIndex:startIndex+windowLength], magGyro_longer[startIndex+i:startIndex+i+windowLength]])[0,1])
        

    print("Correlationbox for gyro is calculated while fixing ", holdStillName)    
            
    return covarianceBox, magGyro_longer, magGyro_shorter, (-maxSlideLeftRange,maxSlideRightRange)

