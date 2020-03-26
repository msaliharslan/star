from utility import *
from fetcher import data
from delay import *
from correlation import *


def calculateRotationMatrixFromGyros(dataIndexBias, windowLength): #this functions requires gyro values fetched

    correlationBox,_,_ = createCorrelationBoxForGyros_Mag(data.gyro_d435i, data.gyro_t265, windowLength)

    gyro_longer = data.gyro_d435i if data.gyro_d435i.shape[0] > data.gyro_t265.shape[0] else data.gyro_t265
    gyro_shorter = data.gyro_d435i if data.gyro_d435i.shape[0] <= data.gyro_t265.shape[0] else data.gyro_t265

    delay_float_index = getFloatIndexDelayForGivenCorrelationBox(correlationBox) # delay is added to the longer one, since this delay is calculated keeping shorter one still

    matched_vectors_linearfit = []

    maxRight = gyro_shorter.shape[0] if gyro_shorter.shape[0] < (gyro_longer.shape[0] - delay_float_index) else gyro_longer.shape[0]

    A = None
    B = None

    first_iteration = True
    for i in range(dataIndexBias, maxRight):

        v1 = gyro_shorter.iloc[i,:]

        v2_0 = gyro_longer.iloc[i + 0 + int(delay_float_index), :]
        v2_1 = gyro_longer.iloc[i + 1 + int(delay_float_index), :]
        
        if ( np.amax(np.isnan( (v1,v2_0, v2_1)) ) ):
            continue              

        v2 = lineFit( [v2_0, v2_1],  delay_float_index - int(delay_float_index))

        matched_vectors_linearfit.append( (v1, v2) )
        
        if(first_iteration):
            A = np.stack((v1))
            B = np.stack((v2))
            first_iteration = False

        A = np.vstack((A, v1))
        B = np.vstack((B, v2))        

    R = np.linalg.lstsq(A,B)[0]

    return R


def calculateRotationMatrixFromAccs(dataIndexBias, windowLength): #this functions requires acc values fetched

    correlationBox,_,_ = createCorrelationBoxForAccs_Mag(data.acc_d435i, data.acc_t265, windowLength)

    acc_longer = data.acc_d435i if data.acc_d435i.shape[0] > data.acc_t265.shape[0] else data.acc_t265
    acc_shorter = data.acc_d435i if data.acc_d435i.shape[0] <= data.acc_t265.shape[0] else data.acc_t265

    delay_float_index = getFloatIndexDelayForGivenCorrelationBox(correlationBox) # delay is added to the longer one, since this delay is calculated keeping shorter one still

    matched_vectors_linearfit = []

    maxRight = acc_shorter.shape[0] if acc_shorter.shape[0] < (acc_longer.shape[0] - delay_float_index) else acc_longer.shape[0]

    A = None
    B = None

    first_iteration = True
    for i in range(dataIndexBias, maxRight):

        v1 = acc_shorter.iloc[i,:]

        v2_0 = acc_longer.iloc[i + 0 + int(delay_float_index), :]
        v2_1 = acc_longer.iloc[i + 1 + int(delay_float_index), :]
        
        if ( np.amax(np.isnan( (v1,v2_0, v2_1)) ) ):
            continue              

        v2 = lineFit( [v2_0, v2_1],  delay_float_index - int(delay_float_index))

        matched_vectors_linearfit.append( (v1, v2) )
        
        if(first_iteration):
            A = np.stack((v1))
            B = np.stack((v2))
            first_iteration = False

        A = np.vstack((A, v1))
        B = np.vstack((B, v2))        

    R = np.linalg.lstsq(A,B)[0]

    return R