from utility import *
import fetcher
from delay import *
from covariance import *


matched_vectors_linearfit_acc = None
matched_vectors_linearfit_gyro = None

def calculateRotationMatrixFromGyros(fetcherIndexBias, windowLength): #this functions requires gyro values fetched
    
    global matched_vectors_linearfit_gyro

    covarianceBox,_,_,indexes = createCovarianceBoxForGyros_Mag(fetcher.gyro_d435i, fetcher.gyro_t265, windowLength)

    gyro_longer = fetcher.gyro_d435i if fetcher.gyro_d435i.shape[0] > fetcher.gyro_t265.shape[0] else fetcher.gyro_t265
    gyro_shorter = fetcher.gyro_d435i if fetcher.gyro_d435i.shape[0] <= fetcher.gyro_t265.shape[0] else fetcher.gyro_t265

    delay_float_index = getFloatIndexDelayForGivenCovarianceBox(covarianceBox, indexes[0]) # delay is added to the longer one, since this delay is calculated keeping shorter one still

    matched_vectors_linearfit_gyro = []

    maxRight = gyro_shorter.shape[0] - int(delay_float_index) if gyro_shorter.shape[0] < (gyro_longer.shape[0] - int(delay_float_index)) else gyro_longer.shape[0] - int(delay_float_index)

    A = None
    B = None

    first_iteration = True
    for i in range(fetcherIndexBias, maxRight - 1):

        v1 = gyro_shorter.iloc[i,:]

        v2_0 = gyro_longer.iloc[i + 0 + int(delay_float_index), :]
        v2_1 = gyro_longer.iloc[i + 1 + int(delay_float_index), :]
        
        if ( np.amax(np.isnan( (v1,v2_0, v2_1)) ) ):
            continue              

        v2 = lineFit( [v2_0, v2_1],  delay_float_index - int(delay_float_index))

        matched_vectors_linearfit_gyro.append( (v1, v2) )
        
        if(first_iteration):
            A = np.stack((v1))
            B = np.stack((v2))
            first_iteration = False

        A = np.vstack((A, v1))
        B = np.vstack((B, v2))        

    R = np.linalg.lstsq(A,B,rcond=None)[0]

    return R


def calculateRotationMatrixFromAccs(fetcherIndexBias, windowLength): #this functions requires acc values fetched
    
    global matched_vectors_linearfit_acc

    correlationBox,_,_,indexes = createCovarianceBoxForAccs_Mag(fetcher.acc_d435i, fetcher.acc_t265, windowLength)

    acc_longer = fetcher.acc_d435i if fetcher.acc_d435i.shape[0] > fetcher.acc_t265.shape[0] else fetcher.acc_t265
    acc_shorter = fetcher.acc_d435i if fetcher.acc_d435i.shape[0] <= fetcher.acc_t265.shape[0] else fetcher.acc_t265

    delay_float_index = getFloatIndexDelayForGivenCovarianceBox(correlationBox, indexes[0]) # delay is added to the longer one, since this delay is calculated keeping shorter one still

    matched_vectors_linearfit_acc = []

    maxRight = acc_shorter.shape[0] - int(delay_float_index) if acc_shorter.shape[0] < (acc_longer.shape[0] - delay_float_index) else acc_longer.shape[0] - int(delay_float_index)

    A = None
    B = None

    first_iteration = True
    for i in range(fetcherIndexBias, maxRight - 1):

        v1 = acc_shorter.iloc[i,:]

        v2_0 = acc_longer.iloc[i + 0 + int(delay_float_index), :]
        v2_1 = acc_longer.iloc[i + 1 + int(delay_float_index), :]
        
        if ( np.amax(np.isnan( (v1,v2_0, v2_1)) ) ):
            continue              

        v2 = lineFit( [v2_0, v2_1],  delay_float_index - int(delay_float_index))

        matched_vectors_linearfit_acc.append( (v1, v2) )
        
        if(first_iteration):
            A = np.stack((v1))
            B = np.stack((v2))
            first_iteration = False

        A = np.vstack((A, v1))
        B = np.vstack((B, v2))        

    R = np.linalg.lstsq(A,B, rcond=None)[0]

    return R