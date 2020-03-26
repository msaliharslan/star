import numpy as np


def getIntIndexDelayForAccs(correlationBox, slideLeftIndex):
    return np.argmax(correlationBox) + slideLeftIndex
def getIntIndexDelayForGyros(correlationBox, slideLeftIndex):
    return np.argmax(correlationBox) + slideLeftIndex

def getFloatIndexDelayForGivenCorrelationBox(correlationBox, slideLeftIndex):
    maxIndex = np.argmax(correlationBox)
    
    y1 = correlationBox[maxIndex-1]
    y2 = correlationBox[maxIndex]
    y3 = correlationBox[maxIndex+1]

    a = (y1 + y3 - 2*y2) / 2
    b = (y2-y1) + a

    return (-b/(2*a)) + maxIndex + slideLeftIndex