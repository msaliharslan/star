import numpy as np


def getIntIndexDelayForAccs(coverianceBox, slideLeftIndex):
    return np.argmax(coverianceBox) + slideLeftIndex
def getIntIndexDelayForGyros(coverianceBox, slideLeftIndex):
    return np.argmax(coverianceBox) + slideLeftIndex

def getFloatIndexDelayForGivenCoverianceBox(coverianceBox, slideLeftIndex):
    maxIndex = np.argmax(coverianceBox)
    
    y1 = coverianceBox[maxIndex-1]
    y2 = coverianceBox[maxIndex]
    y3 = coverianceBox[maxIndex+1]

    a = (y1 + y3 - 2*y2) / 2
    b = (y2-y1) + a

    return (-b/(2*a)) + maxIndex + slideLeftIndex