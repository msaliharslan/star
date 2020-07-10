import os
import numpy as np

os.chdir("../")
import fetcher


fisheyeWidth = 800
fisheyeHeight = 848

fisheye1Frames_extracted = []
fisheye2Frames_extracted = []


def pickFramesFromExtractedData(numSamples=10):
        

    targetRecords = open( "Records/targetRecords.txt", "r" )
    
    
    for focusDir in targetRecords:
        
        
        focusDir = focusDir.rstrip(".bag\n")
        fetcher.insertFocusDataPath(focusDir)
        fetcher.fetchFisheyeDataT265()
        
        numFrames = min(fetcher.fisheye1_t265.shape[0], fetcher.fisheye2_t265.shape[0])
        
        targetFrameIndexes = np.linspace(0, numFrames - 1, numSamples, dtype=np.uint32)
        
        for targetIndex in targetFrameIndexes:
            
            data1 = np.array((fetcher.fisheye1_t265[targetIndex][1:-1]).split(", "),dtype=np.uint8)
            data1 = data1.reshape((fisheyeWidth,fisheyeHeight))
            
            data2 = np.array((fetcher.fisheye2_t265[targetIndex][1:-1]).split(", "),dtype=np.uint8)
            data2 = data2.reshape((fisheyeWidth,fisheyeHeight))    
            
            fisheye1Frames_extracted.append(data1)
            fisheye2Frames_extracted.append(data2)
            
            
            
    targetRecords.close()
    
    return (fisheye1Frames_extracted, fisheye2Frames_extracted)
