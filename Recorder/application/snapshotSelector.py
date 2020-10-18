import glob
import os
import shutil

#parameters

targetRecordIndex = 1
numShots = 6 


#find and change into the record directory

os.chdir("../Records")


targetRecordFolderName = ""

for directory in os.listdir(os.getcwd()):
    if(str(targetRecordIndex) == directory.split("_")[0]):
        targetRecordFolderName = directory
        

os.chdir("./" + targetRecordFolderName)


# store timestamps

fisheyeTimeStamps_dict = {}
rgbTimeStamps_dict = {}
depthTimeStamps_dict = {}



# fill out fisheyeTimeStamps

fileNames = glob.glob("./leftFisheye/*")
for fileName in fileNames:
    fisheyeTimeStamps_dict[fileName.split(".")[1].split("_")[-1]] = fileName


# fill out rgbTimeStamps

fileNames = glob.glob("./rgb/*")
for fileName in fileNames:
    rgbTimeStamps_dict[fileName.split(".")[1].split("_")[-1]] = fileName


# fill out depthTimeStamps

fileNames = glob.glob("./depth/*")
for fileName in fileNames:
    depthTimeStamps_dict[fileName.split(".")[1].split("_")[-1]] = fileName
    

fisheyeTimeStamps = list(fisheyeTimeStamps_dict.keys())
rgbTimeStamps = list(rgbTimeStamps_dict.keys())
depthTimeStamps = list(depthTimeStamps_dict.keys())
    
fisheyeTimeStamps.sort()
rgbTimeStamps.sort()
depthTimeStamps.sort()




#select the necessary timestamps according to the fisheye

selectedTimeStamps_fisheye = []

span = len(fisheyeTimeStamps) / numShots
startIndex = span / 2


for i in range(numShots):
    selectedTimeStamps_fisheye.append(fisheyeTimeStamps[int(i * span + startIndex)])
    



# synchronize the timestamps of rgb to of the fisheye

selectedTimeStamps_rgb = []

matchIndex = 0

for i in range(numShots):
    isSmaller = True
    while(isSmaller and matchIndex < len(rgbTimeStamps)):
        
        if(rgbTimeStamps[matchIndex] > selectedTimeStamps_fisheye[i]):
            isSmaller = False
            timeStamp = None
            
            currentFisheyeTimeStamp = selectedTimeStamps_fisheye[i]
            if(matchIndex != 0 and (abs(currentFisheyeTimeStamp - rgbTimeStamps[matchIndex]) > abs(currentFisheyeTimeStamp - rgbTimeStamps[matchIndex-1]))):
                matchIndex -= 1
                
            selectedTimeStamps_rgb.append(rgbTimeStamps[matchIndex])
        else:
            matchIndex +=1
    

# synchronize the timestamps of depth to of the fisheye

selectedTimeStamps_depth = []

matchIndex = 0

for i in range(numShots):
    isSmaller = True
    while(isSmaller and matchIndex < len(rgbTimeStamps)):
        
        if(rgbTimeStamps[matchIndex] > selectedTimeStamps_fisheye[i]):
            isSmaller = False
            timeStamp = None
            
            currentFisheyeTimeStamp = selectedTimeStamps_fisheye[i]
            if(matchIndex != 0 and (abs(currentFisheyeTimeStamp - rgbTimeStamps[matchIndex]) > abs(currentFisheyeTimeStamp - rgbTimeStamps[matchIndex-1]))):
                matchIndex -= 1
                
            selectedTimeStamps_depth.append(rgbTimeStamps[matchIndex])
        else:
            matchIndex +=1




#create snapshots session directory
            
os.mkdir("../../../SnapShots/t265_d435i/" + targetRecordFolderName)

os.mkdir("../../../SnapShots/t265_d435i/" + targetRecordFolderName + "/leftFisheye")
os.mkdir("../../../SnapShots/t265_d435i/" + targetRecordFolderName + "/rightFisheye")
os.mkdir("../../../SnapShots/t265_d435i/" + targetRecordFolderName + "/rgb")
os.mkdir("../../../SnapShots/t265_d435i/" + targetRecordFolderName + "/depth")


#first send leftFisheye images to snapshots

copyPath = "../../../SnapShots/t265_d435i/" + targetRecordFolderName + "/leftFisheye/"

for i in range(numShots):
    shutil.copy(fisheyeTimeStamps_dict[selectedTimeStamps_fisheye[i]], copyPath + selectedTimeStamps_fisheye[i])
    
































