import os
import sklearn
import h5py as h5
import math
import numpy as np
np.random.seed(1337) # for reproducibility
import pandas as pd
import csv
from pandas import read_csv, DataFrame
import random
random.seed( 3 ) # for reproducibility

"""
    Edit these to alter the number of packets and bytes pulled from each file.
"""

numOfPackets = 6 #Number of strings of network data used for each dataset
numOfBytes = 256  #Number of bytes used from each string of network data

numOfSf = 644 #Total number of superfiles in dataset +1

dataTuple = []

"""
    Assigns each class/subclass an integer label.
"""
def gNum(type):
    # typeTuple = [("Google+",0),("GoogleEarth",1),("GoogleMap",2),("GoogleMusic",3),("GooglePlay",4),("Hangouts",5),("WebMail_Gmail",6),("YouTube",7),("Google_Common",8),("GoogleAnalytics",9),("GoogleSearch",10),("GoogleAdsense",11),("TCPConnect",12),("HTTP",13),("HTTPS",14)]
    typeTuple = [("GoogleEarth",0),("GoogleMap",1),("GoogleMusic",2),("GooglePlay",3),("Hangouts",4),("WebMail_Gmail",5),("YouTube",6),("Google_Common",7),("GoogleAnalytics",8),("GoogleSearch",9),("GoogleAdsense",10),("TCPConnect",11),("HTTP",12),("HTTPS",13)]
    dic = dict(typeTuple)
    return dic[type]

"""
    Assigns each device type a interger label. (Android/IOS to int)
"""
def gDev(type):
    typeTuple = [("Android",0),("IOS",1)]
    dic = dict(typeTuple)
    return dic[type]

"""
    Takes a string of network packet data and converts bytes of hex data into normalized floats. Returns a list of floats for each string of bytes.
"""
def pad_and_convert(hexStr):
    if len(hexStr) < numOfBytes:
        hexStr += '00' * (numOfBytes-len(hexStr))
    else:
        hexStr = hexStr[:numOfBytes]
    return [float(int(hexStr[i]+hexStr[i+1], 16)/256) for i in range(0, numOfBytes, 2)]


"""
    Reads in a directory of files and extracts the needed network data and labels.
    The data is added into one large matrix that is then further processed later
    in the program.
"""
def getFiles():
    superFileNum = 0
    prevIDs = []
    os.chdir("/Users/brycekroencke/Documents/TrafficClassification/Project Related Files")
    os.chdir("/Users/brycekroencke/Documents/TrafficClassification/files/")
    for directories in os.listdir(os.getcwd()):
        if not directories.startswith('.') and directories != "Google+":
            dir = os.path.join('/Users/brycekroencke/Documents/TrafficClassification/files/', directories)
            os.chdir(dir)
            for idx, subdirectories in enumerate(os.listdir(os.getcwd())):
                if not subdirectories.startswith('.') and subdirectories != "Google+":
                    subdir = os.path.join(dir, subdirectories)
                    subdirSplit = subdirectories.split("_")
                    deviceType = subdirSplit[1]
                    os.chdir(subdir)
                    for filename in os.listdir(subdir):
                        if not filename.startswith('.'):
                            pktArr = []
                            if os.path.isfile:
                                file = (os.path.join(subdir, filename))
                                with open(file) as tsv:
                                    fileWithClassesRemoved = filename.split("-", 2)[2]
                                    fileUniqueID = (fileWithClassesRemoved.split("_", 1)[1]).rsplit(".", 3)[0]
                                    splitFilename = filename.split("-")
                                    underscoreSplitFilename = filename.split("_")
                                    fileClass = splitFilename[0]
                                    fileSubclass = splitFilename[1]
                                    dotSplitFilename = (underscoreSplitFilename[6]).split(".")
                                    fileFlowstate = filename[-15]
                                    if fileUniqueID not in prevIDs:
                                        prevIDs.append(fileUniqueID)
                                    for i in [i for i,x in enumerate(prevIDs) if x == fileUniqueID]:
                                        superFileNum = i
                                    okClasses = ['GoogleEarth', 'GoogleMap', 'WebMail_Gmail', 'Hangouts', 'GooglePlay', 'YouTube', 'GoogleMusic']
                                    okSubclasses = ['HTTP', 'GoogleEarth', 'GoogleMap', 'Google_Common', 'GoogleSearch', 'GoogleAnalytics', 'TCPConnect', 'HTTPS', 'WebMail_Gmail', 'Hangouts', 'GooglePlay', 'YouTube', 'GoogleMusic', 'GoogleAdsense']
                                    if fileClass in okClasses and fileSubclass in okSubclasses:
                                        count = 0
                                        startTime = 0
                                        pktStr = ""
                                        totalPktStr = ""
                                        numOfPacksRead = 0
                                        for idx2, line in enumerate(csv.reader(tsv, dialect="excel-tab")):
                                            if count == 0:
                                                startTime = line[1]
                                            if count <= numOfPackets:
                                                count = count + 1
                                                pktStr = line[3]
                                                pktArr.append(pad_and_convert(pktStr[0:numOfBytes]))
                                                numOfPacksRead = idx2
                                        if numOfPacksRead < numOfPackets:
                                            morePkts = numOfPackets-numOfPacksRead
                                            for i in range(morePkts):
                                                pktArr.append(pad_and_convert(""))
                                        flat_list = [item for sublist in pktArr for item in sublist]
                                        newList = [superFileNum, startTime, gDev(deviceType), gNum(directories), gNum(fileSubclass)]
                                        newList.extend(flat_list)
                                        dataTuple.append(newList)


"""
    We have a list of list. Each list within the list represents a file.
    Formatted as follows [Superfile#, Time, DeviceType, Class, Subclass, 1stPkt, ... , lastPkt]
    DeviceType:
    0 -> Android
    1 -> IOS

    Class/Subclass:
    0 -> Google+
    1 -> GoogleEarth
    2 -> GoogleMap
    3 -> GoogleMusic
    4 -> GooglePlay
    5 -> Hangouts
    6 -> WebMail_Gmail
    7 -> YouTube
    8 -> Google_Common
    9 -> GoogleAnalytics
    10 -> GoogleSearch
    11 -> GoogleAdsense
    12 -> TCPConnect
    13 -> HTTP
    14 -> HTTPS
"""



getFiles()

"""
Gets the start time of each flow and subtracts it from the rest of the files
within the flow. The data tuple is updated to contain the relative start times.
"""
numOfSfList = []
for i in range(numOfSf):
    tempList = []
    for j in range(len(dataTuple)):
        if dataTuple[j][0] == i:
            tempList.append(dataTuple[j])
            numOfSfList.append(i)
    startTime = min(l[1] for l in tempList)
    for time in tempList:
        time[1] = (float(time[1]) - float(startTime))


"""
Seperated the larger data matrix into smaller matricies.
Metadata contains: time, class, subclass
sclabelList contains: subclass
clabelList contains: class
dataList contains: packet data
"""
metaList = []
clabelList, sclabelList = [], []
dataList = []
for i in range(len(dataTuple)):
    metaList.append(dataTuple[i][0:4])
    sclabelList.append(dataTuple[i][4])
    clabelList.append(dataTuple[i][3])
    dataList.append(dataTuple[i][5:])

"""
Creates a hdf5 file and 4 datasets that contain the entire datasets metadata,
subclass labels, class labels, and packet data.
"""
os.chdir("/Users/brycekroencke/Documents/TrafficClassification/Project Related Files")
f = h5.File('trafficData.hdf5','w')
f.create_dataset("data", data=dataList)
f.create_dataset("metadata", data=metaList)
f.create_dataset("subClassLabels", data=sclabelList)
f.create_dataset("classLabels", data=clabelList)
f.close()


"""
Preprocessing step that splits the dataset into 10 ~equal sets for 10 fold cross validation.
This process ensures that all superfiles are placed into the same fold.
"""
SFPercentOfTotalData = dict((x, round((numOfSfList.count(x)/len(numOfSfList)),5)) for x in set(numOfSfList))
ranList = list(range(0,numOfSf))
ArrayInTenths = []
tempList = []
perTotal = 0
for i in range(len(ranList)):
    value = random.choice(ranList)
    ranList.remove(value)
    perTotal = perTotal + SFPercentOfTotalData[value]
    if perTotal < .10:
        for x in dataTuple:
            if x[0] == value:
                tempList.append(x)
    else:
        print(perTotal)
        ArrayInTenths.append(tempList)
        perTotal = 0
        tempList = []
print(perTotal)
ArrayInTenths.append(tempList)
perTotal = 0
tempList = []


"""
Adds the 10 fold cross validation data into the hdf5 file.
"""
os.chdir("/Users/brycekroencke/Documents/TrafficClassification/Project Related Files")

f = h5.File('trafficData.hdf5','w')
for i in range(10):
    tempD = []
    tempL1, tempL2 = [], []
    for j in range(len(ArrayInTenths[i])):
        tempD.append(ArrayInTenths[i][j][5:])
        tempL2.append(ArrayInTenths[i][j][4])
        tempL1.append(ArrayInTenths[i][j][3])
    f.create_dataset(str(i)+"data", data=tempD)
    f.create_dataset(str(i)+"Clabel", data=tempL1)
    f.create_dataset(str(i)+"SClabel", data=tempL2)
f.close()
