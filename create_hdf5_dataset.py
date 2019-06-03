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
from collections import Counter

"""
    Edit these to alter the number of packets and bytes pulled from each file.
"""

numOfPackets = 6 #Number of strings of network data used for each dataset
numOfBytes = 256  #Number of bytes used from each string of network data
pktThreshold = 8

"""
    Pathways to change various parts of the dataset
"""

#Pathway to inside of directory containing the raw network data
network_files_pathway = "/Users/brycekroencke/Documents/TrafficClassification/files/"
#Pathway to the directory in which the new h5 file is to be stored
directory_for_h5 = "/Users/brycekroencke/Documents/TrafficClassification/Project Related Files"
#Name of newly created hdf5 file
name_of_lstm_h5 = "trafficData_lstm.hdf5"
name_of_cnn_h5 = "trafficData_cnn.hdf5"


"""
    Other globals
"""

#Global list used to create dataset
dataTuple = []

"""
    Assigns each class/subclass an integer label.
"""
def gNum(type):
    # typeTuple = [("Google+",0),("GoogleEarth",1),("GoogleMap",2),("GoogleMusic",3),("GooglePlay",4),("Hangouts",5),("WebMail_Gmail",6),("YouTube",7),("Google_Common",8),("GoogleAnalytics",9),("GoogleSearch",10),("GoogleAdsense",11),("TCPConnect",12),("HTTP",13),("HTTPS",14)]
    typeTuple = [("GoogleEarth",0),("GoogleMap",1),("GoogleMusic",2),("GooglePlay",3),("Hangouts",4),("WebMail_Gmail",5),("YouTube",6),("Google_Common",7),("GoogleAnalytics",8),("GoogleSearch",9),("GoogleAdsense",10),("TCPConnect",11),("HTTP",12),("HTTPS",13)]
    dic = dict(typeTuple)
    return dic[type]

def gClass(type):
    # typeTuple = [("Google+",0),("GoogleEarth",1),("GoogleMap",2),("GoogleMusic",3),("GooglePlay",4),("Hangouts",5),("WebMail_Gmail",6),("YouTube",7),("Google_Common",8),("GoogleAnalytics",9),("GoogleSearch",10),("GoogleAdsense",11),("TCPConnect",12),("HTTP",13),("HTTPS",14)]
    typeTuple = [(0,"GoogleEarth"),(1,"GoogleMap"),(2,"GoogleMusic"),(3,"GooglePlay"),(4,"Hangouts"),(5,"WebMail_Gmail"),(6,"YouTube"),(7,"Google_Common"),(8,"GoogleAnalytics"),(9,"GoogleSearch"),(10,"GoogleAdsense"),(11,"TCPConnect"),(12,"HTTP"),(13, "HTTPS")]
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
Sorts a matrix by a given element
"""
def Sort(sub_li, el):
    l = len(sub_li)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_li[j][el] > sub_li[j + 1][el]):
                tempo = sub_li[j]
                sub_li[j]= sub_li[j + 1]
                sub_li[j + 1]= tempo
    return sub_li



"""
    Reads in a directory of files and extracts the needed network data and labels.
    The data is added into one large matrix that is then further processed later
    in the program.
"""
def getFiles():
    superFileNum = 0
    prevIDs = []
    os.chdir(network_files_pathway)
    for directories in os.listdir(os.getcwd()):
        if not directories.startswith('.') and directories != "Google+":
            dir = os.path.join(network_files_pathway, directories)
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
                                    okClasses = ['GoogleEarth', 'GoogleMap', 'WebMail_Gmail', 'Hangouts', 'GooglePlay', 'YouTube', 'GoogleMusic']
                                    okSubclasses = ['HTTP', 'GoogleEarth', 'GoogleMap', 'Google_Common', 'GoogleSearch', 'GoogleAnalytics', 'TCPConnect', 'HTTPS', 'WebMail_Gmail', 'Hangouts', 'GooglePlay', 'YouTube', 'GoogleMusic', 'GoogleAdsense']
                                    if fileClass in okClasses and fileSubclass in okSubclasses:
                                        count = 0
                                        startTime = 0
                                        pktStr = ""
                                        totalPktStr = ""
                                        numOfPacksRead = 0
                                        maxPacks = 0
                                        for idx2, line in enumerate(csv.reader(tsv, dialect="excel-tab")):
                                            if count == 0:
                                                startTime = line[1]
                                            if count <= numOfPackets:
                                                count = count + 1
                                                pktStr = line[3]
                                                pktArr.append(pad_and_convert(pktStr[0:numOfBytes]))
                                                numOfPacksRead = idx2

                                            maxPacks = idx2
                                        if numOfPacksRead < numOfPackets:
                                            morePkts = numOfPackets-numOfPacksRead
                                            for i in range(morePkts):
                                                pktArr.append(pad_and_convert(""))

                                        if maxPacks >= pktThreshold:
                                            if fileUniqueID not in prevIDs:
                                                prevIDs.append(fileUniqueID)
                                            for i in [i for i,x in enumerate(prevIDs) if x == fileUniqueID]:
                                                superFileNum = i
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
    0 -> GoogleEarth
    1 -> GoogleMap
    2 -> GoogleMusic
    3 -> GooglePlay
    4 -> Hangouts
    5 -> WebMail_Gmail
    6 -> YouTube
    7 -> Google_Common
    8 -> GoogleAnalytics
    9 -> GoogleSearch
    10 -> GoogleAdsense
    11 -> TCPConnect
    12 -> HTTP
    13 -> HTTPS
"""

getFiles()

"""
Gets the start time of each flow and subtracts it from the rest of the files
within the flow. The data tuple is updated to contain the relative start times.
"""
numOfSf = 0
for tuple in dataTuple:
    if tuple[0] > numOfSf:
        numOfSf = tuple[0]

numOfSf = numOfSf + 1
print(numOfSf)



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




sfCutOff = 10 #number of timestamps per TimeDistribution


#FIND TOTAL NUMBER OF SUPERFILES AND CONSTRUCT A DICT
totalSf = []
for i in range(29992):
    totalSf.append(int(dataTuple[i][0]))
sfDic = Counter(totalSf)
okaySfs = []
sfDicTrimmed = dict((k, v) for k, v in sfDic.items() if v >= sfCutOff)
for k, v in sfDicTrimmed.items():
    okaySfs.append(k)

print(len(set(okaySfs)))
#np.set_printoptions(threshold=np.nan)

#TRIM THE SUPERFILES THAT ARE UNDER THE SF CUTOFF NUMBER
overCutoff = []
for i in range(29992):
    if dataTuple[i][0] in okaySfs:
         overCutoff.append(dataTuple[i])

#SORT THE FILES THAT ARE OVER THE SF CUTOFF NUMBER BY START TIME WHILE
#GROUPING BY SF NUMBER
sortedSF = Sort(overCutoff, 0)
end = 0
start = 0
for j in list(set(okaySfs)):
    for i in range(len(sortedSF)):
        if sortedSF[i][0] == j:
            end = end + 1
    sortedSF[start:end] = Sort(sortedSF[start:end], 1)
    start = end


print("-------------------------")
print("-------------------------")
print("-------------------------")
#make list from 0 to total # of files in superfile - the size of the batch.
#if size of this list is less than or equal to the sizeOfbatch use all indexes in the list
#else pick a random number from the list and add the sequence from that index to to index + batch size
#remove index from list when done
batchList = []
X_train = []
y_train = []
y_train_sc = []
time = []

batchesPerSf = 5
sizeOfbatch = 5
end = 0
start = 0
for i in list(set(okaySfs)):
    for m in range(len(sortedSF)):
        if sortedSF[m][0] == i:
            end = end + 1
    #print(sfDicTrimmed[i])
    listOfIds = list(range(0, sfDicTrimmed[i]-sizeOfbatch))
    print(len(listOfIds))
    if len(listOfIds) <= batchesPerSf:
        for j in listOfIds:
            #add sequence j to j + sizeOfbatch into array
            X_train_sub = []
            time_sub = []
            for x in range(sizeOfbatch):
                X_train_sub.append(sortedSF[j+x+start][5:])
                y_train_sub = sortedSF[j+x+start][3]
                y_train_sc_sub = sortedSF[j+x+start][4]
                time_sub.append(sortedSF[j+x+start][1])
            X_train.append(X_train_sub)
            y_train.append(y_train_sub)
            y_train_sc.append(y_train_sc_sub)
            time.append(time_sub)

            #print(str(j+start)+" -> "+str(j+sizeOfbatch+start))
    else:
        for j in range(batchesPerSf):
            # X_train_sub = []
            # time_sub = []
            # for i in range(5):
            #     X_train_sub.append(sortedSF[i+start][5:])
            #     y_train_sub = sortedSF[i+start][3]
            #     y_train_sc_sub = sortedSF[i+start][4]
            #     time_sub.append(sortedSF[i+start][1])
            #
            # X_train.append(X_train_sub)
            # y_train.append(y_train_sub)
            # y_train_sc.append(y_train_sc_sub)
            # time.append(time_sub)
            randomIndx = random.choice(listOfIds)
            X_train_sub = []
            time_sub = []
            time_for_norm = sortedSF[randomIndx+start][1]
            for x in range(sizeOfbatch):
                X_train_sub.append(sortedSF[randomIndx+x+start][5:])
                y_train_sub = sortedSF[randomIndx+x+start][3]
                y_train_sc_sub = sortedSF[randomIndx+x+start][4]
                time_sub.append(sortedSF[randomIndx+x+start][1] - time_for_norm)
            listOfIds.remove(randomIndx)
            X_train.append(X_train_sub)
            y_train.append(y_train_sub)
            y_train_sc.append(y_train_sc_sub)
            time.append(time_sub)
    start = end


print("-------------------------")
print("-------------------------")
print("-------------------------")



# #ADD EACH SF SEQUENCE TO THE TRAINING DATASET
# X_train = []
# y_train = []
# for j in list(set(okaySfs)):
#     X_train_sub = []
#     count = 1
#     for i in sortedSF:
#         if i[0] == j and count < sfCutOff:
#             count = count + 1
#             X_train_sub.append(i[5:])
#             y_train_sub = i[3]
#     X_train.append(X_train_sub)
#     y_train.append(y_train_sub)
y_train = np.array(y_train)
X_train = np.array(X_train)
time = np.array(time)
print(X_train.shape)
print(y_train.shape)
print(time.shape)
print(time[:3])

os.chdir(directory_for_h5)
f = h5.File(name_of_lstm_h5,'w')
f.create_dataset("X_train", data=X_train)
f.create_dataset("y_train", data=y_train)
f.create_dataset("y_train_sub_class", data=y_train_sc)
f.create_dataset("time", data=time)
f.close()


#
# f = h5.File(name_of_cnn_h5,'w')
# f.create_dataset("X_train", data=dataList)
# f.create_dataset("y_train", data=clabelList)
# f.create_dataset("y_train_sc", data=sclabelList)
# f.close()


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
os.chdir(directory_for_h5)
f = h5.File(name_of_cnn_h5,'w')
f.create_dataset("X_train", data=dataList)
f.create_dataset("y_train", data=clabelList)
f.create_dataset("y_train_sc", data=sclabelList)
for i in range(10):
    tempD = []
    tempL1, tempL2 = [], []
    for j in range(len(ArrayInTenths[i])):
        tempD.append(ArrayInTenths[i][j][5:])
        tempL2.append(ArrayInTenths[i][j][4])
        tempL1.append(ArrayInTenths[i][j][3])
    f.create_dataset(str(i)+"_X_train", data=tempD)
    f.create_dataset(str(i)+"_y_train", data=tempL1)
    f.create_dataset(str(i)+"_y_train_sub", data=tempL2)
f.close()
