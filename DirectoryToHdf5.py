import os
import sklearn
import h5py as h5
import math
import numpy as np
np.random.seed(1337) # for reproducibility
import pandas as pd
import csv
from pandas import read_csv, DataFrame




numOfPackets = 32 #Number of strings of network data used for each dataset
numOfBytes = 784  #Number of bytes used from each string of network data
numOfClasses = 15 #Total number of classes to be classified. (Number of different labels)

labelList = []
dataList = []


"""
    Assigns each class/subclass an integer label.
"""
def gNum(type):
    typeTuple = [("Google+",0),("GoogleEarth",1),("GoogleMap",2),("GoogleMusic",3),("GooglePlay",4),("Hangouts",5),("WebMail_Gmail",6),("YouTube",7),("Google_Common",8),("GoogleAnalytics",9),("GoogleSearch",10),("GoogleAdsense",11),("TCPConnect",12),("HTTP",13),("HTTPS",14)]
    dic = dict(typeTuple)
    return dic[type]


"""
    Takes a string of network packet data and converts bytes of hex data into normalized floats. Returns a list of floats for each string of bytes.
"""
def pad_and_convert(hexStr):
    if len(hexStr) < 128:
        hexStr += '00' * (128-len(hexStr))
    else:
        hexStr = hexStr[:254]
    return [float(int(hexStr[i]+hexStr[i+1], 16)/128) for i in range(0, 128, 2)]


"""
    Reads in a directory of files and extracts the needed network data and labels.
"""
def getFiles():
    #text_file = open("superFileInfo.txt", "w")
    #text_file.write('{:9}'.format("Superfile")+" "+'{:15}'.format("Class")+" "+'{:19}'.format("Subclass")+" "+'{:50}'.format("Unique File ID")+"\n")
    #text_file.write("----- --------------- --------------- --------------------------------------------------"+"\n")
    superFileNum = 0
    prevIDs = []
    os.chdir("/Users/brycekroencke/Documents/TrafficClassification/Project Related Files")
    f = h5.File('trafficData.hdf5','w')
    os.chdir("/Users/brycekroencke/Documents/TrafficClassification/files/")
    for directories in os.listdir(os.getcwd()):
        if not directories.startswith('.'):
            dir = os.path.join('/Users/brycekroencke/Documents/TrafficClassification/files/', directories)
            os.chdir(dir)
            for idx, subdirectories in enumerate(os.listdir(os.getcwd())):
                if not subdirectories.startswith('.'):
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
                                    #Getting unique # ID
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
                                        #text_file.write('{:9}'.format(str(i))+" "+'{:15}'.format(fileClass)+" "+'{:19}'.format(fileSubclass)+" "+'{:50}'.format(fileUniqueID)+"\n")
                                        #print('{:9}'.format(str(i))+" "+'{:15}'.format(fileClass)+" "+'{:19}'.format(fileSubclass)+" "+'{:50}'.format(fileUniqueID)+"\n")
                                        superFileNum = i
                                    #if directories == fileSubclass:
                                    okSubclasses = ['HTTP', 'GoogleEarth', 'GoogleMap', 'Google_Common', 'Google+', 'GoogleSearch', 'GoogleAnalytics', 'TCPConnect', 'HTTPS', 'WebMail_Gmail', 'Hangouts', 'GooglePlay', 'YouTube', 'GoogleMusic', 'GoogleAdsense']
                                    if fileSubclass in okSubclasses:
                                        count = 0
                                        pktStr = ""
                                        totalPktStr = ""
                                        numOfPacksRead = 0
                                        for idx2, line in enumerate(csv.reader(tsv, dialect="excel-tab")):
                                            if count <= 4:
                                                count = count + 1
                                                pktStr = line[3]
                                                pktArr.append(pad_and_convert(pktStr[0:128]))
                                                numOfPacksRead = idx2
                                        #print("\n\n"+str(count)+"\n\n")
                                        print(numOfPacksRead)
                                        if numOfPacksRead < 4:
                                            print("here")
                                            morePkts = 4-numOfPacksRead
                                            for i in range(morePkts):
                                                print("****")
                                                pktArr.append(pad_and_convert(""))
                                        flat_list = [item for sublist in pktArr for item in sublist]
                                        print(len(flat_list))
                                        fileGrp = f.create_group(directories+"/"+deviceType+"/"+fileSubclass+"/"+str(filename))
                                        fileDset = fileGrp.create_dataset("pkts", data=flat_list)
                                        fileSuperNum = fileGrp.create_dataset("superNum", data=superFileNum)
                                        #print(pktArr)

    f.close()


getFiles()
