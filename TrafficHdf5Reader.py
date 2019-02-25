"""
    This program reads a preexisting hdf5 file containing information from the
    original directory of network packet data.
"""


import os
import sklearn
import h5py as h5
import numpy as np
import random
np.random.seed(1337) # for reproducibility
import pandas as pd
import csv
from pandas import read_csv, DataFrame
from collections import Counter

nNumArr = []
def visitor_func(name, node):
    global nNumArr
    if isinstance(node, h5.Dataset):
        splitName = (node.name).split("/")
        cls = splitName[1]
        dev = splitName[2]
        sbCls = splitName[3]
        filename = splitName[4]
        nodeType = splitName[5]
        if nodeType == "pkts":
            pkt = node[:1,:1]
            #print(pkt)
        if nodeType == "superNum":
            sNum = node.value
            nNumArr.append(sNum)
            #print(sNum)

    else:
        splitName = (node.name).split("/")
        filename = splitName[4]


with h5.File('/Users/scifa/Documents/ucd-traffic-classification/trafficData.hdf5', 'r') as f:
    #f.visititems(visitor_func)
    Subclasses = ['HTTP', 'GoogleEarth', 'GoogleMap', 'Google_Common', 'Google+', 'GoogleSearch', 'GoogleAnalytics', 'TCPConnect', 'HTTPS', 'WebMail_Gmail', 'Hangouts', 'GooglePlay', 'YouTube', 'GoogleMusic', 'GoogleAdsense']
    Classes = ['GoogleEarth', 'GoogleMap', 'Google+', 'WebMail_Gmail', 'Hangouts', 'GooglePlay', 'YouTube', 'GoogleMusic']
    Devices = ['IOS', 'Android']
    for c in Classes:
        for d in Devices:
            for s in Subclasses:
                if c+"/"+d+"/"+s in f:
                    f[c][d][s].visititems(visitor_func)
nNumArr.sort()
#print(Counter(nNumArr).sort())
numOfFilesPerSF = dict((x,nNumArr.count(x)) for x in set(nNumArr))
SFPercentOfTotalData = dict((x, round((nNumArr.count(x)/len(nNumArr)),5)) for x in set(nNumArr))
"""
print('{:<10}'.format('Superfile'),'{:<10}'.format("# of files"), '{:<10}'.format("% of total"))
print('{:<10}'.format('----------'),'{:<10}'.format("----------"), '{:<10}'.format("----------"))
newlineStr = r"\newline"
for i in numOfFilesPerSF:
    print('{:<4}'.format(i),'{:<5}'.format(numOfFilesPerSF[i]), '{:<7}'.format(SFPercentOfTotalData[i]))
"""
#print(numOfFilesPerSF)
#print(SFPercentOfTotalData)
perTotal, t10, t20, t30, t40, t50, t60, t70, t80, t90, t100 = 0, 0,0,0,0,0,0,0,0,0,0
tenPer, twePer, thiPer, fouPer, fifPer, sixPer, sevPer, eigPer, ninPer, hunPer = [], [],[],[],[],[],[],[],[],[]
ranList = list(range(0,657))
for i in range(657):
    value = random.choice(ranList)
    ranList.remove(value)
    perTotal = perTotal + SFPercentOfTotalData[value]
    if perTotal < .10:
        tenPer.append(value)
        t10= t10 + SFPercentOfTotalData[value]
    elif perTotal >=.10 and perTotal < .20:
        twePer.append(value)
        t20= t20 + SFPercentOfTotalData[value]
    elif perTotal >=.20 and perTotal < .30:
        thiPer.append(value)
        t30= t30 + SFPercentOfTotalData[value]
    elif perTotal >=.30 and perTotal < .40:
        fouPer.append(value)
        t40= t40 + SFPercentOfTotalData[value]
    elif perTotal >=.40 and perTotal < .50:
        fifPer.append(value)
        t50= t50 + SFPercentOfTotalData[value]
    elif perTotal >=.50 and perTotal < .60:
        sixPer.append(value)
        t60= t60 + SFPercentOfTotalData[value]
    elif perTotal >=.60 and perTotal < .70:
        sevPer.append(value)
        t70= t70 + SFPercentOfTotalData[value]
    elif perTotal >=.70 and perTotal < .80:
        eigPer.append(value)
        t80= t80 + SFPercentOfTotalData[value]
    elif perTotal >=.80 and perTotal < .90:
        ninPer.append(value)
        t90= t90 + SFPercentOfTotalData[value]
    elif perTotal >=.90 and perTotal <= 1.00:
        hunPer.append(value)
        t100= t100 + SFPercentOfTotalData[value]
print(tenPer, t10, "\n")
print(twePer, t20, "\n")
print(thiPer, t30, "\n")
print(fouPer, t40, "\n")
print(fifPer, t50, "\n")
print(sixPer, t60, "\n")
print(sevPer, t70, "\n")
print(eigPer, t80, "\n")
print(ninPer, t90, "\n")
print(hunPer, t100, "\n")
