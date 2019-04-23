from keras.models import load_model

import os
import sklearn
import h5py as h5
import numpy as np
np.random.seed(1337) # for reproducibility
import random
random.seed( 3 ) # for reproducibility
import pandas as pd
import csv
import keras
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Flatten, GlobalAveragePooling1D, Dropout, Activation, TimeDistributed, LSTM
from keras.utils import to_categorical
from random import shuffle
from scipy import stats
from collections import Counter


#Increases the print size of pandas output
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199







#Network hyperparameters
epochs = 4
learningRate = .01
batch_size = 14
num_classes = 7
sfCutOff = 10





def Sort(sub_li, el):
    l = len(sub_li)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_li[j][el] > sub_li[j + 1][el]):
                tempo = sub_li[j]
                sub_li[j]= sub_li[j + 1]
                sub_li[j + 1]= tempo
    return sub_li




with h5.File('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/trafficData3.hdf5', 'r') as f:
    #X_train = f["data"][:]
    #y_train = f["subClassLabels"][:]
    dataTuple = f["wholeData"][:]


totalSf = []
for i in range(29992):
    totalSf.append(int(dataTuple[i][0]))

#print(totalSf)
sfDic = Counter(totalSf)
okaySfs = []
sfDicTrimmed = dict((k, v) for k, v in sfDic.items() if v >= sfCutOff)
for k, v in sfDicTrimmed.items():
    okaySfs.append(k)

print(len(set(okaySfs)))
np.set_printoptions(threshold=np.nan)

overCutoff = []
for i in range(29992):
    if dataTuple[i][0] in okaySfs:
         overCutoff.append(dataTuple[i])

sortedSF = Sort(overCutoff, 0)



end = 0
start = 0
for j in list(set(okaySfs)):
    #print(j)
    for i in range(len(sortedSF)):
        if sortedSF[i][0] == j:
            end = end + 1
    #print(start, end)
    sortedSF[start:end] = Sort(sortedSF[start:end], 1)
    start = end

# for i in range(100):
#     print(sortedSF[i][0], sortedSF[i][1])


X_train = []
y_train = []
for j in list(set(okaySfs)):
    X_train_sub = []
    y_train_sub = []
    count = 1
    for i in sortedSF:
        if i[0] == j and count < sfCutOff:
            count = count + 1
            X_train_sub.append(i[5:])
            y_train_sub.append(i[3])
    X_train.append(X_train_sub)
    y_train.append(y_train_sub)

y_train = np.array(y_train)
X_train = np.array(X_train)
print(X_train.shape)
print(y_train.shape)

X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state = 0)
#y_train = np_utils.to_categorical(y_train, num_classes)
X_train = np.expand_dims(X_train, axis=3)



benchmark_model_name = 'benchmark-model.h5'
trained_model = load_model(benchmark_model_name)
# generate predictions



result = trained_model.predict_classes(X_train, batch_size=batch_size, verbose=0)
for i in range(len(result)):
    if (stats.mode(result[i])[0][0] == int(y_train[i][0])):
        print(1)
    else:
        print(0)

# results = trained_model.predict_classes(X_train)
#
# for i in range(len(results)):
#     print(results[i], y_train[i])
