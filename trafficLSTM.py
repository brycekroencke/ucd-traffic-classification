"""
    This program reads a preexisting hdf5 file containing information from the
    original directory of network packet data.
"""


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
from collections import Counter


#Increases the print size of pandas output
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199





#Network hyperparameters
epochs = 4
learningRate = .0005
batch_size = 7
num_classes = 7


sfCutOff = 10 #number of timestamps per TimeDistribution

"""
Loads datasets from the hdf5 file that were generated from toH5.py and does preprocessing on the data
"""
def load_data():
    with h5.File('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/trafficData3.hdf5', 'r') as f:
        dataTuple = f["wholeData"][:]
        metadata = f["metadata"][:]


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
        np.set_printoptions(threshold=np.nan)

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

        #ADD EACH SF SEQUENCE TO THE TRAINING DATASET
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

        #FORMAT DATA FOR TIME DISTRIBUTED CNN INPUT
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state = 0)
        X_valid = X_train[:50]
        y_valid = y_train[:50]
        X_train = X_train[50:]
        y_train = y_train[50:]
        X_train = np.array(X_train)
        X_valid = np.array(X_valid)
        y_train = np.array(y_train)
        y_valid = np.array(y_valid)
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_valid = np_utils.to_categorical(y_valid, num_classes)
        X_train = np.expand_dims(X_train, axis=3)
        X_valid = np.expand_dims(X_valid, axis=3)
    return X_train, y_train, X_valid, y_valid


"""
Defines the 1D DCNN model using TimeDistributed
"""
def get_model():
    activation = 'relu'
    model = Sequential()
    model.add(TimeDistributed(Conv1D(256, (2), padding='same',  strides=1, input_shape=X_train.shape[1:], name = "con1")))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Conv1D(128,(2), strides=1, activation=activation, name = "con2")))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.5))
    #WE WANT TO HAVE RETURN SEQUENCES = FALSE SO THAT THE RETURN VALUE IS A SINGLE VALUE
    model.add(LSTM(50, return_sequences=True, name = "lstm_layer", dropout=0.5))
    model.add(Dense(50))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


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
Go from class label to an integer value.
"""
def gNum(type):
    typeTuple = [("GoogleEarth",0),("GoogleMap",1),("GoogleMusic",2),("GooglePlay",3),("Hangouts",4),("WebMail_Gmail",5),("YouTube",6),("Google_Common",7),("GoogleAnalytics",8),("GoogleSearch",9),("GoogleAdsense",10),("TCPConnect",11),("HTTP",12),("HTTPS",13)]
    dic = dict(typeTuple)
    return dic[type]




X_train, y_train, X_valid, y_valid = load_data()

model = get_model()
model.load_weights('my_model_weights.h5', by_name=True)
model.fit(X_train, y_train, verbose=1, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid), shuffle = True)
print(model.summary())
#y_pred = model.predict(X_valid)
#y_pred = (y_pred > .5)
benchmark_model_name = 'benchmark-model.h5'
model.save(benchmark_model_name)

print(model.evaluate(X_valid, y_valid))
