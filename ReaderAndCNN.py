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
import math
from sklearn.model_selection import train_test_split, StratifiedKFold
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout, Convolution2D, Conv2D, MaxPooling2D, Lambda, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, AveragePooling2D, Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from sklearn.preprocessing import minmax_scale
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Flatten, GlobalAveragePooling1D, Dropout, Activation
from keras.utils import to_categorical
from keras.utils import np_utils
from random import shuffle
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199



nNumArr = []
num_classes = 15

def load_data_kfold():
    with h5.File('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/trafficData.hdf5', 'r') as f:
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
    nOnlyArr = []
    for x in nNumArr:
        nOnlyArr.append(x[0])
    numOfFilesPerSF = dict((x,nOnlyArr.count(x)) for x in set(nOnlyArr))
    SFPercentOfTotalData = dict((x, round((nOnlyArr.count(x)/len(nOnlyArr)),5)) for x in set(nOnlyArr))
    perTotal, t10, t20, t30, t40, t50, t60, t70, t80, t90, t100 = 0, 0,0,0,0,0,0,0,0,0,0
    X_train1 =[]
    X_train2 =[]
    X_train3 =[]
    X_train4 =[]
    X_train5 =[]
    X_train6 =[]
    X_train7 =[]
    X_train8 =[]
    X_train9 =[]
    X_train10 =[]
    y_train1 =[]
    y_train2 =[]
    y_train3 =[]
    y_train4 =[]
    y_train5 =[]
    y_train6 =[]
    y_train7 =[]
    y_train8 =[]
    y_train9 =[]
    y_train10 =[]
    ranList = list(range(0,657))
    with h5.File('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/trafficData.hdf5', 'r') as f:
        for i in range(657):
            value = random.choice(ranList)
            ranList.remove(value)
            perTotal = perTotal + SFPercentOfTotalData[value]
            if perTotal < .10:
                #tenPer.append(value)
                t10= t10 + SFPercentOfTotalData[value]
                for x in nNumArr:
                    if x[0] == value:
                        #print(f[x[1]][:])
                        X_train1.append(f[x[1]][:])
                        y_train1.append(x[2])
            elif perTotal >=.10 and perTotal < .20:
                t20= t20 + SFPercentOfTotalData[value]
                for x in nNumArr:
                    if x[0] == value:
                        X_train2.append(f[x[1]][:])
                        y_train2.append(x[2])
            elif perTotal >=.20 and perTotal < .30:
                t30= t30 + SFPercentOfTotalData[value]
                for x in nNumArr:
                    if x[0] == value:
                        X_train3.append(f[x[1]][:])
                        y_train3.append(x[2])
            elif perTotal >=.30 and perTotal < .40:
                t40= t40 + SFPercentOfTotalData[value]
                for x in nNumArr:
                    if x[0] == value:
                        X_train4.append(f[x[1]][:])
                        y_train4.append(x[2])
            elif perTotal >=.40 and perTotal < .50:
                t50= t50 + SFPercentOfTotalData[value]
                for x in nNumArr:
                    if x[0] == value:
                        X_train5.append(f[x[1]][:])
                        y_train5.append(x[2])
            elif perTotal >=.50 and perTotal < .60:
                t60= t60 + SFPercentOfTotalData[value]
                for x in nNumArr:
                    if x[0] == value:
                        X_train6.append(f[x[1]][:])
                        y_train6.append(x[2])
            elif perTotal >=.60 and perTotal < .70:
                t70= t70 + SFPercentOfTotalData[value]
                for x in nNumArr:
                    if x[0] == value:
                        X_train7.append(f[x[1]][:])
                        y_train7.append(x[2])
            elif perTotal >=.70 and perTotal < .80:
                t80= t80 + SFPercentOfTotalData[value]
                for x in nNumArr:
                    if x[0] == value:
                        X_train8.append(f[x[1]][:])
                        y_train8.append(x[2])
            elif perTotal >=.80 and perTotal < .90:
                t90= t90 + SFPercentOfTotalData[value]
                for x in nNumArr:
                    if x[0] == value:
                        X_train9.append(f[x[1]][:])
                        y_train9.append(x[2])
            elif perTotal >=.90 and perTotal <= 1.00:
                t100= t100 + SFPercentOfTotalData[value]
                for x in nNumArr:
                    if x[0] == value:
                        X_train10.append(f[x[1]][:])
                        y_train10.append(x[2])

    X_train = np.array([np.array(X_train1), np.array(X_train2), np.array(X_train3), np.array(X_train4), np.array(X_train5), np.array(X_train6), np.array(X_train7), np.array(X_train8), np.array(X_train9), np.array(X_train10)])
    y_train = np.array([y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7, y_train8, y_train9, y_train10])
    return X_train, y_train


def get_model():
    numOfClasses = 15
    activation = 'relu'
    model = Sequential()
    model.add(Conv1D(256, strides=2, input_shape=X_train_cv.shape[1:], activation=activation, kernel_size=4, padding='same'))
    model.add(MaxPooling1D())
    #(data_format='channels_first'))
    #model.add(Conv1D(128, strides=1, activation=activation, kernel_size=2, padding='same'))
    #model.add(MaxPooling1D)#(data_format='channels_first'))
    model.add(Flatten())
    #model.add(Dense(128, activation=activation))
    #model.add(Dense(128, activation=activation))
    #model.add(Dense(32, activation=activation))
    #model.add(Dense(32, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(numOfClasses, activation='softmax'))
    #print(model.summary())
    opt = keras.optimizers.Adam(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def Average(lst):
    return sum(lst) / len(lst)

def get_callbacks(name_weights, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    #reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    early_stop = keras.callbacks.EarlyStopping(monitor='acc', patience=1)
    return [mcp_save, early_stop] # reduce_lr_loss,

def gNum(type):
    typeTuple = [("Google+",0),("GoogleEarth",1),("GoogleMap",2),("GoogleMusic",3),("GooglePlay",4),("Hangouts",5),("WebMail_Gmail",6),("YouTube",7),("Google_Common",8),("GoogleAnalytics",9),("GoogleSearch",10),("GoogleAdsense",11),("TCPConnect",12),("HTTP",13),("HTTPS",14)]
    dic = dict(typeTuple)
    return dic[type]

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
            pkt = node[:]
        if nodeType == "superNum":
            sNum = node.value
            nNumArr.append((sNum, "/"+cls+"/"+dev+"/"+sbCls+"/"+filename+"/pkts", gNum(sbCls)))
    else:
        splitName = (node.name).split("/")
        filename = splitName[4]

batch_size = 128
final_confusion_matrix = np.zeros((num_classes, num_classes))
X_train, y_train = load_data_kfold()

for j in range(10):

    X_train_cv = []
    y_train_cv = []

    print('\nFold ',j)
    X_valid_cv = X_train[j]
    y_valid_cv = y_train[j]
    if j == 0:
        for i in range(1,10):
            for p in range(len(y_train[i])):
                X_train_cv.append(X_train[i][p])
                y_train_cv.append(y_train[i][p])
        print("j = " + str(j) + "  range = ")
        for x in range(1,10):
            print(str(x) + " ")
    elif j == 9:
        for i in range(0,9):
            for p in range(len(y_train[i])):
                X_train_cv.append(X_train[i][p])
                y_train_cv.append(y_train[i][p])
        print("j = " + str(j) + "  range = ")
        for x in range(1,10):
            print(str(x) + " ")
    else:
        for i in range(0,j):
            for p in range(len(y_train[i])):
                X_train_cv.append(X_train[i][p])
                y_train_cv.append(y_train[i][p])
        for i in range(j+1,10):
            for p in range(len(y_train[i])):
                X_train_cv.append(X_train[i][p])
                y_train_cv.append(y_train[i][p])
        print("j = " + str(j) + "  first range = "+ str(range(0,j)) + "  second range = " + str(range(j+1,10)))

    X_train_cv = np.array(X_train_cv)
    X_valid_cv = np.array(X_valid_cv)
    y_train_cv = np.array(y_train_cv)
    y_valid_cv = np.array(y_valid_cv)
    y_train_cv = np_utils.to_categorical(y_train_cv, num_classes = 15)
    y_valid_cv = np_utils.to_categorical(y_valid_cv, num_classes = 15)
    X_train_cv = np.expand_dims(X_train_cv, axis=2)
    X_valid_cv = np.expand_dims(X_valid_cv, axis=2)

    #print(X_train_cv)
    print(X_train_cv.shape)
    name_weights = "final_model_fold" + str(j) + "_weights.h5"
    callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)

    model = get_model()
    model.fit(X_train_cv, y_train_cv, verbose=1, epochs=4, batch_size=16, validation_data=(X_valid_cv, y_valid_cv), shuffle = True, callbacks = callbacks)
    print(model.evaluate(X_valid_cv, y_valid_cv))
    y_pred = model.predict(X_valid_cv)
    #y_pred = (y_pred > 0.5)
    print(pd.crosstab(y_valid_cv.argmax(axis=1), y_pred.argmax(axis=1), rownames=['True'], colnames=['Predicted'], margins=True))
    print("\n\n")
    print(pd.crosstab(y_valid_cv.argmax(axis=1), y_pred.argmax(axis=1), rownames=['True'], colnames=['Predicted'], margins=True, normalize = 'columns', dropna = False))
