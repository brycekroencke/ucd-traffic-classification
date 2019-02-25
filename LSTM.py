### LOAD PACKAGES
import os
import sklearn
import tensorflow as tf
import h5py as h5
import numpy as np
np.random.seed(1337) # for reproducibility
import keras
import pandas as pd
import csv
from pandas import read_csv, DataFrame
from sklearn.preprocessing import minmax_scale
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Flatten, TimeDistributed, LSTM, GlobalAveragePooling1D, Dropout, Activation
from keras.utils import to_categorical
from keras.utils import np_utils
from random import shuffle
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


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
    if len(hexStr) < 400:
        hexStr += '00' * (400-len(hexStr))
    else:
        hexStr = hexStr[:400]
    return [float(int(hexStr[i]+hexStr[i+1], 16)/255) for i in range(0, 400, 2)]


"""
    Reads in a directory of files and extracts the needed network data and labels.
"""
def getFiles():
    global dataList
    global labelList
    os.chdir("/Users/scifa/Documents/ai_research/files/")
    for directories in os.listdir(os.getcwd()):
        dir = os.path.join('/Users/scifa/Documents/ai_research/files/', directories)
        os.chdir(dir)
        for subdirectories in os.listdir(os.getcwd()):
            subdir = os.path.join(dir, subdirectories)
            subdirSplit = subdirectories.split("_")
            deviceType = subdirSplit[1]
            os.chdir(subdir)
            for filename in os.listdir(subdir):
                d = []
                l = []
                if os.path.isfile:
                    file = (os.path.join(subdir, filename))
                    with open(file) as tsv:
                        splitFilename = file.split("-")
                        underscoreSplitFilename = filename.split("_")
                        fileSubclass = splitFilename[1]
                        dotSplitFilename = (underscoreSplitFilename[6]).split(".")
                        fileFlowstate = filename[-15]
                        d = []
                        #if directories == fileSubclass:
                        okSubclasses = ['HTTP', 'GoogleEarth', 'GoogleMap', 'Google_Common', 'Google+', 'GoogleSearch', 'GoogleAnalytics', 'TCPConnect', 'HTTPS', 'WebMail_Gmail', 'Hangouts', 'GooglePlay', 'YouTube', 'GoogleMusic', 'GoogleAdsense']
                        if fileSubclass in okSubclasses:
                            count = 0
                            pktStr = ""
                            totalPktStr = ""
                            for line in csv.reader(tsv, dialect="excel-tab"):
                                if count < 4:
                                    count = count + 1
                                    pktStr = line[3]
                                    totalPktStr = totalPktStr + pktStr[0:100]
                                    #dataList.append(pad_and_convert(line[3]))
                            dataList.append(pad_and_convert(totalPktStr))
                            labelList.append(fileSubclass)

def main():
    print("hello")
    #Open directory and extract the needed network data and labels
    getFiles()
    global dataList
    global labelList
    #Convert label into an int value
    intLabelList = []
    for i in labelList:
        intLabelList.append(gNum(i))

    #Determine size of test and train data
    totalSizeOfDataset = len(dataList)
    testSize = int(.10 * totalSizeOfDataset)
    trainSize = int(.90 * totalSizeOfDataset)

    #Initial shuffle of data
    dataList, intLabelList = sklearn.utils.shuffle(dataList, intLabelList, random_state = 0)

    #Break data into training and test
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(0, trainSize):
        y_train.append(intLabelList[i])
        x_train.append(dataList[i])
    for i in range(trainSize, totalSizeOfDataset):
        y_test.append(intLabelList[i])
        x_test.append(dataList[i])

    #Fix the data so the CNN can read it
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np_utils.to_categorical(y_train, numOfClasses)
    y_train = np.array(y_train)
    y_test = np_utils.to_categorical(y_test, numOfClasses)
    y_test = np.array(y_test)
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)


    #Making the 1D CNN model
    activation = 'relu'
    model = Sequential()
    model.add(TimeDistributed(Conv1D(512, strides=2, input_shape=x_train.shape[1:], activation=activation, kernel_size=4, padding='same')))
    model.add(TimeDistributed(MaxPooling1D(data_format='channels_first')))
    model.add(TimeDistributed(Conv1D(256, strides=1, activation=activation, kernel_size=2, padding='same')))
    model.add(TimeDistributed(MaxPooling1D(data_format='channels_first')))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(128, activation=activation)))
    model.add(TimeDistributed(Dense(128, activation=activation)))
    model.add(TimeDistributed(Dense(32, activation=activation)))
    model.add(TimeDistributed(Dense(32, activation=activation)))
    model.add(LSTM(20, return_sequences=False, name="lstm_layer"))
    model.add(Dense(numOfClasses, activation='softmax'))
    print(model.summary())
    opt = keras.optimizers.Adam(lr=0.004)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    #Enable early stopping and model saving
    os.chdir("/Users/scifa/Documents/ai_research")
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='acc', patience=1)
    ]

    #Run the model
    result = model.fit(x_train, y_train, verbose=1, epochs=50, batch_size=128, validation_data=(x_test, y_test), callbacks=callbacks_list, shuffle = True)

    #Generating Confusion Matrix
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)
    cm =confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    np.set_printoptions(threshold=np.nan, linewidth=100)
    print(cm)

main()
