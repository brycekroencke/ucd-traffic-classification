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
from keras.layers import Dense, Flatten, GlobalAveragePooling1D, Dropout, Activation
from keras.utils import to_categorical
from random import shuffle


#Increases the print size of pandas output
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199



#Network hyperparameters
num_classes = 14
epochs = 10
learningRate = .005
batch_size = 64

"""
Switch uses to determine if the network should be trained on all data, or only data where
the class and subclss labels match.
1 -> Class and Subclass match
0 -> Class and Subclass do not have to match
"""
match = 0

nNumArr = []


"""
Loads datasets from the hdf5 file that were generated from toH5.py
"""
def load_data_kfold():
    X_train, y_train, allClass = [], [], []
    with h5.File('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/trafficData.hdf5', 'r') as f:
        if num_classes == 14:
            for i in range(10):
                X_train.append(f[str(i)+"data"][:])
                y_train.append(f[str(i)+"SClabel"][:])
        if num_classes == 7:
            for i in range(10):
                X_train.append(f[str(i)+"data"][:])
                y_train.append(f[str(i)+"Clabel"][:])
                allClass.append(f[str(i)+"SClabel"][:])
        if match == 1:
            ytemp, ytemp1 = [], []
            xtemp, xtemp1 = [], []
            for i in range(10):
                xtemp = []
                ytemp = []
                for j in range(len(y_train[i])):
                    if y_train[i][j] == allClass[i][j]:
                        xtemp.append(X_train[i][j])
                        ytemp.append(y_train[i][j])
                xtemp1.append(xtemp)
                ytemp1.append(ytemp)
            X_train = xtemp1
            y_train = ytemp1

        np.set_printoptions(threshold=np.nan)
        print(y_train)

    return X_train, y_train


"""
Defines the 1D DCNN model
"""
def get_model():
    activation = 'relu'
    model = Sequential()
    model.add(Conv1D(256, strides=1, input_shape=X_train_cv.shape[1:], activation=activation, kernel_size=2))
    model.add(MaxPooling1D())
    model.add(Conv1D(128, strides=1, activation=activation, kernel_size=2))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(64, activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(num_classes, activation=activation))
    model.add(Dense(num_classes, activation='softmax'))

    opt = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    return model



def get_callbacks(name_weights, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    early_stop = keras.callbacks.EarlyStopping(monitor='acc', patience=1)
    return [mcp_save, early_stop]

"""
Go from class label to an integer value.
"""

def gNum(type):
    # typeTuple = [("Google+",0),("GoogleEarth",1),("GoogleMap",2),("GoogleMusic",3),("GooglePlay",4),("Hangouts",5),("WebMail_Gmail",6),("YouTube",7),("Google_Common",8),("GoogleAnalytics",9),("GoogleSearch",10),("GoogleAdsense",11),("TCPConnect",12),("HTTP",13),("HTTPS",14)]
    typeTuple = [("GoogleEarth",0),("GoogleMap",1),("GoogleMusic",2),("GooglePlay",3),("Hangouts",4),("WebMail_Gmail",5),("YouTube",6),("Google_Common",7),("GoogleAnalytics",8),("GoogleSearch",9),("GoogleAdsense",10),("TCPConnect",11),("HTTP",12),("HTTPS",13)]
    dic = dict(typeTuple)
    return dic[type]


X_train, y_train = load_data_kfold()
predList = []
actList = []


"""
10 fold cross validation training.
"""
for j in range(10):
    print('\nFold ',j)


    """
    Assigns the remaining 1/10 of the dataset to validation. Every itteration of j
    would create a generate a different dataet.
    """
    X_valid_cv = X_train[j]
    y_valid_cv = y_train[j]

    """
    Assigns the remaining 9/10 of the dataset to the training.
    """
    X_train_cv = []
    y_train_cv = []
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


    """
    Augments the data into the proper form for the network to read.
    """
    X_train_cv, y_train_cv = sklearn.utils.shuffle(X_train_cv, y_train_cv, random_state = 0)
    X_train_cv = np.array(X_train_cv)
    X_valid_cv = np.array(X_valid_cv)
    y_train_cv = np.array(y_train_cv)
    y_valid_cv = np.array(y_valid_cv)
    y_train_cv = np_utils.to_categorical(y_train_cv, num_classes)
    y_valid_cv = np_utils.to_categorical(y_valid_cv, num_classes)
    X_train_cv = np.expand_dims(X_train_cv, axis=2)
    X_valid_cv = np.expand_dims(X_valid_cv, axis=2)


    print(X_train_cv.shape)
    name_weights = "final_model_fold" + str(j) + "_weights.h5"
    callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)

    """
    Training of the model.
    """
    model = get_model()
    model.fit(X_train_cv, y_train_cv, verbose=1, epochs=epochs, batch_size=batch_size, validation_data=(X_valid_cv, y_valid_cv), shuffle = True, callbacks = callbacks)
    print(model.evaluate(X_valid_cv, y_valid_cv))

    """
    Generates a confusion matrix for each fold of training. Helps understand networks predictive
    accuracy.
    """
    y_pred = model.predict(X_valid_cv)
    y_pred = (y_pred > 0.5)
    print(pd.crosstab(y_valid_cv.argmax(axis=1), y_pred.argmax(axis=1), rownames=['True'], colnames=['Predicted'], margins=True))
    print("\n\n")
    print(pd.crosstab(y_valid_cv.argmax(axis=1), y_pred.argmax(axis=1), rownames=['True'], colnames=['Predicted'], margins=True, normalize = 'columns', dropna = False))
    actList.append(y_valid_cv)
    predList.append(y_pred)
    print(predList)
    print(actList)

print(model.summary())

"""
Creates one final confusion matrix for the average results across all 10 folds of the
cross validation.
"""
predList = np.concatenate(predList, axis=0 )
actList = np.concatenate(actList, axis=0 )
print(predList)
print(pd.crosstab(actList.argmax(axis = 1), predList.argmax(axis = 1), rownames=['True'], colnames=['Predicted'], margins=True))
print("\n\n")
print(pd.crosstab(actList.argmax(axis = 1), predList.argmax(axis = 1), rownames=['True'], colnames=['Predicted'], margins=True, normalize = 'columns', dropna = False))
