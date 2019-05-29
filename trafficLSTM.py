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
from keras.layers import Dense, Flatten, GlobalAveragePooling1D, Dropout, Activation, TimeDistributed, LSTM, Input
from keras.utils import to_categorical
from random import shuffle
from collections import Counter
from sklearn.metrics import confusion_matrix


#Increases the print size of pandas output
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199





#Network hyperparameters
epochs = 4
learningRate = .0001
batch_size = 10
num_classes = 7


sfCutOff = 10 #number of timestamps per TimeDistribution

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

"""
Defines the 1D DCNN model using TimeDistributed
This model uses a concatenation layer to add in the time data for lstm
"""
def get_model_with_time():
    activation = 'relu'
    main_input = Input(shape = X_train.shape[1:], name='main_input')
    auxiliary_input = Input(shape = time_train.shape[1:], name='aux_input')

    x = TimeDistributed(Conv1D(256, (2), padding='same',  strides=1, name = "con1"))(main_input)
    x = TimeDistributed(MaxPooling1D())(x)
    x = TimeDistributed(Conv1D(128,(2), strides=1, activation=activation, name = "con2"))(x)
    x = TimeDistributed(MaxPooling1D())(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(64))(x)
    x = TimeDistributed(Dense(64))(x)
    x = TimeDistributed(Dense(14, activation=activation, name="den3"))(x)
    print(x)
    print(auxiliary_input)
    x = keras.layers.concatenate([x, auxiliary_input], axis=2)
    print(x)
    x = LSTM(50, return_sequences=False, dropout=0.5)(x)
    main_output = Dense(num_classes, activation = 'sigmoid', name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, main_output])
    return model


"""
Defines the 1D DCNN model using TimeDistributed
This model does not use time data, only the packets
"""
def get_model_without_time():
    activation = 'relu'
    main_input = Input(shape = X_train.shape[1:], name='main_input')

    x = TimeDistributed(Conv1D(256, (2), padding='same',  strides=1, name = "con1"))(main_input)
    x = TimeDistributed(MaxPooling1D())(x)
    x = TimeDistributed(Conv1D(128,(2), strides=1, activation=activation, name = "con2"))(x)
    x = TimeDistributed(MaxPooling1D())(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(64))(x)
    x = TimeDistributed(Dense(64))(x)
    x = TimeDistributed(Dense(14, activation=activation, name="den3"))(x)
    x = LSTM(50, return_sequences=False, dropout=0.5)(x)
    main_output = Dense(num_classes, activation = 'sigmoid', name='main_output')(x)
    model = Model(inputs=main_input, outputs=main_output)
    return model


def run_with_time():
    model = get_model_with_time()
    model.load_weights('my_model_weights.h5', by_name=True)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit([X_train, time_train], [y_train, y_train], epochs=epochs, batch_size=batch_size, verbose = 1)
    print(model.summary())
    benchmark_model_name = 'benchmark-model.h5'
    model.save(benchmark_model_name)
    print(model.evaluate([X_valid, time_valid], [y_valid, y_valid]))
    y_pred = classifier.predict([X_valid, time_valid])
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_valid, y_pred)
    print(cm)

def run_without_time():
    model = get_model_without_time()
    model.load_weights('my_model_weights.h5', by_name=True)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose = 1)
    print(model.summary())
    benchmark_model_name = 'benchmark-model.h5'
    model.save(benchmark_model_name)
    print(model.evaluate(X_valid, y_valid))
    y_pred = classifier.predict(X_valid)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_valid, y_pred)
    print(cm)


"""
    Load all data from hdf5 file and assign it to training and validation sets
"""
with h5.File('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/trafficData5.hdf5', 'r') as f:
    X_train = f["X_train"][:]
    y_train = f["y_train"][:]
    time_data = f["time"][:]

X_train, y_train, time_data = sklearn.utils.shuffle(X_train, y_train, time_data, random_state = 0)
X_valid = X_train[:50]
y_valid = y_train[:50]
X_train = X_train[50:]
y_train = y_train[50:]
time_train = time_data[50:]
time_valid = time_data[:50]
X_train = np.array(X_train)
X_valid = np.array(X_valid)
y_train = np.array(y_train)
y_valid = np.array(y_valid)
y_train = np_utils.to_categorical(y_train, num_classes)
y_valid = np_utils.to_categorical(y_valid, num_classes)
X_train = np.expand_dims(X_train, axis=3)
X_valid = np.expand_dims(X_valid, axis=3)
time_train = np.expand_dims(time_train, axis=3)
time_valid = np.expand_dims(time_valid, axis=3)
print(X_train.shape)
print(time_train.shape)


#Choose if you want to run the model with or without time concatenation
#run_without_time()
run_with_time()
