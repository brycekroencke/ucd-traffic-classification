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
import sys
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Flatten, GlobalAveragePooling1D, Dropout, Activation
from keras.utils import to_categorical
from random import shuffle
import pylab as plt


#Increases the print size of pandas output
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

np.set_printoptions(threshold=sys.maxsize)


#Network hyperparameters
epochs = 10
learningRate = .001  #.005
batch_size = 128    #64

"""
Switch uses to determine if the network should be trained on all data, or only data where
the class and subclss labels match.
1 -> Class and Subclass match
0 -> Class and Subclass do not have to match
"""
match = 0
nNumArr = []
int_to_subclass= []

"""
Loads datasets from the hdf5 file that were generated from toH5.py
"""
def load_data_kfold():
    X_train, y_train, allClass, total = [], [], [], []
    with h5.File('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/trafficData_large_cnn.hdf5', 'r') as f:
        # if num_classes == 14:
        #     X_train = f["X_train"][:]
        #     y_train = f["y_train"][:]
        # if num_classes == 7:
        #     X_train=f["X_train"][:]
        #     y_train=f["y_train"][:]
        #     allClass=f["y_train_sc"][:]
        X_train=f["X_train"][:]
        y_train=f["y_train"][:]
        allClass=f["y_train_sc"][:]
        int_to_class=f["class_strings"][:]
        int_to_subclass=f["class_strings"][:]
        print(y_train)
        num_classes = len(set(y_train))
        int_to_class = [n[0].decode('utf-8') for n in int_to_class]
        print(int_to_class)
        if match == 1:
            ytemp, ytemp1 = [], []
            xtemp, xtemp1 = [], []
            for j in range(len(y_train)):
                if y_train[j] == allClass[j]:
                    xtemp.append(X_train[j])
                    ytemp.append(y_train[j])
            X_train = xtemp
            y_train = ytemp


    return X_train, y_train, num_classes, int_to_class


"""
Defines the 1D DCNN model
"""
def get_model():
    activation = 'relu'
    model = Sequential()
    model.add(Conv1D(256, strides=1, input_shape=X_train_cv.shape[1:], activation=activation, kernel_size=2, name = "con1"))
    model.add(MaxPooling1D())
    model.add(Conv1D(128, strides=1, activation=activation, kernel_size=2,name = "con2"))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(64, activation=activation,name = "den1"))
    model.add(Dense(64, activation=activation,name = "den2"))
    model.add(Dense(num_classes, activation=activation,name = "den3"))
    model.add(Dense(num_classes, activation='softmax',name = "den4"))

    opt = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    return model



def get_callbacks(name_weights, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    early_stop = keras.callbacks.EarlyStopping(monitor='acc', patience=1)
    return [mcp_save, early_stop]



X_train, y_train, num_classes, int_to_class = load_data_kfold()
X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state = 0)
predList = []
actList = []

size_of_val = int(len(X_train) * .3)


X_valid_cv = X_train[:size_of_val]
y_valid_cv = y_train[:size_of_val]
X_train_cv = X_train[size_of_val:]
y_train_cv = y_train[size_of_val:]



"""
Augments the data into the proper form for the network to read.
"""
#X_train_cv, y_train_cv = sklearn.utils.shuffle(X_train_cv, y_train_cv, random_state = 0)
X_train_cv = np.array(X_train_cv)
X_valid_cv = np.array(X_valid_cv)
y_train_cv = np.array(y_train_cv)
y_valid_cv = np.array(y_valid_cv)
y_train_cv = np_utils.to_categorical(y_train_cv, num_classes)
y_valid_cv = np_utils.to_categorical(y_valid_cv, num_classes)
X_train_cv = np.expand_dims(X_train_cv, axis=2)
X_valid_cv = np.expand_dims(X_valid_cv, axis=2)


print(X_train_cv.shape)
name_weights = 'weights_large_cnn.h5'
callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)

"""
Training of the model.
"""
model = get_model()
#model.load_weights("/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/ucd-traffic-classification/final_model_fold0_weights.h5")
# model.save_weights('test_weights.h5')

model.fit(X_train_cv, y_train_cv, verbose=1, epochs=epochs, batch_size=batch_size, validation_data=(X_valid_cv, y_valid_cv), shuffle = True, callbacks = callbacks)
print(model.evaluate(X_valid_cv, y_valid_cv))

"""
Generates a confusion matrix for each fold of training. Helps understand networks predictive
accuracy.
"""
y_pred = model.predict(X_valid_cv)
y_pred = np.argmax(y_pred, axis=1)
y_valid_cv = np.argmax(y_valid_cv, axis=1)
total = 0
# for i in range(len(y_pred)):
#     pred_int = int(y_pred[i])
#     act_int = int(y_valid_cv[i])
#     if act_int == pred_int:
#         total = total + 1
#         print(int_to_class[act_int], "        ", int_to_class[pred_int], "TRUE    *******")
#     else:
#         print(int_to_class[act_int], "        ", int_to_class[pred_int], "FALSE")


predicted_class = []
for i in y_pred:
    predicted_class.append(int_to_class[i])

actual_class = []
for i in y_valid_cv:
    actual_class.append(int_to_class[i])

cm = confusion_matrix(actual_class, predicted_class, int_to_class)




print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticks(np.arange(len(int_to_class)))
ax.set_yticks(np.arange(len(int_to_class)))
ax.set_xticklabels([''] + int_to_class)
ax.set_yticklabels([''] + int_to_class)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# print("\n\n")
# print(pd.crosstab(y_valid_cv, y_pred, rownames=['True'], colnames=['Predicted'], margins=True, normalize = 'columns', dropna = False))
# print(y_pred.argmax(axis = 1))
# print(y_valid_cv.argmax(axis = 1))
#
#
#
# print(model.summary())
#
# """
# Creates one final confusion matrix for the average results across all 10 folds of the
# cross validation.
# """
# print(pd.crosstab(X_valid_cv.argmax(axis = 1), y_pred.argmax(axis = 1), rownames=['True'], colnames=['Predicted'], margins=True))
# print("\n\n")
# print(pd.crosstab(X_valid_cv.argmax(axis = 1), y_pred.argmax(axis = 1), rownames=['True'], colnames=['Predicted'], margins=True, normalize = 'columns', dropna = False))
