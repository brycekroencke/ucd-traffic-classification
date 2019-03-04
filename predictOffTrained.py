from keras.models import load_model
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


model0 = load_model('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/ucd-traffic-classification/final_model_fold0_weights.h5')
model1 = load_model('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/ucd-traffic-classification/final_model_fold1_weights.h5')
model2 = load_model('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/ucd-traffic-classification/final_model_fold2_weights.h5')
model3 = load_model('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/ucd-traffic-classification/final_model_fold3_weights.h5')
model4 = load_model('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/ucd-traffic-classification/final_model_fold4_weights.h5')
model5 = load_model('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/ucd-traffic-classification/final_model_fold5_weights.h5')
model6 = load_model('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/ucd-traffic-classification/final_model_fold6_weights.h5')
model7 = load_model('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/ucd-traffic-classification/final_model_fold7_weights.h5')
model8 = load_model('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/ucd-traffic-classification/final_model_fold8_weights.h5')
model9 = load_model('/Users/brycekroencke/Documents/TrafficClassification/Project Related Files/ucd-traffic-classification/final_model_fold9_weights.h5')
