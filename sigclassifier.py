import glob
import os
import random

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
np.random.seed(123)
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras.layers import Input, concatenate
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image as prep_image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


#from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import random
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from pairs_generator import pairs_generator
from data_generator import DataGenerator
from preprocess_image import preprocess_image


def integer_encode(X, Y):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    Y_oh = onehot_encoder.transform(integer_encoded.reshape(len(label_encoder.transform(Y)), 1))
    X_train, X_test, Y_train_oh, Y_test_oh = train_test_split(X, Y_oh, test_size=0.0, random_state=1)

    return [X_train, X_test, Y_train_oh, Y_test_oh, label_encoder]
input_shape = (155,220,1)
NIS_images = glob.glob('D:/nkasturi122817/axisbank/signatureverification/NISDCC-offline-all-001-051-6g/'+"*.png")
CEDAR_images = glob.glob('D:\\nkasturi122817\\axisbank\\GML-master\\signatures\\all\\'+"*.png")
print(CEDAR_images)
i = 0
X = np.empty((len(NIS_images)+len(CEDAR_images), *input_shape))
y = np.empty(len(NIS_images)+len(CEDAR_images), dtype=int)
for im_path in NIS_images:
    img = preprocess_image(im_path)
    img = prep_image.img_to_array(img)
    X[i,] = img
    y[i] = int(0)
    i = i+1
for im_path in CEDAR_images:
    img = preprocess_image(im_path)
    img = prep_image.img_to_array(img)
    X[i,] = img
    y[i] = int(1)
    i = i+1
X_train, X_test, Y_train_oh, Y_test_oh, label_encoder = integer_encode(X, y)
print(y)

lr = 0.0004
num_classes = len(set(y))
seq = Sequential()
seq.add(Convolution2D(96, (11, 11), activation='relu', name='conv1_1', strides=(4, 4), input_shape=input_shape,
                      kernel_initializer='glorot_uniform', dim_ordering='tf'))
seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=3, momentum=0.9))
seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
seq.add(ZeroPadding2D((2, 2), dim_ordering='tf'))

seq.add(Convolution2D(256, (5, 5), activation='relu', name='conv2_1', strides=(1, 1),
                      kernel_initializer='glorot_uniform', dim_ordering='tf'))
seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=3, momentum=0.9))
seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
seq.add(Dropout(0.3))  # added extra
seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))

seq.add(Convolution2D(384, (3, 3), activation='relu', name='conv3_1', strides=(1, 1),
                      kernel_initializer='glorot_uniform', dim_ordering='tf'))
seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))

seq.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_2', strides=(1, 1),
                      kernel_initializer='glorot_uniform', dim_ordering='tf'))
seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
seq.add(Dropout(0.3))  # added extra
#    model.add(SpatialPyramidPooling([1, 2, 4]))
seq.add(Flatten(name='flatten'))
seq.add(Dense(1024, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
seq.add(Dropout(0.5))

seq.add(Dense(128, kernel_regularizer=l2(0.0005), activation='relu',
              kernel_initializer='glorot_uniform'))  # softmax changed to relu
seq.add(Dropout(0.5))
seq.add(Dense(num_classes, activation='sigmoid'))


print(seq.summary())

path = "D:\\nkasturi122817\\signatureverification\\NISDCC-offline-all-001-051-6g\\"
dir = "D:\\nkasturi122817\\signatureverification\\NISDCC-offline-all-001-051-6g"

rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
adadelta = Adadelta()
seq.compile(loss='categorical_crossentropy', optimizer=rms, metrics= ['accuracy'])
#fname = os.path.join('D:\\nkasturi122817\\axisbank\\Axis_SigVerify-master', 'model_prep_acc.h5')
#seq.load_weights(fname)
#checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
#tbpointer = TensorBoard(log_dir='/graph', histogram_freq=0, write_graph=True, write_images=True)

seq.fit(X_train,Y_train_oh,  epochs = 2, batch_size = 32, shuffle=True)
seq.save( "sigclassifier.h5")
seq = load_model("sigclassifier.h5")
scores = seq.evaluate(X, y, verbose=0)
print(scores)
print("%s: %.2f%%" % (seq.metrics_names[1], scores[1]*100))
#print(X_train)
# scores = seq.evaluate(X_test, Y_test_oh, verbose=0)
# print("%s: %.2f%%" % (seq.metrics_names[1], scores[1]*100))
print(seq.predict(X))
