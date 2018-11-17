import glob
import os
import random

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

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

#from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import random
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from pairs_generator import pairs_generator
from data_generator import DataGenerator
from preprocess_image import preprocess_image
# Parameters

def contrastive_loss(y_true, y_pred):
    margin = 1
    print(K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))))
    return (K.mean((1-y_true) * K.square(y_pred) + (y_true) * K.square(K.maximum(margin - y_pred, 0))))


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)







def create_base_network_signet(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (5, 13), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Conv2D(64, (3, 7), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="sigmoid"))
    # convolutional_net = Sequential()
    # convolutional_net.add(Conv2D(filters=64, kernel_size=(10, 10),
    #                              activation='relu',
    #                              input_shape=input_shape,
    #                              name='Conv1'))
    # convolutional_net.add(MaxPool2D())
    #
    # convolutional_net.add(Conv2D(filters=128, kernel_size=(7, 7),
    #                              activation='relu',
    #                              name='Conv2'))
    # convolutional_net.add(MaxPool2D())
    #
    # convolutional_net.add(Conv2D(filters=128, kernel_size=(4, 4),
    #                              activation='relu',
    #                              name='Conv3'))
    # convolutional_net.add(MaxPool2D())
    #
    # # convolutional_net.add(Conv2D(filters=256, kernel_size=(4, 4),
    # #                              activation='relu',
    # #                              name='Conv4'))
    # # convolutional_net.add(MaxPool2D())
    #
    # convolutional_net.add(Flatten())
    # convolutional_net.add(
    #     Dense(units=4096, activation='relu',
    #           name='Dense1'))
    # convolutional_net.add(
    #     Dense(units=128, activation='sigmoid',
    #           name='Dense11'))
    return model


def compute_accuracy_roc(predictions, labels):
    '''Compute ROC accuracy with a range of thresholds on distances.
    '''
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)

    step = 0.01
    max_acc = 0

    for d in np.arange(dmin, dmax + step, step):
        idx1 = predictions.ravel() <= d
        idx2 = predictions.ravel() > d

        tpr = float(np.sum(labels[idx1] == 1)) / nsame
        tnr = float(np.sum(labels[idx2] == 0)) / ndiff
        acc = 0.5 * (tpr + tnr)
        #       print ('ROC', acc, tpr, tnr)

        if (acc > max_acc):
            max_acc = acc

    return max_acc

if __name__ == '__main__':
    path = "D:\\nkasturi122817\\signatureverification\\NISDCC-offline-all-001-051-6g\\"
    dir = "D:\\nkasturi122817\\signatureverification\\NISDCC-offline-all-001-051-6g"

    input_shape = (62,128,1)
    params = {'dim': input_shape,
              'batch_size': 200,
              'n_classes': 2,
              'n_channels': 1,
              'shuffle': True}
    pairs, train_IDs, valid_IDs = pairs_generator('D:\\nkasturi122817\\signatureverification\\NISDCC-offline-all-001-051-6g\\data.csv')
    partition = {'train':train_IDs,'validation': valid_IDs}
    training_generator = DataGenerator(partition['train'], pairs, path, **params)
    #print(training_generator.labels)
    validation_generator = DataGenerator(partition['validation'], pairs, path, **params)
    base_network = create_base_network_signet(input_shape)
    print(base_network.summary())
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model(input=[input_a, input_b], output=distance)
    rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
    adadelta = Adadelta()
    model.compile(loss=contrastive_loss, optimizer=rms)
    fname = os.path.join('D:\\nkasturi122817\\signatureverification\\NISDCC-offline-all-001-051-6g', 'weights_NISDCC.hdf5')
    checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
    # history = model.fit_generator(generator=training_generator,
    #                     validation_data=validation_generator,
    #                     use_multiprocessing=True,
    #                     workers=6,  callbacks=[checkpointer], epochs=5 )
    # print(history.history)
    model.load_weights(fname)
    print('Loading Best Weights')

    tr_pred = model.predict_generator(generator=validation_generator)
    # print(tr_pred)
    with open('result.txt', 'w') as f:
        f.write(str(tr_pred))
        f.close()
