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
from keras.layers import Conv2D, MaxPooling2D
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
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


global samples_per_train
samples_per_train = 50
global cur_train_index
cur_train_index = 0
global batch_sz
batch_sz = 50
path = "D:\\nkasturi122817\\signatureverification\\NISDCC-offline-all-001-051-6g\\"
dir = "D:\\nkasturi122817\\signatureverification\\NISDCC-offline-all-001-051-6g"

def next_train():
    while True:
        global cur_train_index
        print(samples_per_train)
        if cur_train_index == samples_per_train:
            cur_train_index = 0

        temp_cur_train_index = cur_train_index + batch_sz

        if temp_cur_train_index > samples_per_train:
            temp_cur_train_index = samples_per_train
        selected_pairs = pairs.iloc[cur_train_index:temp_cur_train_index, :]
        #print("cur_train_index " + str(cur_train_index) + "selected_pairs.names" + str(selected_pairs.names))
        image_pairs = []
        label_pairs = []

        for index, rows in selected_pairs.iterrows():
            #print(index)
            img1 = preprocess_image(glob.glob(path + "NISDCC-" + rows.names + "_6g.PNG")[0])
            # print(img1)

            img1 = prep_image.img_to_array(img1)  # , dim_ordering='tf')

            img2 = preprocess_image(glob.glob(path + "NISDCC-" + rows.names_2 + "_6g.PNG")[0])

            img2 = prep_image.img_to_array(img2)  # , dim_ordering='tf')

            image_pairs += [[img1, img2]]
            label_pairs += [int(rows.Label)]

        cur_train_index = temp_cur_train_index

        images = [np.array(image_pairs)[:, 0], np.array(image_pairs)[:, 1]]
        labels = np.array(label_pairs)
        yield (images, labels)


def next_valid():
    while True:
        global cur_train_index
        if cur_train_index == samples_per_train:
            cur_train_index = 0

        temp_cur_train_index = cur_train_index + batch_sz

        if temp_cur_train_index > samples_per_train:
            temp_cur_train_index = samples_per_train
        selected_pairs = pairs.iloc[cur_train_index:temp_cur_train_index, :]
        image_pairs = []
        label_pairs = []

        for index, rows in selected_pairs.iterrows():
            img1 = preprocess_image(glob.glob(path + "NISDCC-" + rows.names + "_6g.PNG")[0])
            #print(img1)

            img1 = prep_image.img_to_array(img1)  # , dim_ordering='tf')

            img2 = preprocess_image(glob.glob(path + "NISDCC-" + rows.names_2 + "_6g.PNG")[0])

            img2 = prep_image.img_to_array(img2)  # , dim_ordering='tf')

            image_pairs += [[img1, img2]]
            label_pairs += [int(rows.Label)]

        cur_train_index = temp_cur_train_index

        images = [np.array(image_pairs)[:, 0], np.array(image_pairs)[:, 1]]
        labels = np.array(label_pairs)
#        print(images.shape)
        yield (images, labels)


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
    model.add(Dense(128, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform'))  # softmax changed to relu

    return model


#if __name__ == '__main__':

input_shape = (62,128,1)
params = {'dim': input_shape,
          'batch_size': 64,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}
pairs, train_IDs, valid_IDs = pairs_generator('D:\\nkasturi122817\\signatureverification\\NISDCC-offline-all-001-051-6g\\data.csv')
partition = {'train':train_IDs,'validation': valid_IDs}
training_generator = DataGenerator(partition['train'], pairs, path, **params)
validation_generator = DataGenerator(partition['validation'], pairs, path, **params)
base_network = create_base_network_signet(input_shape)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
processed_a = base_network(input_a)
processed_b = base_network(input_b)
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = Model(input=[input_a, input_b], output=distance)
rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
adadelta = Adadelta()
model.compile(loss=contrastive_loss, optimizer=rms)

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)
# model.save(os.path.join(dir,"model.h5"))
#model = load_model(os.path.join(dir,"model.h5"))
# for index, rows in pairs.iterrows():
#     img1 = preprocess_image(glob.glob(path + "NISDCC-" + rows.names + "_6g.PNG")[0])
#     # print(img1)
#
#     img1 = prep_image.img_to_array(img1)  # , dim_ordering='tf')
#
#     img2 = preprocess_image(glob.glob(path + "NISDCC-" + rows.names_2 + "_6g.PNG")[0])
#
#     img2 = prep_image.img_to_array(img2)  # , dim_ordering='tf')
prediction = model.predict_generator(generator=next_train(), steps=2, max_queue_size=10, workers=1, use_multiprocessing=True, verbose=0)
print(prediction)
#pairs.iloc[index,8] = model.predict([img1,img2])
#pairs.iloc[index, 9] = model.predict(img2)
# pairs.to_csv('test.csv')
