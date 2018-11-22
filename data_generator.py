import glob
import keras
import numpy as np
from preprocess_image import preprocess_image
from keras.preprocessing import image as prep_image
import numpy as np

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, path, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.path = path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        image_pairs = []
        label_pairs = []
        X1 = np.empty((self.batch_size, *self.dim))
        X2 = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            data =  self.labels[self.labels['ID']==ID]
            img1_code = data['names']
            img2_code = data['names_2']
            label = data['Label']
            #print(img1_code)
            img1 = preprocess_image((self.path + img1_code + ".png").values[0])
            #print(img1)

            img1 = prep_image.img_to_array(img1)  # , dim_ordering='tf')

            img2 = preprocess_image((self.path + img2_code + ".png").values[0])

            img2 = prep_image.img_to_array(img2)  # , dim_ordering='tf')
            image_pairs += [[img1, img2]]
            label_pairs += [int(label)]
            X1[i,] = img1
            X2[i,] = img2

            # Store class
            y[i] = int(label)

        return [X1,X2] , y
