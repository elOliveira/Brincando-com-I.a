# Image Super Resolution Using Autoencoders in Keras

# task 1 Import lib

import os
import re
from scipy import ndimage, misc
from skimage.transform import resize, rescale
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf
print(tf.__version__)


# task 2  what are auutoencoders?

# task 3 build an autoenconder

input_img = Input(shape = (256,256,3))
l1 = Conv2D(64, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(input_img)

l2 = Conv2D(64, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(l1)

l3 = MaxPooling2D(padding = 'same') (l2)

l4 = Conv2D(128, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(l3)

l5 = Conv2D(128, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(l4)

l6 = MaxPooling2D(padding = 'same') (l5)

l7 = Conv2D(256, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(l6)

encoder = Model(input_img , 7)


encoder.summary()


# task 4 Build the decoder to complete the network

input_img = Input(shape = (256,256,3))
l1 = Conv2D(64, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(input_img)

l2 = Conv2D(64, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(l1)

l3 = MaxPooling2D(padding = 'same') (l2)

l4 = Conv2D(128, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(l3)

l5 = Conv2D(128, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(l4)

l6 = MaxPooling2D(padding = 'same') (l5)

l7 = Conv2D(256, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(l6)

l8 = UpSampling2D()(l7)

l9 = Conv2D(256, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(l8)
    
l10 = Conv2D(256, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(l9)

l11 = add([l5,l10])

l12 = UpSampling2D()(l11)

l13 = Conv2D(64, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(l12)

l14 = Conv2D(64, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(l13)

l15 = add([l14, l14])

decoded = Conv2D(3, (3,3),
    padding = 'same' ,
    activation = 'relu',
    activity_regularizer =  regularizers.l1(10e-10))(l15)

autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer = 'adadelta', loss = 'mean_squared_error')

#task 5 create dataset and specify  training routine

def train_batches( just_load_dataset = False):

    batches = 256

    batch = 0
    batch_nb = 0

    max_batches = -1
    ep = 4

    images = []
    x_train_n = []
    x_train_down = []

    x_train_n2 = []
    x_train_down2 = []

    for root, dirnames, filenames in os.walk("/home/rhyme/Desktop/Project/data/cars_train"):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|JPEG|png|bmp|tiff)$", filename):

                if batch_nb == max_batches:
                    return x_train_n2 , x_train_down2
                
                filepath = os.path.join(root,filename)
                image = pyplot.imread(filepath)
                
                if len(image.shape) > 2 :

                    image_resized = resize(image, (256,256))
                    x_train_n.append(image_resized)
                    x_train_down.append(rescale(rescale(image_resized, 0.5) , 2.0))
                    batch+= 1

                    if batch == batches:
                        batch_nb += 1

                        x_train_n2 = np.array(x_train_n)
                        x_train_down2 = np.array(x_train_down)

                        if just_load_dataset:
                            return x_train_n2, x_train_down2

                        print('Training batch', batch_nb, '(', batches , ')' )

                        autoencoder.fit(x_train_down2, x_train_n2,
                            epochs = ep,
                            batch_size = 10,
                            shuffle = True,
                            validation_split = 0.15
                         )

                        x_train_n = []
                        x_train_down = []

                        batch = 0

    return x_train_n2, x_train_down2


# Task 6 Load The Dataset and pre trained model

x_train_n, x_train_down = train_batches(just_load_dataset = True )

autoencoder.load_weights('/home/rhyme/Desktop/Project/data/sr.img_net.mse.final_model5.no_ALGUMACOISAAQUI')

# Task 7 Model predictions and visualizing the results

encoder.load_wheights('/home/rhyme/Desktop/Project/data/encoder_weights.hdf5')
encoded_imgs = encoder.predict(x_train_down)
encoded_imgs.shape

sr1 = np.clip(autoencoder.predict(x_train_down), 0.0,1.0)

image_index = 251 # np.random.randint(0.256)

plt.figure(figsize = (128,128))
i = 1

ax = plt.subplot(10,10,i)
plt.imshow(x_train_down[image_index])
i += 1

ax = plt.subplot(10,10,i)
plt.imshow(x_train_down[image_index], interpolation = 'bicubic')
i += 1

ax = plt.subplot(10,10,i)
plt.imshow(encoded_imgs[image_index].reshape((64*64,256)))
i += 1

ax = plt.subplot(10,10,i)
plt.imshow(sr1[image_index])
i += 1

ax = plt.subplot(10,10,i)
plt.imshow(x_train_n[image_index])
plt.show()
