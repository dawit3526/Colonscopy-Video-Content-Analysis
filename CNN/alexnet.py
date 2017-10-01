# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
    Model Name:

        AlexNet - using the Functional Keras API

        Replicated from the Caffe Zoo Model Version.

    Paper:

         ImageNet classification with deep convolutional neural networks by Krizhevsky et al. in NIPS 2012

    Alternative Example:

        Available at: http://caffe.berkeleyvision.org/model_zoo.html

        https://github.com/uoguelph-mlrg/theano_alexnet/tree/master/pretrained/alexnet

    Original Dataset:

        ILSVRC 2012

"""
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.models import Model
from keras import regularizers
#from keras.utils.visualize_util import plot
#from KerasLayers.Custom_layers import LRN2D
from keras.layers.normalization import BatchNormalization
# global constants
NB_CLASS = 4         # number of classes
LEARNING_RATE = 0.01
MOMENTUM = 0.9
ALPHA = 0.0001
BETA = 0.75
GAMMA = 0.1
DROPOUT = 0.5
WEIGHT_DECAY = 0.0005
LRN2D_norm = True       # whether to use batch normalization
# Theano - 'th' (channels, width, height)
# Tensorflow - 'tf' (width, height, channels)
DIM_ORDERING = 'tf'


def conv2D_lrn2d(x, nb_filter, nb_row, nb_col,
                 border_mode='same', subsample=(1, 1),
                 activation='relu', LRN2D_norm=True,
                 weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORDERING):
    '''

        Info:
            Function taken from the Inceptionv3.py script keras github


            Utility function to apply to a tensor a module Convolution + lrn2d
            with optional weight decay (L2 weight regularization).
    '''
    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      bias=False,
                      dim_ordering=dim_ordering)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    if LRN2D_norm:
       
        x = BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)(x)
        x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    return x


def create_model():
    # Define image input layer
    if DIM_ORDERING == 'th':
        INP_SHAPE = (3, 224, 224)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 1
    elif DIM_ORDERING == 'tf':
        INP_SHAPE = (224, 224, 3)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 3
    else:
        raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))

    # Channel 1 - Convolution Net Layer 1
    x = conv2D_lrn2d(
        img_input, 48, 11,11, subsample=(
            1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            4, 4), pool_size=(
                4, 4), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 2
    x = conv2D_lrn2d(x, 256, 5, 5, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 3
    x = conv2D_lrn2d(x, 72, 5, 5, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 4
    x = conv2D_lrn2d(x, 1024, 6, 6, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

   
    # Channel 1 - Cov Net Layer 7
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    # Channel 1 - Cov Net Layer 8
    x = Dense(1024, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    # Final Channel - Cov Net 9
    x = Dense(output_dim=4,
              activation='softmax')(x)
    
    return x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING


def check_print():
    # Create the Model
    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()

    # Create a Keras Model - Functional API
    model = Model(input=img_input,
                  output=[x])
    model.summary()

    # Save a PNG of the Model Build
    #plot(model, to_file='./Model/AlexNet.png')

    model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])
    print('Model Compiled')
    return model
model = check_print()

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from keras.preprocessing import image
import glob
img_rows, img_cols = 224,224
image_size = (224, 224)
# number of channels
img_channels = 1

#%%
#  data


train_path ='C:/Users/pc/Desktop/Train'
labels= []
imlist=[]
train_labels = os.listdir(train_path) 
#label=np.ones((num_samples,),dtype = int) 

for i, label in enumerate(train_labels):
    cur_path = train_path + "/" + label
    for image_path in glob.glob(cur_path + "/*.jpg"):
        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        labels.append(i)
        imlist.append(x)
#imlist = os.listdir(path2)
#im1 = np.array(Image.open('/media/dawit/Data/saveImages' + '/'+ imlist[0])) # open one image to get size
#m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

immatrix = np.array([np.array(im2).flatten()
              for im2 in imlist],'f')
              #for im2 in imlist],'f')
#label=np.ones((num_samples,),dtype = int)
#
#label[0:44]=0
#label[44:437]=1
#label[437:]=2
data,Label = shuffle(immatrix,labels, random_state=2)
train_data = [data,Label]
img=immatrix[605].reshape(img_rows,img_cols)
plt.imshow(img)
plt.imshow(img,cmap='gray')
print (train_data[0].shape)
print (train_data[1].shape)
#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 4
# number of epochs to train
nb_epoch = 20

# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i = 100
plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i,:])

hist = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1,
               verbose=1, validation_data=(X_test, y_test))
