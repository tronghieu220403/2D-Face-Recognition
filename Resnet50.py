
import numpy as np
import random

import keras

from keras.initializers import glorot_uniform
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Normalization  
from keras.layers import Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from matplotlib.pyplot import imshow


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure   

    Parameters
    ----------
    X : tensor
        input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f : integer
        specifying the shape of the middle CONV's window for the main path
    filters : list
        python list of integers, defining the number of filters in the CONV layers of the main path
    stage : integer
        used to name the layers, depending on their position in the network
    block : str
        used to name the layers, depending on their position in the network

    Returns
    -------
    X : tensor
        output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. we'll need this later to add back to the main path. 
    X_shortcut = X

    # First component of main path
    X = keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
               name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=101))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', 
               name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=101))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)

    # Third component of main path
    X = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=101))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, stride=2):
    """
    Implementation of the convolutional block as defined in Figure   

    Parameters
    ----------
    X : tensor
        input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f : integer
        specifying the shape of the middle CONV's window for the main path
    filters : list
        python list of integers, defining the number of filters in the CONV layers of the main path
    stage : integer
        used to name the layers, depending on their position in the network
    block : str
        used to name the layers, depending on their position in the network
    s : integer, optional
        Integer, specifying the stride to be used. The default is 2.

    Returns
    -------
    X : tensor
        output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path 
    X = keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(stride, stride), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=101))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=101))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=101))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(stride, stride), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=101))(X_shortcut)
    X_shortcut = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    return X

def ResNet50(input_shape, outputClasses):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Parameters
    ----------
    input_shape : tuple, optional
        shape of the input image. 
    outputClasses : integer, optional
        number of classes. 

    Returns
    -------
    model : object
        a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = keras.layers.ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=101))(X)
    X = keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = keras.layers.Activation('relu')(X)

    # Stage 2
    
    X = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', stride=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', stride=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', stride=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', stride=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL 
    X = keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # output layer
    X = keras.layers.Flatten()(X)
    
    X = keras.layers.Dense(512, activation='relu', name='fc_rd1', 
            kernel_initializer=glorot_uniform(seed=101))(X)
    
    
    X = keras.layers.Dense(128, activation='sigmoid', name='fc_rd2', 
            kernel_initializer=glorot_uniform(seed=101))(X)
    
    
    X = keras.layers.Dense(outputClasses, activation='softmax', name='fc' + str(outputClasses), 
            kernel_initializer=glorot_uniform(seed=101))(X)

    # Create model
    model = keras.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model



