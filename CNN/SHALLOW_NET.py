# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:31:33 2020

@author: kaval
"""
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

class shallownet:
    def build(width,height,depth,classes):
        #initialize the model along with input shape to be channels last
        model = Sequential()
        inputshape = (height,width,depth)
        
        if K.image_data_format() == "channels_first":
            inputshape = (depth,height,width)
        model.add(Conv2D(32,(3,3),padding="same"))
        model.add(Activation("relu"))
        #we apply same padding to ensure the size of output of convolution operation matches input
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model