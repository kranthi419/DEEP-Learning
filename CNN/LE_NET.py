# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:52:52 2020

@author: kaval
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K

class lenet:
    def build(width,height,depth,classes):
        model = Sequential()
        inputshape = (height,width,depth)
        
        #if using channels first update the input shape
        if K.image_data_format()=="channels_first":
            inputshape = (depth,height,width)
        model.add(Conv2D(20,(5,5),padding='same',input_shape=inputshape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(20,(5,5),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model 