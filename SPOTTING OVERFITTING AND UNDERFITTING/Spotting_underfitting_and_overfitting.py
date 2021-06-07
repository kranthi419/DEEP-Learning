# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 00:52:50 2020

@author: kaval
"""
from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self,figpath=None,jsonpath=None,startat=0):
        # store the output path for figure, the path to json
        # serialized file, and the starting epoch
        super(TrainingMonitor,self).__init__()
        self.figpath = figpath
        self.jsonpath = jsonpath 
        self.startat = startat 
        
        
        
        
        
        
        
    def on_train_begin(self,logs ={}):
        #initialize the binary dictionary
        self.H = {}
         #if the json history path exists, load the training history
        if self.jsonpath is not None:
            if os.path.exists(self.jsonpath):
                self.H = json.loads(open(self.jsonpath).read())
                 
                 #check to see if a starting epoch was supplied
                if self.startat > 0:
                     #loop over the entries in history log and trim any entries
                     # that are past the starting epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startat]
                         
    def on_epoch_end(self,epoch,logs={}):
        #loop over the logs and update the loss, accuracy etx
        #for the entire training process
        for (k,v) in logs.items():
            l = self.H.get(k,[])
            l.append(float(v))
            self.H[k]=l
            
        #check to see the training history should be serialized to file
        if self.jsonpath is not None:
            f  = open(self.jsonpath,"w")
            f.write(json.dumps(self.H))
            f.close()
            
        #ensure at least two epochs have passed before plotting
        if len(self.H["loss"])>1:
            #plot the training loss and accuracy
            N = np.arange(0,len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N,self.H["loss"],label="train_loss")
            plt.plot(N,self.H["val_loss"],label="val_loss")
            plt.plot(N,self.H["accuracy"],label="train_acc")
            plt.plot(N,self.H["val_accuracy"],label="val_acc")
            plt.title("training loss and accuracy[Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("loss/Accuracy")
            plt.legend()
            
            plt.savefig(self.figpath)
            plt.close()
import matplotlib
matplotlib.use("Agg")

#import the necessary packages 
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o","--output",required = True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("[info] process id: {}".format(os.getpid()))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as k

class MiniVGGNet:
    @staticmethod
    def build(width,height,depth,classes):
        #initialize the model along with input shape to be channels last and the channels dimension itself
        model = Sequential()
        inputShape = (height,width,depth)
        chanDim = -1
        
        #if we r using channel first,update the input shape and channels dimension
        if k.image_data_format() == "channels_first":
            inputShape = (depth,heigth,width)
            chanDim = 1
            
        #first conv => relu => con => relu => pool layer set
        model.add(Conv2D(32,(3,3),padding='same',input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        #Second conv => relu => con => relu => pool layer set
        model.add(Conv2D(64,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        # first(and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        #softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        #return the constructed network architecture
        return model
    
    
print("[INFO] LOADING CIFAR-10 DATA...")
((trainx,trainy),(testx,testy)) = cifar10.load_data()

trainx = trainx.astype("float")/255.0
testx = testx.astype("float")/255.0

#convert the label from integer to vectors
lb = LabelBinarizer()
trainy = lb.fit_transform(trainy)
testy = lb.transform(testy)

#intialize the label names for cifar10 dataset
LabelNames = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


print('[INFO] compiling the model...')
opt = SGD(lr=0.01,momentum=0.9,nesterov=True)
model = MiniVGGNet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss='categorical_crossentropy',optimizer=opt,
             metrics=['accuracy'])

figpath = os.path.sep.join([args["output"],"{}.png".format(os.getpid())])      
jsonpath = os.path.sep.join([args["output"],"{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figpath,jsonpath=jsonpath)]

print('[INFO] training network...')
H = model.fit(trainx,trainy,validation_data=(testx,testy),batch_size=64,epochs=40,callbacks=callbacks,verbose=1)  