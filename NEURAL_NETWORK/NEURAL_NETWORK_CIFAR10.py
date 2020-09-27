# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:20:40 2020

@author: kaval
"""
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse

'''
ap = argparse.ArgumentPaser()
ap.add_argument("-o","--output",required=True,
                help="path to the output loss/accuracy plot")
args=vars(ap.parse_args())
'''

((trainx,trainy),(testx,testy)) = cifar10.load_data()

#each image in cifar10 dataset is represented as a 32*32*3
#image,bur in order to apply standard neural network we must
#first "flatten" the image to be sample list of 32*32*3=3072 pixels
trainx = trainx.reshape((trainx.shape[0],32*32*3))
testx = testx.reshape((testx.shape[0],32*32*3))

#scaling the data to range of [0,1]
trainx = trainx.astype("float32")/255.0
testx = testx.astype("float32")/255.0

#convert the labels from integers to vector
lb = LabelBinarizer()
trainy = lb.fit_transform(trainy)
testy = lb.fit_transform(testy)

#applying the one hot encoding each label integer converted into vector 
# example class label 3 is converted into [0,0,0,1,0,0,0,0,0,0]
model = Sequential()
model.add(Dense(1024,input_shape=(3072,),activation="relu"))
model.add(Dense(512,activation="relu"))
model.add(Dense(10,activation="softmax"))

sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy",optimizer = sgd,metrics = ["accuracy"])

H=model.fit(trainx,trainy,validation_data=(testx,testy),epochs=100,batch_size=32)

labelnames = ['airplane','automobile','bird','cat','deer','dog',
              'forge','horse','ship','truck']

predictions = model.predict(testx,batch_size=32)
print(classification_report(testy.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelnames))

'''plt.style.use("ggplot")'''
plt.figure()
plt.plot(np.arange(0,100),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,100),H.history["val_loss"],label="train_loss")
plt.plot(np.arange(0,100),H.history["accuracy"],label="train_acc")
plt.plot(np.arange(0,100),H.history["val_accuracy"],label="val_acc")
plt.title("training loss and accuracy")
plt.xlabel("epochs #")
plt.ylabel("loss/accuracy")
plt.legend() 
'''plt.savefig(args["output"])'''
