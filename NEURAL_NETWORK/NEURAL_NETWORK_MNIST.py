# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:25:41 2020

@author: kaval
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse

#labelbinarizer will be used to one hot encode our integer label as vector labels
#one hot encoding transforms uniform categorical labels from single integer to a vector
#classification report gives us the nicely formatted report displaying the total accuracy of our model.
#the sequential class indicates that out network will be feedforward neural network with keras.
#the sequential class indicates that our network will be feedforward and layers will be added to 
#class sequentially,one on top of the other.
#the dense class is the implementation of fullyconnected layers

'''
ap = argparse.ArgumentPaser()
ap.add_argument("-o","--output",required=True,
                help="path to the output loss/accuracy plot")
args=vars(ap.parse_args())
'''

((trainx,trainy),(testx,testy)) = mnist.load_data()

#each image in mnist dataset is represented as a 28x28x1
#image,bur in order to apply standard neural network we must
#first "flatten" the image to be sample list of 28x28=784 pixels
trainx = trainx.reshape((trainx.shape[0],28*28*1))
testx = testx.reshape((testx.shape[0],28*28*1))

#scaling the data to range of [0,1]
trainx = trainx.astype("float32")/255.0
testx = testx.astype("float32")/255.0

#convert the labels from integers to vector
lb = LabelBinarizer()
trainy = lb.fit_transform(trainy)
testy= lb.fit_transform(testy)

#applying the one hot encoding each label integer converted into vector 
# example class label 3 is converted into [0,0,0,1,0,0,0,0,0,0]

#define the architecture 784-256-128-10 using keras
model = Sequential()
model.add(Dense(256,input_shape=(784,),activation='sigmoid'))
model.add(Dense(128,activation="sigmoid"))
model.add(Dense(10,activation="softmax"))

sgd = SGD(0.01)
model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=['accuracy'])
H=model.fit(trainx,trainy,validation_data=(testx,testy),epochs=100,batch_size=128)

predictions=model.predict(testx,batch_size=128)
print(classification_report(testy.argmax(axis=1),
                           predictions.argmax(axis=1),
                           target_names=[str(x) for x in lb.classes_]))

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