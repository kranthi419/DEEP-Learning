# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:40:29 2020

@author: kaval
"""
#import matplotlib
'''matplotlib.use("Agg")
'''
import matplotlib
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from CNN.MINI_VGG_NET import MiniVGGNet
import matplotlib.pyplot as plt
import numpy as np
import argparse

def step_decay(epoch):
    #initialize  the base learning rate,drop factor and epochs to drop
    initAlpha = 0.01
    factor = 0.5
    dropevery = 5
    
    #compute learning rate for current epoch
    alpha = initAlpha*(factor**np.floor((1+epoch)/dropevery))
    #return learning rate
    return float(alpha)
    
'''ap = argparse.ArgumentParser()
ap,add_argument("-o","--output",required = True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())'''

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

callbacks = [LearningRateScheduler(step_decay)]

print('[INFO] compiling the model...')
opt = SGD(lr=0.01,momentum=0.9,nesterov=True)
model = MiniVGGNet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss='categorical_crossentropy',optimizer=opt,
             metrics=['accuracy'])

print('[INFO] training network...')
H = model.fit(trainx,trainy,validation_data=(testx,testy),batch_size=64,epochs=40,callbacks=callbacks,verbose=1)  

print('[INFO] evaluating network...')
predictions = model.predict(testx,batch_size=64)
print(classification_report(testy.argmax(axis=1),
                           predictions.argmax(axis=1),target_names=LabelNames))  

plt.plot(np.arange(0,40),H.history['loss'],label='train_loss')
plt.plot(np.arange(0,40),H.history['val_loss'],label='validation_loss')
plt.plot(np.arange(0,40),H.history['accuracy'],label='train_accuracy')
plt.plot(np.arange(0,40),H.history['val_accuracy'],label='validation_accuracy')
plt.title('training loss and accuracy on cifar10')
plt.xlabel('epochs #')
plt.ylabel('loss/accuracy')
plt.legend()