# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 00:57:40 2020

@author: kaval
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from CNN.SHALLOW_NET import shallownet

import matplotlib.pyplot as plt
import numpy as np


print("[info] loading cifar10 data...")
((trainx,trainy),(testx,testy)) = cifar10.load_data()
trainx = trainx.astype("float")/255.0
testx = testx.astype("float")/255.0

lb = LabelBinarizer()
trainy = lb.fit_transform(trainy)
testy = lb.transform(testy)

#initialize the label names for cifar10 dataset
LabelNames = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truct"]

#initialize the optimizer and model
print("[info] compiling model...")
opt = SGD(lr =0.01)
model = shallownet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,
              metrics=["accuracy"])
#train the network
print("[info] training network...")
H = model.fit(trainx,trainy,validation_data=(testx,testy),
              batch_size=32,epochs=40,verbose=1)

#evaluate the network
print("[info] evaluating network...")
predictions = model.predict(testx,batch_size=32)
print(classification_report(testy.argmax(axis=1),
                            predictions.argmax(axis=1),target_names=LabelNames))

#plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,40),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,40),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,40),H.history["accuracy"],label="train_acc")
plt.plot(np.arange(0,40),H.history["val_accuracy"],label="val_acc")
plt.title("training loss and accuracy")
plt.xlabel("epoch #")
plt.ylabel("loss/accuracy")
plt.legend()
plt.show()