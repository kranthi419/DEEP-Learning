# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 18:18:13 2020

@author: kaval
"""

from CNN.LE_NET import lenet    
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

print("[info] loading dataset")
((trainx,trainy),(testx,testy)) = mnist.load_data()   

#dont need necessary unless we change architecture of channels_first ordering
#channels_first
if K.image_data_format()=="channels_first":
    trainx = trainx.reshape((trainx.shape[0],1,28,28))
    testx = testx.reshape((testx.shape[0],1,28,28))
#channels_last
else:
    trainx = trainx.reshape((trainx.shape[0],28,28,1))
    testx = testx.reshape((testx.shape[0],28,28,1))
trainx = trainx.astype("float32")/255.0
testx = testx.astype("float32")/255.0

lb = LabelBinarizer()
trainy = lb.fit_transform(trainy)
testy = lb.transform(testy)

print("[info] compiling model...")
opt = SGD(lr=0.01)
model = lenet.build(width=28, height=28,depth=1,classes=10 )
model.compile(loss = "categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
print("[info] training model...")
H = model.fit(trainx,trainy,validation_data=(testx,testy),batch_size=128,epochs=20,verbose=1)
print("[info] evaluating network...")
predictions = model.predict(testx,batch_size=128)
print(classification_report(testy.argmax(axis=1),predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))
#plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,20),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,20),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,20),H.history["accuracy"],label="train_acc")
plt.plot(np.arange(0,20),H.history["val_accuracy"],label="val_acc")
plt.title("training loss and accuracy")
plt.xlabel("epoch #")
plt.ylabel("loss/accuracy")
plt.legend()
plt.show()   