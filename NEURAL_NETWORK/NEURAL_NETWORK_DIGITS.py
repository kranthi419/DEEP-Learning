# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 19:57:57 2020

@author: kaval
"""
from NEURAL_NETWORK import neuralnetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

#loading the data
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data-data.min())/(data.max()-data.min()) #scaling the data
print("[info] samples: {}, dim: {}".format(data.shape[0],data.shape[1]))

#splitting into train and test samples
(trainx,testx,trainy,testy) = train_test_split(data,digits.target,test_size=0.25)

#encoding the values
trainy = LabelBinarizer().fit_transform(trainy)
testy = LabelBinarizer().fit_transform(testy)

#we can encode our class label integer as vector by process called one hot encoding
nn=neuralnetwork([trainx.shape[1],32,16,10])
print("[info] {}".format(nn))
nn.fit(trainx,trainy,epochs=1000)

predictions = nn.predict(testx)
predictions = predictions.argmax(axis=1)
print(classification_report(testy.argmax(axis=1),predictions))
