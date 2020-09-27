# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 16:04:55 2020

@author: kaval
"""
from PERCEPTRON import Perceptron
import numpy as np

# AND GATE PERCEPTRON

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[0],[0],[1]])

p = Perceptron(X.shape[1],alpha=0.1)

# training
p.fit(X,y,epochs=20)

# testing
for (x,target) in zip(X,y):
    pred = p.predict(x)
    print("data = {},ground-truth {},pred= {}".format(x,target[0],pred))