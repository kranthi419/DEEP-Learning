# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 00:53:26 2020

@author: kaval
"""
import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        #initialize the weight matrix and store the learing rate
        self.w = np.random.randn(N+1)/np.sqrt(N)
        self.alpha = alpha
        # N is the number of columns in the input vectors.
        # In context of our bitwise datasets,we'll set N equal to two
        # aplha is our learing rate
        # normally alpha ranges from 0.1,0.01,0.001
        # weight matrix will have n+1 entries one for each class and one bias
        # we divide by sqrt in order to scale our matrix leading to faster convergence
    def step(self, x):
        # apply the step function
        return 1 if x>0 else 0
    
    def fit(self, X,y,epochs=10):
        # inset a column of 1's as the last entry in the feature
        # to allows us to treat bias
        X = np.c_[X,np.ones((X.shape[0]))]
        
        #loop over the desired number of epochs
        for epoch in np.arange(0,epochs):
            # loop over each individual data point
            for (x,target) in zip(X,y):
                
                #doing the dot.product operation and updating the weights
                p = self.step(np.dot(x, self.w))
                if p !=target:
                    error = p - target
                    self.w  += -self.alpha * error * x   
                    
    def predict(self, X,addbias = True):
         # ensure our input is a matrix
         X=np.atleast_2d(X)
         if addbias:
             X=np.c_[X,np.ones((X.shape[0]))] 
             return self.step(np.dot(X,self.w))
   

    
    
    
    
    
    
    