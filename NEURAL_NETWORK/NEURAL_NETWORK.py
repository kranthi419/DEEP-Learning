# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:06:28 2020

@author: kaval
"""
import numpy as np

class neuralnetwork:
    def __init__(self,layers,alpha=0.1):
        #initialize the list of weights matrices, then store the 
        #network architecture and learning rate
        self.w = []
        self.layers = layers
        self.alpha = alpha
        #layers: a list of integers which represents the actual architecture of feedforward network
        #ex:a value of [2,2,1] would imply that our first input layer has two nodes,our hidden layer has two nodes and
        #our final ouput has 1 node
        #alpha: learning rate of our network.this value is applied during the weight update phase.
        # W is the list of weights in each layer
        
        #start looping from the index of first layer but stop before we reach last two layers
        for i in np.arange(0,len(layers)-2):
            #randomly initialize a weight matrix connecting the number of nodes in each respective layer
            #adding an extra node for bias
            w1 = np.random.randn(layers[i]+1,layers[i+1]+1)
            self.w.append(w1/np.sqrt(layers[i]))
            
            #the matrix is mxn since wish to connect every node in current layer to every node in next layer
            #layers[i] +1 here '1' is adding bias term
            
        #the last two layers are a special case where the input connections 
        #need a bias term but output does not
        w1 = np.random.randn(layers[-2]+1,layers[-1])
        self.w.append(w1/np.sqrt(layers[-2])) #scaling the weights
        
        #next a "magic method" named __repr__ this function is useful for debugging
    def __repr__(self):
         # construct and return a string that represent the network architecture
         return "neuralnetwork: {}".format("-".join(str(l) for l in self.layers))
         '''nn=neuralnetwork([2,2,1])
         print(nn)
         output: neuralnetwork: 2-2-1'''
    def sigmoid(self,x):
        # compute and return sigmoid activation value for given input
        return 1.0/(1+np.exp(-x))
    
    #we use derivative of sigmoid in backward pass
    def  sigmoid_deriv(self,x):
        # compute derivative of sigmoid function
        return x*(1-x)
    #fitting the data
    def fit(self,X,y,epochs=1000,displayupdate=100):
        #insert a column of 1's as the last entry in feature matrix (adding the bias)
        X = np.c_[X,np.ones((X.shape[0]))]
        
        for epoch in np.arange(0,epochs):
            for (x,target) in zip(X,y):
                self.fit_partial(x, target)
                # check to see if we should display training update
            if epoch==0 or (epoch+1) % displayupdate == 0:

                loss = self.calculate_loss(X,y)
                print("[info] epoch={}, loss={:.7f}".format(epoch+1,loss))
    def fit_partial(self,x,y):
        #construct output activation for each layer as our data point flows through the network
        #first activation is special case just input of feature vector itself
        A = [np.atleast_2d(x)]
        #this list is stores the output activations for each layer as our data point x forward propagates
        
        '''FORWARD PASS'''
        for layer in np.arange(0,len(self.w)):
            #feedforward the activation at current layer by taking dot product between
            #the activation and weight matrix
            #this is called net input to current layer
            net = A[layer].dot(self.w[layer])
            #apply the activation function to net input
            out = self.sigmoid(net)
            
            #once we have net output, add it to our list of activations
            A.append(out)
        '''BACKWARD PASS'''
        #the first phase of backpropagation is to compute the
        #difference between our prediction(the final output activation in activation list) and the true target value
        error = A[-1]-y
        #from here we need to apply chain rule and vuild list of deltas 'D':the first entry of deltas
        #is simply the errors of output layer times the derivative of our activation function for 
        #the output value
        D = [error * self.sigmoid_deriv(A[-1])]
        #once you understand the chain rule it becomes super easy
        #to implement with a for loop and loop over the layers in reverse order(ignoring the last two since we already taken in account)
        for layer in np.arange(len(A)-2,0,-1):
            #the delta for current layer is equal to delta of previous layer dotted with weight matrix of current layer
            #followed by multiplying the delta by derivative of non linear activation function for activation of current layer
            delta = D[-1].dot(self.w[layer].T)
            delta = delta*self.sigmoid_deriv(A[layer])
            D.append(delta)
        #here we simply taking the delta from previous layer,dotting it with weights of current layer and then multiplying by 
        #the derivative of activation.this process is repeated until we reach the first layer in network
        
        #since we looped over the layer in reverse order we reverse the deltas
        D=D[::-1]
        #weight update phase
        #loop over the layers
        for layer in np.arange(0,len(self.w)):
            #update the weights by taking the dot product of layer activations with their respective deltas,then 
            #multiplying this value by some small learning rate and adding to our weight matrix --this is where actual learning take place
            self.w[layer]+=-self.alpha*A[layer].T.dot(D[layer])
        
    def predict(self,X,addbias=True):
        #initialize the output prediction as input features this value will be (forward) propagated through the network
        #to obtain the final prediction
        p=np.atleast_2d(X)
            
        #check to see if bias column should be added
        if addbias:
            p=np.c_[p,np.ones((p.shape[0]))]
        for layer in np.arange(0,len(self.w)):
            #compute output prediction by simply taking the dot product
            p = self.sigmoid(np.dot(p,self.w[layer]))
        return p
    def calculate_loss(self,X,targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X,addbias=False)
        loss = 0.5*np.sum((predictions-targets)**2)

        return loss
    