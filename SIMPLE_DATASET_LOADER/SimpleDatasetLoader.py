# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:24:11 2020

@author: kaval
"""
import numpy as np
import cv2
import os

class simpledatasetloader:
    def __init__(self, preprocessors=None):
        #store the image preprocessor
        self.preprocessors = preprocessors 
        #if preprocessors are None,initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []
    def load(self,imagePaths,verbose=-1):
        data=[]
        labels = []
        
        for (i,imagePath) in enumerate(imagePaths):
            #load the images and extract the class label assuming
            #that our path has following format:
            #/path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
                    
            data.append(image)
            labels.append(label)
            
            if verbose > 0 and i>0 and (i+1)%verbose==0:
                print("[info] processed {}/{}".format(i+1,len(imagePaths)))
        return (np.array(data),np.array(labels)) 
