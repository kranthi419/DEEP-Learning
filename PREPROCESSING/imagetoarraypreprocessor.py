# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:28:09 2020

@author: kaval
"""
from tensorflow.keras.preprocessing.image import img_to_array

class imagetoarraypreprocessor:
    def __init__(self,dataformat=None):
        #store the image data format
        self.dataformat = dataformat 
    def preprocess(self,image):
        #apply the keras utility function that correctly re arranges
        #the dimensions of image
        return img_to_array(image,data_format=self.dataformat)
    #the dataformat is default to zero which indicates the setting inside keras.json should be used
    #we could explicity supply a channels_first or channels_last string but it is best to let keras choose which image dimension oredering
    # to be used based on configuration file            
   