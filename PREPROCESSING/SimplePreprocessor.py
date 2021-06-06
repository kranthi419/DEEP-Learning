# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:23:39 2020

@author: kaval
"""
import cv2

class simplepreprocessor:
    def __init__(self,width,height,inter=cv2.INTER_AREA):
        #store the target image width,height and interpolation
        #method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
    def preprocess(self,image):
        #resize the image to a fixed size,ignoring aspect ratio
        return cv2.resize(image,(self.width,self.height),
                          interpolation=self.inter)