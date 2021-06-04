# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 21:25:33 2020

@author: kaval
"""
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, k):
    # grab the spatial dimensions of image and kernel
    (ih,iw) = image.shape[:2]
    (kh,kw) = k.shape[:2]
    
    #allocate memory for the output image,taking care to "pad"
    #the borders os the input image so the spatial size(i.e.,
    #width and height) are not reduced
    
    pad = (kw-1)//2
    image = cv2.copyMakeBorder(image,pad,pad,pad,pad,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((ih,iw),dtype = "float")
    
    #recall that we "center" our computation around the center(x,y)
    #co ordinate of the input image that the kernel is currently positioned over.
    #This positioning implies there is no such thing as "center" pixels for pixels that
    #fall along the border of image(as the corners of kernel would be "hanging off"
    #the image where the values are undefined)
    
    #loop over the input image, "sliding" the kernel across
    #each (x,y) coordinate from left to right and top to bottom
    for y in np.arange(pad,ih+pad):
        for x in np.arange(pad,iw+pad):
            
            #extract the ROI if image by extracting the "center" region
            #of the current (x,y) coordinates dimensions
            roi= image[y-pad:y+pad+1,x-pad:x+pad+1]
            
            #perform the actual convolution by taking the element wise
            #multiplication between roi and kernel then summing the matrix
            k=(roi*k).sum()
            
            #store the convoled value in output(x,y)coordinate of output image
            output[y-pad,x-pad]=k
    #rescale the output image to be in range[0,255]
    output = rescale_intensity(output,in_range=(0,255))
    output = (output *255).astype("uint8")
    
    #return the output image
    return output

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,
                help="path to the input image")
args = vars(ap.parse_args())


#construct average blurring kernels used to smooth an image
smallblur = np.ones((7,7),dtype="float")*(1.0/(7*7))
largeblur = np.ones((21,21),dtype="float")*(1.0/(21*21))

#construct a sharpening filter
sharpen = np.array((
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]),
    dtype="int")
#construct the laplacian kernel used to detect edge like
laplacian = np.array((
    [0,1,0],
    [1,-4,1],
    [0,1,0]),
    dtype="int")
#construct the sobel xaxis kernel (detect edge)
sobelx = np.array((
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]),
    dtype="int")
#construct the sobel yaxis kernel (detect edge)
sobely = np.array((
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]),
    dtype="int")
emboss = np.array((
    [-2,-1,0],
    [-1,1,1],
    [0,1,2]),
    dtype="int")
kernelbank = (
    ("small_blur",smallblur),
    ("large_blur",largeblur),
    ("sharpen",sharpen),
    ("laplacian",laplacian),
    ("sobel_x",sobelx),
    ("sobel_y",sobely),
    ("emboss",emboss))
#load the input image and convert it to grayscale
image = cv2.imread(args["image"])

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
for (kernelname,k) in kernelbank:
    
    print("[info] applying {} kernel".format(kernelname))
    convolveoutput = convolve(gray,k)
    opencvoutput = cv2.filter2D(gray,-1,k)
    
    cv2.imshow("original",gray)
    cv2.imshow("{}-convolve".format(kernelname),convolveoutput)
    cv2.imshow("{}-opencv".format(kernelname),opencvoutput)
    cv2.waitKey(0)
    cv2.destroyALLWindows()