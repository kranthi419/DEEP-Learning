#Resnet with Bottleneck and pre-activation

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as k

# Inside the residual module we will adding the outputs of two branches
# which will be accomplished via this add method and we also import l2 function to 
# perform L2 weight decay 
''' regurlarization is extremely important when training resnet since the network is depth and it 
prone to overfitting'''

class resnet:
    @staticmethod 
    def residual_module(data,k,stride,chandim,red=False,reg=0.0001,bneps=2e-5,bnmom=0.9):
        ''' the data parameter is simply input to residual module
        the k is number of filters that will learned by final conv in bottleneck
        the 1st two conv layers will learn k/4 filters as per He.et al.paper.
        red (i.e reduce) boolean will control whether we are reducing spatial dimensions(True)/(False)
        we can apply regularization strength to all conv layers in residual module via reg
        bneps parameter controls the e responsible for avoiding "division by zero" errors when normalizing inputs
        bnmom controls  the momentum for moving average
        '''
        
        #the shortcut branch of resnet module should be initialize as input data
        shortcut = data
         #1st block of resnet module are 1x1 CONVs
        bn1 = BatchNormalization(axis=chandim,epsilon=bneps,momentum=bnmom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(k * 0.25),(1,1),use_bias=False,kernel_regularizer=l2(reg))(act1)
        # the first pre-activation of bottleneck branch can see in line 37 to 39 
        # the we excluding bias term because according to He et al.,the biases are in BN layers that immediately follow the convolutions so
        # there is no need to introduce a second bias term
        # above we implemented k/4 using 1x1 filters
     
        #2nd block of resnet module are 3x3 CONVs
        #2nd conv layer in bottleneck
        bn2 = BatchNormalization(axis=chandim,epsilon=bneps,momentum=bnmom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(k * 0.25),(3,3),strides=stride,padding="same",use_bias=False,
                       kernel_regularizer=l2(reg))(act2)
         
        # final block os resnet module is another set of 1x1 convs
        bn3 = BatchNormalization(axis=chandim,epsilon=bneps,momentum=bnmom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(k,(1,1),use_bias=False,kernel_regularizer=l2(reg))(act3)
        
        #next step we need to check if we need to spatial dimensions thereby alleviating need to apply max pooling
        if red:
            shortcut = Conv2D(k,(1,1),strides=stride,use_bias=False,kernel_regularizer=l2(reg))(act1)
            # if we want to reduce the size we use stride > 1
            # the output of final conv3 in bottleneck is added together with shortcut,thus serving as output of residual module
            x = add([conv3,shortcut])
            return x
    #the residual module will serve as our building block when creating deep residual networks
    @staticmethod 
    def build(width,height,depth,classes,stages,filters,reg=0.0001,bneps=2e-5,bnmom=0.9,dataset = "cifar"):
        
        inputshape = (height,width,depth)
        chandim = -1
        #if we using channels first update the input shape
        if k.image_data_format() == "channel_first":
            inputshape = (depth,height,width)
            chandim = 1
        
        inputs = Input(shape=inputshape)
        x = BatchNormalization(axis=chandim,epsilon=bneps,momentum=bnmom)(inputs)
        if dataset == 'cifar10':
            x = Conv2D(filters[0],(3,3),use_bias=False,padding="same",kernel_regularizer=l2(reg))(x)
        
        # the reason behind uses a BN layer as 1st layer is applying normalization to our input is an added level of normalization
        # in fact applying batch normalization on input itself can sometimes remove the need to apply mean normalization to inputs
        # here is a list as conv layer learns a total of filters[0],3x3
        
        for i in range(0,len(stages)):
            stride = (1,1) if i==0 else (2,2)
            x = resnet.residual_module(x,filters[i+1],stride,chandim,red=True,bneps=bneps,bnmom=bnmom)
            
            for j in range(0,stages[i]-1):
                x = resnet.residual_module(x,filters[i+1],(1,1),chandim,bneps=bneps,bnmom=bnmom)
            #
            
            x = BatchNormalization(axis=chandim,epsilon=bnps,momentum=bnmom)(x)
            x = Activation("relu")(x)
            x = AveragePooling2D((8,8))(x)
            
            x = Flatten()(x)
            x = Dense(classes,kernel_regularizer=l2(reg))(x)
            x = Activation("softmax")(x)
            
            model = Model(inputs,x,name="resnet")
            
            return model 
import matplotlib
#matplotlib.use("Agg")

#import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as k
import numpy as np
import argparse
import sys

# set a high recursion limit so Theano doesn't complain
sys.setrecursionlimit(5000)

'''
ap=argparse.ArgumentParser()
ap.add_argument("-c","--checkpoints",required=True,
                help="path to output checkpoint directory")
ap.add_argument("-m","--model",type=str,
                help="path to specific model checkpoint to load")
ap.add_argument("-s","--start-epoch",type=int,default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())
'''
print("[info] loading cifar10 data")
((trainx,trainy),(testx,testy)) = cifar10.load_data()
trainx = trainx.astype("float")
testx = testx.astype("float")

# apply mean subtraction to data
mean = np.mean(trainx,axis=0)
trainx-=mean
testx-=mean

lb = LabelBinarizer()
trainy = lb.fit_transform(trainy)
testy = lb.transform(testy)

aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,horizontal_flip=True,
                         fill_mode="nearest")

print("[info] compiling model..")
opt = SGD(lr=1e-1)
model = resnet.build(32,32,3,10,(9,9,9),(64,64,128,256),reg=0.0005)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=['accuracy'])
model.fit_generator(aug.flow(trainx,trainy,batch_size=128),
                    validation_data=(testx,testy),steps_per_epoch=len(trainx)//128,epochs=100,verbose=1)           
                                        
