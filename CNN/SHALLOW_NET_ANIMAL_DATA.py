from PREPROCESSING.imagetoarraypreprocessor import imagetoarraypreprocessor
from PREPROCESSING.SimplePreprocessor import simplepreprocessor
from datasets.SimpleDatasetLoader import simpledatasetloader
from CNN.SHALLOW_NET import shallownet

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

'''
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,
                help="path to input dataset")
args = vars(ap.parse_args())

#grab the list of images that we will describing
print("[INFO] loading images...")
'''

from imutils import paths
imagePaths = list(paths.list_images("animals"))


#intialize the image preprocessor
sp = simplepreprocessor(32,32)
iap = imagetoarraypreprocessor()

#load the dataset from disk and scale the raw pixel
sdl = simpledatasetloader(preprocessors=[sp,iap]) 
#(data,labels ) = sdl.load(imagepaths, verbose=500)

(data,labels ) = sdl.load(imagePaths, verbose=50)
data = data.astype("float")/255.0

#spliting the dataset into training and testing
(trainx,testx,trainy,testy) = train_test_split(data,labels,test_size=0.25,random_state=42)

#convert the label into integers
trainy = LabelBinarizer().fit_transform(trainy)
testy = LabelBinarizer().fit_transform(testy)

print("[INFO] compiling model")
opt = SGD(lr=0.005)
model = shallownet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

#training the model
H = model.fit(trainx,trainy,validation_data=(testx,testy),
              batch_size=32,epochs=100,verbose=1)

#evaluating and predicting values
print("[info] evaluating network...")
predictions = model.predict(testx,batch_size=32)
print(classification_report(testy.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=["cat","dog","pandas"]))

#plot the training loss and accuracy
#plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,100),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,100),H.history["accuracy"],label="train_acc")
plt.plot(np.arange(0,100),H.history["val_accuracy"],label="val_acc")
plt.title("training loss and accuracy")
plt.xlabel("epoch #")
plt.ylabel("loss/accuracy")
plt.legend()
plt.show()
