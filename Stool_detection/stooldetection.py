# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 20:05:44 2017

@author: Dawit
"""

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
#from imutils import paths
import numpy as np
import itertools
import argparse
#import imutils
import cv2
import os
import glob
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm,decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Expert Prediction ')
    plt.xlabel('Manual Tuning Prediction')
def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

 
def extract_color_histogram(image, bins=(256,256,256)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
     
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
 
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	#if imutils.is_cv2():
	#hist = cv2.normalize(hist)
 
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	#else:
	cv2.normalize(hist, hist)
 
	# return the flattened histogram as the feature vector
	return hist.flatten()


rawImages = []
features = []
labels = []

for imagePath in glob.glob('D:/StoolTrainingImages/*.jpg'):
    
    
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
        image = cv2.imread(imagePath)
        #value = getPrvalue(image)
         
        label = imagePath.split(os.path.sep)[-1].split("_")[0]
     
    	# extract raw pixel intensity "features", followed by a color
    	# histogram to characterize the color distribution of the pixels
    	# in the image
        #pixels = image_to_feature_vector(image)
        hist = extract_color_histogram(image)
     
    	# update the raw images, features, and labels matricies,
    	# respectively
        
        #rawImages.append(pixels)
        features.append(hist)
        #features.append(value)
        if 'Bud' in label:
            labels.append(0)
        elif 'suboptimal' in label:
            labels.append(1)
        elif 'optimal' in label:
            labels.append(2)
        else:
            labels.append(3)
        

rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.15, random_state=42)

clf = SVC(C=100,gamma = 10 ,kernel='rbf')
model = clf.fit(trainFeat, trainLabels)  
acc = model.score(testFeat,testLabels)
Predicted = model.predict(testFeat)
cnf_matrix = confusion_matrix(testLabels, Predicted)

class_names=["bud", "suboptimal", "optimal","good"]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')


print('Accuracy:',acc)
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
# create a parameter grid: map the parameter names to the values that should be saved
param_grid_dt = dict(gamma=gamma_range, C=C_range) # for DT
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# instantiate the grid
grid = GridSearchCV(SVC(kernel ='rbf'), param_grid_dt, cv=cv, scoring='accuracy')
grid.fit(trainFeat,trainLabels)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(trainFeat, trainLabels)
        classifiers.append((C, gamma, clf))