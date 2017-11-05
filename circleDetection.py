#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 20:45:10 2017

@author: dawit
"""

import cv2
import numpy as np;
def performMorpholigicalErosion(thresh):
     kernel = np.ones((5,5),np.uint8)
     errod=cv2.erode(thresh,kernel,iterations=1)
     erroded =cv2.dilate(errod,kernel,iterations=1)
     return erroded
 
def performBinarization(image):
    avgPixelIntensity = cv2.mean(image)
    meancolor = avgPixelIntensity [0]
    value =[meancolor,80,120]
    value = np.array(value)
    threshCoeff = 0.15 + (np.min(value) - 80) * ((0.2- 0.15) / (120 - 80));
    treshvalue = int(threshCoeff*255)
    ret,thresh = cv2.threshold(image,treshvalue,255,cv2.THRESH_BINARY_INV)
    return thresh
# Read image
im = cv2.imread("/media/dawit/Data/DistentionDataFoldeClassified/Train/Open/open0.jpg", cv2.IMREAD_GRAYSCALE)
im = performBinarization(im)
im = performMorpholigicalErosion(im)

params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 0.15
params.maxThreshold = 0.20
params.minDistBetweenBlobs = 50.0
 
# Filter by Area.
#params.filterByArea = True
params.minArea = 100.0
params.maxArea=100000.0
 
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = False
params.filterByCircularity = False
params.filterByArea = True
 
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)
# Set up the detector with default parameters.

 
# Detect blobs.
keypoints = detector.detect(im)
print(len(keypoints))
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()