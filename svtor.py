# Implementing Scalable Vocabulary Trees for Object Recognition
# Author: Pranshu Gupta
###############################################################################
import cv2
import numpy as np
import os

N = 20
rootDir = 'data/full'

def dumpFeatures(rootDir):
	features = []
	for dirName, subdirList, fileList in os.walk(rootDir):
		n = 0
		for fname in fileList:
			# print("Reading Image: " + dirName + "/" + fname)
			img = cv2.imread(dirName + "/" + fname)
			gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			sift = cv2.xfeatures2d.SIFT_create()
			kp, des = sift.detectAndCompute(gray, None)
			for d in des:
				features.append(d)
			n = n + 1
			if n >= N:
				break
	features = np.array(features)
	return features

print("Extracting DataSet Features: " + rootDir)
features = dumpFeatures(rootDir)
