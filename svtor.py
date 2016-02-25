# Implementing Scalable Vocabulary Trees for Object Recognition
# Author: Pranshu Gupta
#------------------------------------------------------------------------------------------------------------

import cv2, time, os
import numpy as np
from sklearn.cluster import KMeans


N = 20									# Number of samples to take as training set
rootDir = 'data/full'					# The root directory of the dataset
model =  KMeans(n_clusters=5)			# The KMeans Clustering Model
nodes = {}								# List of nodes (list of SIFT descriptors)
nodeIndex = 0							# Index of the last node for which subtree was constructed
tree = {}								# A dictionary in the format - node: [child1, child2, ..]
branches = 5							# The branching factor in the vocabulary tree
leafClusterSize = 20					# Minimum size of the leaf cluster
imagesInLeaves = {}

def dumpFeatures(rootDir):
	features = []
	for dirName, subdirList, fileList in os.walk(rootDir):
		n = 0
		for fname in fileList:
			# print("Reading Image: " + dirName + "/" + fname)
			img = cv2.imread(dirName + "/" + fname)
			gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			sift = cv2.xfeatures2d.SIFT_create()
			kp, des = sift.detectAndCompute(gray, None)
			for d in des:
				features.append(d)
			n = n + 1
			if n >= N:
				break
	features = np.array(features)
	return features


def constructTree(node, cluster, depth):
	global nodeIndex					# Changes made in this variable will be global 
	global nodes						# Changes made in this variable will be global
	global tree 						# Changes made in this variable will be global
	global imagesInLeaves				# Changes made in this variable will be global
	tree[node] = []
	if len(cluster) >= leafClusterSize:
		model.fit(cluster)
		childCluster = [[] for i in range(branches)]
		for i in range(len(cluster)):
			childCluster[model.labels_[i]].append(cluster[i])
		for i in range(branches):
			nodeIndex = nodeIndex + 1
			nodes[nodeIndex] = model.cluster_centers_[i]
			tree[node].append(nodeIndex)
			constructTree(nodeIndex, childCluster[i], depth + 1)
	else:
		imagesInLeaves[node] = []


def lookup(descriptor, node):
	D = float("inf")
	goto = None
	for child in tree[node]:
		dist = np.linalg.norm([nodes[child] - descriptor])
		if D > dist:
			D = dist
			goto = child
	if tree[goto] == []:
		return goto
	return lookup(descriptor, goto)	


def tfidf(filename):
	global imagesInLeaves
	img = cv2.imread(dirName + "/" + fname)
	gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(gray, None)	
	for d in des:
		leafID = lookup(d, 0)
		imagesInLeaves[leafID].append(filename)

print("Extracting Features: " + rootDir + " ...")
features = dumpFeatures(rootDir)

print("Constructing Vocabulary Tree ... ")
root = features.mean(axis = 0)
nodes[0] = root
constructTree(0, features, 0)

print("Mapping images to leaf nodes of the tree ...")
for dirName, subdirList, fileList in os.walk(rootDir):
	n = 0
	for fname in fileList:
		filename = dirName + "/" + fname
		tfidf(filename)
		n = n + 1
		if n >= N:
			break

