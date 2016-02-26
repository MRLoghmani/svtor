# Implementing Scalable Vocabulary Trees for Object Recognition
# Author: Pranshu Gupta
#------------------------------------------------------------------------------------------------------------

import cv2, time, os, math, operator
import numpy as np
from sklearn.cluster import KMeans

#------------------------------------------------------------------------------------------------------------
N = 20									# Number of samples to take as training set
rootDir = 'data/full'					# The root directory of the dataset
model =  KMeans(n_clusters=5)			# The KMeans Clustering Model
nodes = {}								# List of nodes (list of SIFT descriptors)
nodeIndex = 0							# Index of the last node for which subtree was constructed
tree = {}								# A dictionary in the format - node: [child1, child2, ..]
branches = 5							# The branching factor in the vocabulary tree
leafClusterSize = 20					# Minimum size of the leaf cluster
imagesInLeaves = {} 					# Dictionary in the format - leafID: [img1:freq, img2:freq, ..]
doc = {}
bestN = 4
#------------------------------------------------------------------------------------------------------------

# Function to dump all the SIFT descriptors from training data in the feature space
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

# Function to construct the vocabulary tree
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
		imagesInLeaves[node] = {}

# Function to lookup a SIFT descriptor in the vocabulary tree, returns a leaf cluster
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

# Constructs the inverted file frequency index
def tfidf(filename):
	global imagesInLeaves
	img = cv2.imread(dirName + "/" + fname)
	gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(gray, None)	
	for d in des:
		leafID = lookup(d, 0)
		if filename in imagesInLeaves[leafID]:
			imagesInLeaves[leafID][filename] += 1
		else:
			imagesInLeaves[leafID][filename] = 1

# This function returns the weight of a leaf node
def weight(leafID):
	return math.log1p(len(imagesInLeaves[leafID])/1.0*N)

# Returns the scores of the images in the dataset
def getScores(q):
	scores = {}
	for dirName, subdirList, fileList in os.walk(rootDir):
		n = 0
		for fname in fileList:
			img = dirName + "/" + fname
			scores[img] = 0
			for leafID in q:
				if leafID in doc[img]:
					scores[img] += math.fabs(q[leafID] - doc[img][leafID])
			n = n + 1
			if n >= N:
				break
	return scores

# Return the bestN best matches
def findBest(scores, bestN):
	sorted_scores = sorted(scores.items(), key = operator.itemgetter(1))
	return sorted_scores[:bestN]


# Finds 4 best matches for the query
def match(filename):
	# q is the frequency of this image appearing in each of the leaf nodes
	q = {}
	img = cv2.imread(dirName + "/" + fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(gray, None)
	for d in des:
		leafID = lookup(d, 0)
		if leafID in q:
			q[leafID] += 1
		else:
			q[leafID] = 1
	s = 0.0
	for key in q:
		q[key] = q[key]*weight(key)
		s += q[key]
	for key in q:
		q[key] = q[key]/s
	scores = getScores(q)
	return findBest(scores, bestN)

#------------------------------------------------------------------------------------------------------------

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
#
for leafID in imagesInLeaves:
	for img in imagesInLeaves[leafID]:
		if img not in doc:
			doc[img] = {}
		doc[img][leafID] = weight(leafID)*(imagesInLeaves[leafID][img])
for img in doc:
	s = 0.0
	for leafID in doc[img]:
		s += doc[img][leafID]
	for leafID in doc[img]:
		doc[img][leafID] /= s

print("Finding Best Matches for each image ...")
for dirName, subdirList, fileList in os.walk(rootDir):
	n = 0
	for fname in fileList:
		filename = dirName + "/" + fname
		group = match(filename)
		print(filename, ": ", group[0][0], group[1][0], group[2][0], group[3][0])
		n = n + 1
		if n >= N:
			break

#------------------------------------------------------------------------------------------------------------