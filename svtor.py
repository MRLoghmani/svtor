# Implementing Scalable Vocabulary Trees for Object Recognition
# Group 13, CS676A, IITK
#------------------------------------------------------------------------------------------------------------

import cv2, time, os, math, operator, re, sys
import numpy as np
from sklearn.cluster import KMeans


#------------------------------------------------------------------------------------------------------------
N = 100								# Number of samples to take as training set
rootDir = 'data/full'					# The root directory of the dataset
nodes = {}								# List of nodes (list of SIFT descriptors)
nodeIndex = 0							# Index of the last node for which subtree was constructed
tree = {}								# A dictionary in the format - node: [child1, child2, ..]
branches = 5							# The branching factor in the vocabulary tree
leafClusterSize = 20					# Minimum size of the leaf cluster
imagesInLeaves = {} 					# Dictionary in the format - leafID: [img1:freq, img2:freq, ..]
doc = {}								# 
bestN = 4								#
result = np.array([0,0,0,0])			#
maxDepth = 5
avgDepth = 0
# If the values are supplied as command line arguments
if len(sys.argv) == 3:
	branches = int(sys.argv[1])
	maxDepth = int(sys.argv[2])
model =  KMeans(n_clusters=branches, n_jobs=4)	# The KMeans Clustering Model
sift = cv2.xfeatures2d.SIFT_create(nfeatures=500) # SIFT Feature extractor model
leafClusterSize = 2*branches
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
			
			kp, des = sift.detectAndCompute(gray, None)
			for d in des:
				features.append(d)
			# del kp, des
			n = n + 1
			if n >= N:
				break
	features = np.array(features)
	return features

# Function to construct the vocabulary tree
def constructTree(node, featuresIDs, depth):
	global nodeIndex					# Changes made in this variable will be global 
	global nodes						# Changes made in this variable will be global
	global tree 						# Changes made in this variable will be global
	global imagesInLeaves				# Changes made in this variable will be global
	global avgDepth
	tree[node] = []
	if len(featuresIDs) >= leafClusterSize and depth < maxDepth :
		# Here we will fetch the cluster from the indices and then use it to fit the kmeans
		# And then just after that we will delete the cluster
		# Using the array of indices instead of cluster themselves will reduce the memory usage by 128 times :)
		cluster = []
		for i in featuresIDs:
			cluster.append(features[i])
		model.fit(cluster)
		del cluster
		childFeatureIDs = [[] for i in range(branches)]
		for i in range(len(featuresIDs)):
			childFeatureIDs[model.labels_[i]].append(featuresIDs[i])
		for i in range(branches):
			nodeIndex = nodeIndex + 1
			nodes[nodeIndex] = model.cluster_centers_[i]
			tree[node].append(nodeIndex)
			constructTree(nodeIndex, childFeatureIDs[i], depth + 1)
	else:
		imagesInLeaves[node] = {}
		avgDepth = avgDepth + depth

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
	
	kp, des = sift.detectAndCompute(gray, None)	
	for d in des:
		leafID = lookup(d, 0)
		if filename in imagesInLeaves[leafID]:
			imagesInLeaves[leafID][filename] += 1
		else:
			imagesInLeaves[leafID][filename] = 1
	# del kp, des

# This function returns the weight of a leaf node
def weight(leafID):
	return math.log1p(N/1.0*len(imagesInLeaves[leafID]))

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

def accuracy(F, M1, M2, M3, M4):
	a = [0,0,0,0]
	group = int(F/4)
	if int(M1/4) == group:
		a[0] = 1
	if int(M2/4) == group:
		a[1] = 1
	if int(M3/4) == group:
		a[2] = 1
	if int(M4/4) == group:
		a[3] = 1
	return np.array(a)

# Finds 4 best matches for the query
def match(filename):
	# q is the frequency of this image appearing in each of the leaf nodes
	q = {}
	img = cv2.imread(dirName + "/" + fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
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

def getImgID(s):
	return int((re.findall("\d+", s))[0])

#------------------------------------------------------------------------------------------------------------

print("Extracting Features: " + rootDir + " ...")
features = dumpFeatures(rootDir)

print("Constructing Vocabulary Tree ... ")
root = features.mean(axis = 0)
nodes[0] = root
# Do not send the feature array itself but an array of indices into the construct tree function
# This will save memory by a factor of 128, an awesome little trick, why didn't I think it before
featuresIDs = [x for x in range(len(features))]
constructTree(0, featuresIDs, 0)

# del features

avgDepth = int(avgDepth/len(imagesInLeaves))

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
		# print(getImgID(filename), ": ", getImgID(group[0][0]), getImgID(group[1][0]), getImgID(group[2][0]), getImgID(group[3][0]))
		print(getImgID(filename), ": ", accuracy(getImgID(filename), getImgID(group[0][0]), getImgID(group[1][0]), getImgID(group[2][0]), getImgID(group[3][0])))
		result = result + accuracy(getImgID(filename), getImgID(group[0][0]), getImgID(group[1][0]), getImgID(group[2][0]), getImgID(group[3][0]))
		n = n + 1
		if n >= N:
			break

print(branches, maxDepth, avgDepth, result/N)
#------------------------------------------------------------------------------------------------------------