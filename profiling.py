# This code is for Tony's assignment of profiling the centroids
from sklearn.cluster import KMeans
import numpy as np
import math

# Get coords
# Currently vectors are split by line and each coordinate is separated by " "
# Modifications can be made to this for different format datasets
def getVectors(file):
    vectors = []
    while True:
        vector=file.readline()
        if not vector:
            break
        stringVector = vector.split(" ")[:-1]
        intVector = [int(element) for element in stringVector]
        vectors.append(intVector)
    file.close()
    return vectors

# Split each vector into pairs of coords
# For now we assume there are an even number of dimensions
def splitVectors(vectors, split):
    dim = len(vectors[0])
    miniVectors = [[] for x in range(int(dim / split))]
    for vector in vectors:
        for x in range(int(dim / split)):
            miniVector = []
            for y in range(split):
                miniVector.append(vector[x * split + y])
            miniVectors[x].append(miniVector)
    return miniVectors

def findDist(center, point):
    dim = len(center)
    distSquared = 0
    for x in range(dim):
        distSquared += math.pow(center[x] - point[x], 2)
    return math.sqrt(distSquared)

# Find 16 centroids for each coordinate pair (K-Means Clustering)
# Product Quantization
def findCentroids(miniVectors):
    numVectors = len(miniVectors[0])
    vectorCentroids = [[] for x in range(numVectors)]
    vectorCentroidDists = [[] for x in range(numVectors)]
    for x in range(len(miniVectors)):
        npMiniVector = np.array(miniVectors[x])
        kmeans = KMeans(n_clusters=16, random_state=0, n_init="auto").fit(npMiniVector)
        for y in range(len(kmeans.labels_)):
            vectorCentroids[y].append(kmeans.labels_[y])
            center = kmeans.cluster_centers_[kmeans.labels_[y]]
            point = miniVectors[x][y]
            dist = findDist(center, point)
            vectorCentroidDists[y].append(dist)
    return vectorCentroids, vectorCentroidDists

# Softmax Distribution
# Separate into 8 categories based on softmax value
def softmaxDistribute(vectorCentroids, vectorCentroidDists):
    dim = len(vectorCentroids[0])
    softmax = [[-1 for x in range(dim)] for x in range(len(vectorCentroids))]
    split8 = [[-1 for x in range(dim)] for x in range(len(vectorCentroids))]
    for dimension in range(dim):
        for centroid in range(16):
            sum = 0
            max = -1
            min = -1
            for index in range(len(vectorCentroids)):
                if vectorCentroids[index][dimension] == centroid:
                    dist = vectorCentroidDists[index][dimension]
                    sum += math.exp(dist)
                    if math.exp(dist) > max:
                        max = math.exp(dist)
                    if math.exp(dist) < min or min == -1:
                        min = math.exp(dist) 
            max = max / sum
            min = min / sum
            for index in range(len(vectorCentroids)):
                if vectorCentroids[index][dimension] == centroid:
                    dist = vectorCentroidDists[index][dimension]
                    softmax[index][dimension] = math.exp(dist) / sum
                    split8[index][dimension] = (softmax[index][dimension] - min) / (max - min) * 8
                    split8[index][dimension] = int(split8[index][dimension])
                    if split8[index][dimension] == 8:
                        split8[index][dimension] = 7
    return softmax, split8

def main():
    print("Revving up...")
    file = open("data.txt","r")
    vectors = getVectors(file)
    print("vectors:")
    print(vectors)

    miniVectors = splitVectors(vectors, 2)
    print("mini vectors:")
    print(miniVectors)

    vectorCentroids, vectorCentroidDists = findCentroids(miniVectors)
    print("vector centroids:")
    print(vectorCentroids)
    print("vector centroid distances")
    print(vectorCentroidDists)
    centroidFile = open('centroids.txt','w')
    for vector in vectorCentroids:
        for coord in vector:
            centroidFile.write(str(coord) + " ")
        centroidFile.write("\n")
    centroidFile.close()
    centroidDistFile = open('centroidDist.txt','w')
    for vector in vectorCentroidDists:
        for coord in vector:
            centroidDistFile.write(str(coord) + " ")
        centroidDistFile.write("\n")
    centroidDistFile.close()


    softmax, split8 = softmaxDistribute(vectorCentroids, vectorCentroidDists)
    print("softmax:")
    print(softmax)
    print("split8")
    print(split8)

    softmaxFile = open('softmax.txt','w')
    for vector in softmax:
        for coord in vector:
            softmaxFile.write(str(coord) + " ")
        softmaxFile.write("\n")
    softmaxFile.close()

    splitFile = open('split8.txt', 'w')
    for vector in split8:
        for coord in vector:
            splitFile.write(str(coord) + " ")
        splitFile.write("\n")
    splitFile.close()

if __name__=="__main__":
        main()