import torch
import pickle
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_distances_argmin_min
from sklearn.datasets import make_blobs
import numpy as np
import math
import monitoring_util as mu

def takeSecond(elem):
    return elem[1]

def takeFirst(elem):
    return elem[0]

def getCentroids(representations_object, includedDigits):

    centroids = {}
    rep_statistics = {}

    for idx in includedDigits:
        rep_statistics[idx] = [0,None,0] # Count, sum_rep, centroid

    for rep_id in range(len(representations_object[0])):
        tag = representations_object[2][rep_id]
        representation = representations_object[0][rep_id]
        rep_statistics[tag][0] = rep_statistics[tag][0] + 1
        if rep_statistics[tag][1] is None:
            rep_statistics[tag][1] = representation
        else:
            rep_statistics[tag][1] = rep_statistics[tag][1] + representation

    for tag in includedDigits:
        centroids[tag] = rep_statistics[tag][1] / rep_statistics[tag][0]

    return centroids

def getCentroidStats(representations_object,includedDigits,centroids,widthFactor):

    centroidStats = {}
    for idx in includedDigits:
        centroidStats[idx] = (0,0)

    distanceSet = {}
    for idx in includedDigits:
        distanceSet[idx] = []

    for rep_id in range(len(representations_object[0])):
        tag = representations_object[2][rep_id]
        representation = representations_object[0][rep_id]
        distance = np.linalg.norm(centroids[tag] - representation)
        distanceSet[tag].append(distance)

    for tag in includedDigits:
        np_set = np.array(distanceSet[tag])
        mean = np.mean(np_set)
        stdev = np.mean(np_set)
        centroidStats[tag] = (mean, mean + widthFactor * stdev)

    return centroidStats

def getDistanceBetweenCentroids(centroids,includedDigits):

    output = {}
    crossDistances = []
    for tag1 in includedDigits:
        for tag2 in includedDigits:
            centroid1 = centroids[tag1]
            centroid2 = centroids[tag2]
            distance = np.linalg.norm(centroid1 - centroid2)
            crossDistances.append((tag1,tag2,distance))
            print(f'Tag1:Tag2:Dist   {tag1}  {tag2}   {distance}')

    for tag in includedDigits:
        output[tag] = None

    for item in crossDistances:
        if item[0] != item[1]:
            tag1 = item[0]
            tag2 = item[1]
            value = item[2]
            if output[tag1] is None:
                output[tag1] = value
            else:
                if output[tag1] > value:
                    output[tag1] = value
            if output[tag2] is None:
                output[tag2] = value
            else:
                if output[tag2] > value:
                    output[tag2] = value
    return output

def performClustering(representations_object,batch):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=5, batch_size=64,
                          n_init=10, max_no_improvement=10, verbose=0)
    mbk.fit(representations_object[0])
    mbk_means_cluster_centers = mbk.cluster_centers_
    labels = mbk.labels_

    return labels, mbk_means_cluster_centers

def getAnnotatedRepresentations(representations_object,centroids,includedDigits):

    annotated_reps = []

    for rep_id in range(len(representations_object[0])):
        tag = representations_object[2][rep_id]
        centroid = centroids[tag]
        representation = representations_object[0][rep_id]
        source = representations_object[1][rep_id]
        distance = np.linalg.norm(representation - centroid)
        annotated_reps.append((tag, distance, source))

    annotated_reps.sort(key=takeSecond, reverse=True)
    annotated_reps.sort(key=takeFirst, reverse=False)

    return annotated_reps

def buildPinningSet(annotated_reps, includedDigits, example_count, interCentroidDistances, centroidStats):
    example_count_dict = {}
    for tag in includedDigits:
        example_count_dict[tag] = []

    for rep in annotated_reps:
        tag1 = rep[0]
        example_dist = rep[1]
        source = rep[2]
        closest_cloud_distance = interCentroidDistances[tag]
        maximum_cloud_halfwidth = 0
        for tag2 in includedDigits:
            if tag2 != tag1:
                if centroidStats[tag2][1] > maximum_cloud_halfwidth:
                    maximum_cloud_halfwidth = centroidStats[tag2][1]
        maximum_distance = closest_cloud_distance - maximum_cloud_halfwidth

        if example_dist < maximum_distance:
            if len(example_count_dict[tag1]) < example_count:
                example_count_dict[tag1].append((source,tag1))

    pinning_set = []

    for tag in includedDigits:
        for item in example_count_dict[tag]:
            pinning_set.append(item)

    return pinning_set

def main():
    example_count = 60
    includedDigits = [0, 1, 2, 3, 4]
    source_data_path = 'results/representations.pkl'
    widthFactor = 2

    with open(source_data_path,'rb') as pickleFile:
        representations_object = pickle.load(pickleFile)

    # representation_object = (left_representations,left_values,left_image_class)
    centroids = getCentroids(representations_object,includedDigits)
    centroidStats = getCentroidStats(representations_object, includedDigits, centroids, widthFactor)
    interCentroidDistances = getDistanceBetweenCentroids(centroids, includedDigits)

    annotated_reps = getAnnotatedRepresentations(representations_object,centroids,includedDigits)

    # mu.plotHist(annotated_reps)

    pinning_set = buildPinningSet(annotated_reps, includedDigits, example_count, interCentroidDistances, centroidStats)

    with open('results/pin_set.pkl', 'wb') as pickleFile:
        pickle.dump(pinning_set, pickleFile)
    print('done')

main()