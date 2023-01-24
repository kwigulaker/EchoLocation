from pcd_preprocess import PCD
from time import time
import numpy as np
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import BisectingKMeans



def clusterKMeans(pcd):
    kmeans =  KMeans(init="k-means++")
    kmeans.fit(pcd.points)
    return kmeans

def clusterVarAgglo(pcd):
    linkage = "ward"
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    t0 = time()
    clustering.fit(pcd.points)
    print("%s :\t%.2fs" % (linkage, time() - t0))
    return clustering

def clusterDBSCAN(pcd):
    db = DBSCAN(eps=0.23, min_samples=10).fit(pcd.points)
    return db


def clusterBIRCH(pcd):
    birch = Birch(threshold=0.25).fit(pcd.points)
    return birch

def clusterKMeansBisect(pcd):
    kmeans = BisectingKMeans(init="random",max_iter=100000,bisecting_strategy="biggest_inertia", n_clusters=3).fit(pcd.points)
    return kmeans



if __name__ == "__main__":
    test_pcd = PCD("../EM2040/data/xyz/identifiable_objects/0002_20210607_130418.utm.xyz.xyz")
    clusters = clusterDBSCAN(test_pcd.pcd)
    #centroids = clusters.cluster_centers_
    #cluster_identify = clusters.predict(test_pcd.pcd.points)
    labels = clusters.labels_
    # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)

    # print("Estimated number of clusters: %d" % n_clusters_)
    # print("Estimated number of noise points: %d" % n_noise_)
    # np.savetxt("../EM2040/data/clusters/0001_20210607_130222_clusters_birch.txt",labels)
    np.savetxt("../EM2040/data/clusters/0001_20210607_130222_cluster_ids_bisect.txt",labels)
    #np.savetxt("../EM2040/data/clusters/0001_20210607_130222_clusters_centers_bisect.txt",centroids)