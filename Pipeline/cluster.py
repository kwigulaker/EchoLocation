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
    kmeans.fit(pcd.outliers)
    return kmeans



if __name__ == "__main__":
    test_pcd = PCD("../EM2040/data/xyz/identifiable_objects/0001_20210607_130222.utm.xyz.xyz")
    test_pcd.find_seabed_ransac(3)
    test_kmeans = clusterKMeans(test_pcd)
    
