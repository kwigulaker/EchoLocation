from pcd_preprocess import PCD
from time import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import networkx.algorithms.community as nx_comm

def clusterAgglo(pcd):
    clustering = AgglomerativeClustering(n_clusters=None,distance_threshold=2,compute_full_tree=True).fit(pcd.outliers)
    pcd.clusters = clustering.labels_
    
def clusterDBScan(pcd):
    clustering = DBSCAN(eps=2, min_samples=5).fit(pcd.outliers)
    pcd.clusters = clustering.labels_


def clusterHDBScan(pcd):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    pcd.clusters = np.array([clusterer.fit_predict(pcd.outliers)])


def getConnectedComponents(pcd):
        cc = nx.connected_components(pcd.graph_outliers)
        # Create list of each component, list of indexes.
        cluster_labels = np.empty(len(pcd.outliers))
        list = sorted(cc, key=len, reverse=True)
        print([len(c) for c in list])
        ind = 0
        for component in list:
            for point in component:
                cluster_labels[point] = ind
            ind += 1
        pcd.clusters = cluster_labels
        return cluster_labels

if __name__ == "__main__":
    test_pcd = PCD("../EM2040/data/source/xyz/unknown_objects/0016_20210607_135027.utm.xyz.xyz")
    test_pcd.find_seabed_ransac(True) # Seabed filtering
    test_pcd.generateGraphNN()
    getConnectedComponents(test_pcd) # Clustering
    test_pcd.plot2D(False,True) # Plotting
    np.savetxt("../EM2040/data/seabed/0016_20210607_135027_inliers.txt",test_pcd.seabed) # Save seabed inliers
    test_pcd.writeToClusters("../EM2040/data/clusters/all/0016_20210607_135027/0016_20210607_135027") # Save outliers as multiple clustered pointclouds
