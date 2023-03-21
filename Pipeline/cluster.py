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
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
import networkx.algorithms.community as nx_comm
from community import community_louvain
import itertools
from cdlib import algorithms


def clusterKMeans(pcd):
    kmeans =  KMeans(init="k-means++",n_clusters=12)
    hori = np.array([pcd.outliers[:,0],pcd.outliers[:,1]])
    hori = np.transpose(hori)
    kmeans.fit_predict(hori)
    return kmeans

def clusterAP(pcd):
    hori = np.array([pcd.outliers[:,0],pcd.outliers[:,1]])
    hori = np.transpose(hori)
    af = AffinityPropagation(random_state=0,max_iter=500).fit_predict(pcd.outliers)
    return af

def clusterDBSCAN(pcd):
    db = DBSCAN(eps=0.4, min_samples=5,algorithm='kd_tree').fit(pcd.outliers)
    return db

def clusterOPTICS(pcd):
    clust = OPTICS(min_samples=50, xi=0.5, min_cluster_size=0.05)
    clust.fit(pcd.outliers)
    return clust

def clusterMeanShift(pcd):
    clust = MeanShift(max_iter=1000).fit(pcd.outliers)
    return clust

def clusterAgglo(pcd):
    clust = AgglomerativeClustering(n_clusters=6, n_features_in_= 3).fit(pcd.outliers)
    return clust

def girvanNewman(pcd):
    print("Running Girvan-Newman for community detection...")
    communities = girvan_newman(pcd.graph_outliers)
    node_groups = []
    for com in next(communities):
        node_groups.append(list(com))
    return np.array(node_groups)

def SLPA(pcd):
    print("Running SLPA for community detection...")
    communities = algorithms.slpa(pcd.graph_outliers, t=3000,r=0.01)
    print(communities)

def louvain(pcd):
    print("Running Louvain's method for community detection...")
    partition = community_louvain.best_partition(pcd.graph_outliers)
    G = pcd.graph_outliers
    # draw the graph
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=1,
                        cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()


def clusterFEC(pcd):
    print("Clustering via FEC approach..")

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


    test_pcd = PCD("../EM2040/data/xyz/unknown_objects/0003_20220708_113138_Shipname.utm.xyz.xyz")
    test_pcd.find_seabed_ransac(True) # Seabed filtering
    test_pcd.generateGraphNN() # Graph representation
    getConnectedComponents(test_pcd) # Clustering
    test_pcd.plot2D(False,True) # Plotting
    np.savetxt("../EM2040/data/seabed/0003_20220708_113138_Shipname_inliers.txt",test_pcd.seabed) # Save seabed inliers
    #np.savetxt("../EM2040/data/xyz/seeps/arne/0003_20220708_113138_Shipname_outliers.txt",test_pcd.outliers) # Save seabed outliers
    test_pcd.writeToClusters("../EM2040/data/clusters/0003_20220708_113138_Shipname") # Save outliers as multiple clustered pointclouds
