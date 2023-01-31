import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from scipy.spatial import Delaunay
import pyransac3d as pyrsc
from tqdm import tqdm
from collections import Counter


class PCD:
    # outliers --> points not consistent with seabed, in theory noise and/or objects.
    # seabed --> points consistent with seabed plane(s)
    # outliers_graph --> outlier points represented as graphs (tentative, may be removed)
    # cluster_centroids
    # cluster_labels

    def __init__(self, filename):
        pcd = o3d.io.read_point_cloud(filename) # Pointcloud read from .xyz file
        self.outliers = np.asarray(pcd.points)
    
    def find_seabed_ransac(self,iterations):
        for i in range(iterations):
            print("Iteration: " + str(i))
            seabed = pyrsc.Plane()
            points_np = self.outliers
            print("Current amount of outliers: " + str(points_np.shape))
            best_eq, best_inliers = seabed.fit((self.outliers), 0.15,maxIteration=3000)
            seabed_points = []
            for index in best_inliers:
                seabed_points.append(points_np[index])
            print("New points belonging to seabed: " + str(np.array(seabed_points).shape))
            if i == 0:
                self.seabed = np.array(seabed_points)
            else:
                self.seabed = np.append(self.seabed,np.array(seabed_points),axis=0)
            print("Total points in seabed: " + str(self.seabed.shape))
            self.filter_seabed()
    
    def filter_seabed(self):
        # This step can become computationally complex when other methods are used, so far this is the quickest filtering method I could find
        orig_pcd = self.outliers # Get original pointcloud
        seabed_inliers = self.seabed # Get seabed pointcloud

        # Convert point clouds to sets
        cloud1_set = set(map(tuple, orig_pcd))
        cloud2_set = set(map(tuple, seabed_inliers))

        # Find common points
        common_points = cloud1_set.intersection(cloud2_set)

        # Remove common points (i.e seabed)
        processed_pcd = np.array([point for point in cloud1_set if point not in common_points])

        print("New number of outliers:" + str(processed_pcd.shape))

        self.outliers = processed_pcd
    
    def get_graph_rep(self):
        self.outliers_graph = Delaunay(self.outliers) # Triangulation made to render graph from pcd
        return self.outliers_graph


        




if __name__ == "__main__":
    test_pcd = PCD("../EM2040/data/xyz/identifiable_objects/0001_20210607_130222.utm.xyz.xyz")
    test_pcd.find_seabed_ransac(3)
    np.savetxt("../EM2040/data/seabed/0001_20210607_130222_inliers.txt",test_pcd.seabed)
    np.savetxt("../EM2040/data/seabed/0001_20210607_130222_outliers.txt",test_pcd.outliers)
