import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from scipy.spatial import Delaunay
import pyransac3d as pyrsc
from tqdm import tqdm
from collections import Counter


class PCD:
    # pcd
    # seabed
    # filtered_pcd
    # graph

    def __init__(self, filename):
        self.pcd = o3d.io.read_point_cloud(filename) # Pointcloud read from .xyz file
    
    def find_seabed(self):
        seabed = pyrsc.Plane()
        points_np = np.asarray(self.pcd.points)
        print("Total amount of points: " + str(points_np.shape))
        best_eq, best_inliers = seabed.fit(np.asarray(self.pcd.points), 0.22,maxIteration=3000)
        seabed_points = []
        for index in best_inliers:
            seabed_points.append(points_np[index])
        print("Points belonging to seabed: " + str(best_inliers.shape))
        self.seabed = np.array(seabed_points)
        return best_eq,np.array(seabed_points)
    
    def filter_seabed(self):
        orig_pcd = np.array(self.pcd.points) # Get original pointcloud
        plane_eq,seabed_inliers = self.find_seabed() # Get seabed pointcloud

        # Convert point clouds to sets
        cloud1_set = set(map(tuple, orig_pcd))
        cloud2_set = set(map(tuple, seabed_inliers))

        # Find common points
        common_points = cloud1_set.intersection(cloud2_set)

        # Remove common points (i.e seabed)
        processed_pcd = np.array([point for point in cloud1_set if point not in common_points])

        print("Number of points not consistent with seabed:" + str(processed_pcd.shape))

        self.filtered_pcd = processed_pcd
        return processed_pcd

    
    # 3 Methods for filtering out seabed:
    # --> RANSAC plane approximation, remove inliers
    # --> RANSAC paraboloid approximation, remove inliers
    # --> Iterative RANSAC approach
    
    def get_graph_rep(self,use_filtered):
        self.triang = Delaunay(self.pcd.points) # Triangulation made to render graph from pcd
        return self.triang


        




if __name__ == "__main__":
    test_pcd = PCD("../EM2040/data/xyz/unknown_objects/0016_20210607_135027.utm.xyz.xyz")
    #seabed,inliers = test_pcd.find_seabed()
    test_pcd.filter_seabed()
    #np.savetxt("../EM2040/data/seabed/0016_20210607_135027_seabed.txt",seabed)
    #np.savetxt("../EM2040/data/seabed/0016_20210607_135027_inliers.txt",inliers)

