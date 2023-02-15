import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
import pyransac3d as pyrsc
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

## Class containing a pointcloud, including all necessary preprocessing functions and representations
class PCD:
    # outliers --> points not consistent with seabed, in theory noise and/or objects.
    # seabed --> points consistent with seabed plane(s)
    # norm_outliers --> [[x_min,x_max],[y_min,y_max],[z_min,z_max]], metrics used to normalize- and denormalize pointcloud.
    # graph_outliers --> points not consistent with seabed represented in graph form.
    # clusters --> labels for each point as to which cluster/object they belong.

    def __init__(self, filename):
        pcd = o3d.io.read_point_cloud(filename) # Pointcloud read from .xyz file
        self.outliers = np.asarray(pcd.points)
    
    def find_seabed_ransac(self,iterations):
        for i in range(iterations):
            print("Iteration: " + str(i+1))
            seabed = pyrsc.Plane()
            points_np = self.outliers
            print("Current amount of outliers: " + str(points_np.shape))
            best_eq, best_inliers = seabed.fit((self.outliers), 0.15,maxIteration=3000)
            seabed_points = []
            for index in best_inliers:
                seabed_points.append(points_np[index])
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
    
    def normalize(self):
        print("Normalizing outliers between 0 and 100")
        # Defined normalization range
        x_min_def = 0.0
        x_max_def = 100.0

        y_min_def = 0.0
        y_max_def = 100.0

        z_min_def = 0.0
        z_max_def = 100.0

        # Observed max and min values
        x_max_obs = np.max(self.outliers[:,0])
        x_min_obs = np.min(self.outliers[:,0])
        print("x max:" + str(x_max_obs) + ", x min:" + str(x_min_obs))
        
        y_max_obs = np.max(self.outliers[:,1])
        y_min_obs = np.min(self.outliers[:,1])
        print("y max:" + str(y_max_obs) + ", y min:" + str(y_min_obs))

        z_max_obs = np.max(self.outliers[:,2])
        z_min_obs = np.min(self.outliers[:,2])
        print("z max:" + str(z_max_obs) + ", z min:" + str(z_min_obs))

        self.norm_outliers = np.array([[x_min_obs,x_max_obs],[y_min_obs,y_max_obs],[z_min_obs,z_max_obs]])

        for point in self.outliers:
            point[0] = (x_max_def - x_min_def)/(x_max_obs-x_min_obs)*(point[0]-x_min_obs)+x_min_def
            point[1] = (y_max_def - y_min_def)/(y_max_obs-y_min_obs)*(point[1]-y_min_obs)+y_min_def
            point[2] = (z_max_def - z_min_def)/(z_max_obs-z_min_obs)*(point[2]-z_min_obs)+z_min_def


    def denormalize(self):
        # Defined normalization range
        x_min_def = 0.0
        x_max_def = 100.0

        y_min_def = 0.0
        y_max_def = 100.0

        z_min_def = 0.0
        z_max_def = 100.0

        # Observed max and min values
        x_min_obs = self.norm_outliers[0,0]
        x_max_obs = self.norm_outliers[0,1]

        y_min_obs = self.norm_outliers[1,0]
        y_max_obs = self.norm_outliers[1,1]

        z_min_obs = self.norm_outliers[2,0]
        z_max_obs = self.norm_outliers[2,1]

        for point in self.outliers:
            x_norm = point[0]
            y_norm = point[1]
            z_norm = point[2]

            point[0] = ((x_norm - x_min_def)*(x_max_obs - x_min_obs))/(x_max_def - x_min_def) + x_min_obs
            point[1] = ((y_norm - y_min_def)*(y_max_obs - y_min_obs))/(y_max_def - y_min_def) + y_min_obs
            point[2] = ((z_norm - z_min_def)*(z_max_obs - z_min_obs))/(z_max_def - z_min_def) + z_min_obs

    def generateGraphDelaunay(self):
        # Have no idea whether this is actually useful or not.
        print("Running Delaunay triangulation to generate graph...")
        # Note: for 3d points this generates tetahedrons instead of triangles
        tri = Delaunay(self.outliers)
        edge_indices = tri.simplices # Array of indices in self.outliers that are connected via edges.
        self.graph_outliers = nx.Graph() # NetworkX graph
        node_dict = {i: (self.outliers[i,0],self.outliers[i,1],self.outliers[i,2]) for i in range(len(self.outliers[:,0]))} # Hashable array of positions for each point
        self.graph_outliers.add_nodes_from(node_dict)
        for tetrahedron in edge_indices:
            # 6 edges of tetahedron
            self.graph_outliers.add_edge(tetrahedron[0],tetrahedron[1])
            self.graph_outliers.add_edge(tetrahedron[0],tetrahedron[2])
            self.graph_outliers.add_edge(tetrahedron[0],tetrahedron[3])

            self.graph_outliers.add_edge(tetrahedron[1],tetrahedron[2])
            self.graph_outliers.add_edge(tetrahedron[1],tetrahedron[3])
            self.graph_outliers.add_edge(tetrahedron[2],tetrahedron[3])
            
        print("Number of vertices: " + str(len(self.graph_outliers)))
        print("Number of edges generated: " + str(np.array(self.graph_outliers.edges).shape))
        self.plot2D()

    def generateGraphNN(self,n_neighbors=None):
        print("Generating graph with Nearest Neighbours approach...")
        points = self.outliers
        node_dict = {i: (self.outliers[i,0],self.outliers[i,1],self.outliers[i,2]) for i in range(len(self.outliers[:,0]))} # Hashable array of positions for each point
        self.graph_outliers = nx.Graph()
        self.graph_outliers.add_nodes_from(node_dict)
        tree = KDTree(points, leaf_size=2)
        if n_neighbors is None:
            all_nn_indices = tree.query_radius(points, r=2)  # NNs within distance r of point
            ind = 0
            for point in all_nn_indices:
                for neighbour in point:
                    if(ind == neighbour):
                        continue
                    else:
                        self.graph_outliers.add_edge(ind,neighbour)
                ind += 1
        else:
            for point in points:
                dist, all_nn_indices = tree.query(points, k=n_neighbors)
                ind = 0
                for point in all_nn_indices:
                    for neighbour in point:
                        self.graph_outliers.add_edge(ind,neighbour)
                    ind += 1
                

        print("Number of vertices: " + str(len(self.graph_outliers)))
        print("Number of edges generated: " + str(np.array(self.graph_outliers.edges).shape))

    
    def plot2D(self,edges,clustering):
        # We plot the horizontal plane to judge clustering and edge generation success
        if edges:
            for edge in self.graph_outliers.edges:
                start = edge[0]
                end = edge[1]
                point1 = self.outliers[start]
                point2 = self.outliers[end]
                x_values = [point1[0], point2[0]]
                y_values = [point1[1], point2[1]]
                plt.plot(x_values, y_values, 'bo', linestyle="--", linewidth=0.1)
        x = self.outliers[:,0]
        y = self.outliers[:,1]
        if clustering:
            # Lets make a colormap
            colors = [None]* len(self.clusters)
            # We want the colours of the 10 largest objects to pop, otherwise be filtered.
            index = 0
            for item in self.clusters:
                if(item < 11):
                    colors[index] = item
                else:
                    colors[index] = 11
                index+=1
            plt.scatter(x,y,alpha=1,s=0.01,c=colors,cmap='Paired')
            plt.colorbar()
            plt.show()
        else:
            plt.scatter(x,y,alpha=0.5,s=0.01)
            plt.show()

    ## Write pointcloud as multiple individual pointclouds representing each cluster
    def writeToClusters(self,name,location):
        # name: name of pcd
        # location: directory to write clusters to
        

