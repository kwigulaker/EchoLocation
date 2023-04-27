import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from pyRANSAC import pyransac3d as pyrsc
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import pyproj as pyproj

## Class containing a pointcloud, including all necessary preprocessing functions and representations
class PCD:
    # outliers --> points not consistent with seabed, in theory noise and/or objects.
    # seabed --> points consistent with seabed plane(s)
    # equation --> seabed equation parameters
    # seabed_corners --> 3 points to generate plane.
    # norm_outliers --> [[x_min,x_max],[y_min,y_max],[z_min,z_max]], metrics used to normalize- and denormalize pointcloud.
    # graph_outliers --> points not consistent with seabed represented in graph form.
    # clusters --> labels for each point as to which cluster/object they belong.

    def __init__(self, filename):
        pcd = o3d.io.read_point_cloud(filename) # Pointcloud read from .xyz files
        self.pcd = pcd
        self.outliers = np.asarray(pcd.points)
        self.graph_outliers = None
    
    def find_seabed_ransac(self,remove_neg):
        # What if this just runs indefinitely, until seabeds with less than say 5000 points arent found?
        num_failures = 0
        i = 0
        inlier_threshold = 0.15
        while num_failures < 3:
            print("Iteration: " + str(i+1))
            seabed = pyrsc.Plane()
            points_np = self.outliers
            print("Current amount of outliers: " + str(points_np.shape))
            best_eq, best_inliers,seabed_corners = seabed.fit((self.outliers),inlier_threshold,maxIteration=5000)
            self.equation = best_eq
            self.seabed_corners = seabed_corners
            seabed_points = []
            # This should be a dynamic variable and not hardcoded
            if np.array(best_inliers).shape[0] < 8500:
                num_failures += 1
                continue
            for index in best_inliers:
                seabed_points.append(points_np[index])
            if i == 0:
                self.seabed = np.array(seabed_points)
            else:
                self.seabed = np.append(self.seabed,np.array(seabed_points),axis=0)
            print("Total points in seabed: " + str(self.seabed.shape))
            self.filter_seabed(remove_neg)
            i+=1
    
    def filter_seabed(self,remove_neg):
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

        # We ALSO want to check for points lying horizontally on the seabed, with a negative height deviation

        # We check horizontally whether given point is within 2D limits of each seabed plane
        # Then find nearest neighbour point within seabed point, derive vector between given point and NN
        # Decompose vector into three parts, check sign of z-part, if negative, point is over seabed, if positive --> add point to seabed inliers.

        # This is longwinded as hell right now, no doubt it can be simplified
        if(remove_neg):
            new_seabed_points = []
            for point in processed_pcd:
                x,y = point[0],point[1]
                # First check if point (x,y) lies on the horizontal area defined by the seabed plane.
                pointM = np.array([x,y])
                pointA = np.array([self.seabed_corners[0][0],self.seabed_corners[0][1]])
                pointB = np.array([self.seabed_corners[1][0],self.seabed_corners[1][1]])
                pointC = np.array([self.seabed_corners[2][0],self.seabed_corners[2][1]])
                # https://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not
                vectorAB = pointB-pointA
                vectorAM = pointM-pointA
                vectorBC = pointC-pointB
                vectorBM = pointM-pointB
                # 0 <= dot(AB,AM) <= dot(AB,AB) && 0 <= dot(BC,BM) <= dot(BC,BC)
                if (np.dot(vectorAB,vectorAM) >= 0 and np.dot(vectorAB,vectorAB) >= np.dot(vectorAB,vectorAM)) and (np.dot(vectorBC,vectorBM) >= 0 and np.dot(vectorBC,vectorBC) >= np.dot(vectorBC,vectorBM)):
                    # Point lies in rectangle
                    # Check height difference, we say upwards is the z-axes
                    vec_up = np.array([0,0,1])
                    perp_point = self.getPointPerpendicular(point,self.equation)
                    diff_vec = point - perp_point
                    if(np.dot(vec_up,diff_vec) < 0):
                        # Do as above with filtering out seabed, make two sets
                        new_seabed_points.append(point)
            
            # Add new points to seabed
            if(len(new_seabed_points) > 0):
                self.seabed = np.append(self.seabed,np.array(new_seabed_points),axis=0)

                # Convert point clouds to sets
                cloud1_set = set(map(tuple, processed_pcd))
                cloud2_set = set(map(tuple, new_seabed_points))

                # Find common points
                common_points = cloud1_set.intersection(cloud2_set)

                # Remove common points (i.e seabed)
                processed_pcd = np.array([point for point in cloud1_set if point not in common_points])

        self.outliers = processed_pcd
    
    def getPointPerpendicular(self,point,plane):
        # point = [x,y,z]
        # plane = [a,b,c,d]
        # lambda: a*(x + lambda*a) + b*(y + lambda*b) + c*(z + lambda*c) + d = 0
        # (a^2 + b^2 + c^2)*lambda + a*x + b*y + c*z + d = 0
        x,y,z = point[0],point[1],point[2]
        a,b,c,d = plane[0],plane[1],plane[2],plane[3]
        delta = -(a*x + b*y + c*z + d)/(a*a + b*b + c*c)
        perp_point =  point + [delta*a,delta*b,delta*c]
        return perp_point

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

    def generateGraphNN(self,n_neighbors=None):
        print("Generating graph with Nearest Neighbours approach...")
        points = self.outliers
        node_dict = {i: (self.outliers[i,0],self.outliers[i,1],self.outliers[i,2]) for i in range(len(self.outliers[:,0]))} # Hashable array of positions for each point
        self.graph_outliers = nx.Graph()
        self.graph_outliers.add_nodes_from(node_dict)
        tree = KDTree(points, leaf_size=2)
        if n_neighbors is None:
            #all_nn_indices = tree.query_radius(points, r=0.5)  # NNs within distance r of point
            all_nn_indices = tree.query_radius(points, r=0.5)  # NNs within distance r of point
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
            # This is currently hardcoded to 10 objects, should be dynamic based on object size f.ex but the colourmaps have proven to be not very distinct
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
    def writeToClusters(self,location):
        # location: directory/name to write clusters to
        if self.graph_outliers is None:
            # Clusters generated through non-graph representation.
            cluster_ids = np.unique(self.clusters,return_counts=True)
            print("Number of clusters:" + str(len(cluster_ids[0])))
            print("Points per cluster: " + str(cluster_ids[1]))
            subgraphs = [[] for x in range(len(cluster_ids[0]))]
            point_index = 0
            for cluster_id in self.clusters:
                point = self.outliers[point_index]
                subgraphs[cluster_id].append(point)
                point_index+=1
            noise = []
            indice = 0
            for sub in subgraphs:
                print(np.array(sub).shape)
                if(len(sub) > 60):
                    np.savetxt(location + "_" + str(indice) + ".txt",sub)
                else:
                    for point in sub:
                        noise.append(point)
                indice += 1
            np.savetxt(location + "_noise.txt",noise)

        else:
            subgraphs = [self.graph_outliers.subgraph(c).copy() for c in nx.connected_components(self.graph_outliers)] # Note, these are arrays of indices not actual points
            ind_sub = 0
            noise = []
            for sub in subgraphs:
                curr_indice = 0
                indices = sub.nodes
                points = np.zeros(shape=(len(indices),3))
                # Currently only write individual clusters larger than 100 points, clusters smaller than 100 points get lumped together into 'noise' cluster
                # 100 is an arbitrary size
                if(len(indices) > 60):
                    for indice in indices:
                        points[curr_indice] = [self.outliers[indice][0],self.outliers[indice][1],self.outliers[indice][2]]
                        curr_indice += 1
                    np.savetxt(location + "_" + str(ind_sub) + ".txt",points)

                else:
                    for indice in indices:
                        noise.append( [self.outliers[indice][0],self.outliers[indice][1],self.outliers[indice][2]])
                ind_sub+=1
            np.savetxt(location + "_noise.txt",noise)



