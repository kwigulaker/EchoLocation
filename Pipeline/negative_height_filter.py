import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def downSampleRandom(pcd,threshold):
    indices = np.random.choice(pcd.shape[0], threshold, replace=False)
    downsampled_pcd = pcd[indices, :]
    return downsampled_pcd

def downSampleVoxel(pcd,voxel_size):
    num_points = np.asarray(pcd).shape[0]
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    return np.asarray(downsampled_pcd.points)

#def testDownsampling():





def getPointPerpendicular(point,plane):
        # point = [x,y,z]
        # plane = [a,b,c,d]
        # plabe equation: a*(x + delta*a) + b*(y + delta*b) + c*(z + delta*c) + d = 0
        # (a^2 + b^2 + c^2)*delta + a*x + b*y + c*z + d = 0
        # delta = -(a*x + b*y + c*z + d)/(a*a + b*b + c*c)
        x,y,z = point[0],point[1],point[2]
        a,b,c,d = plane[0],plane[1],plane[2],plane[3]
        delta = -(a*x + b*y + c*z + d)/(a*a + b*b + c*c)
        perp_point =  point + [delta*a,delta*b,delta*c]
        return perp_point

def negativeHeightFilter(point,plane,equation):
        # point defined as [x,y,z]
        # plane contains the three corners defining the bounds of the plane equation
        # equation contains parameters A, B, C, D for the plane equation
        x_min = min(plane[:,0])
        y_min = min(plane[:,1])
        x_max = max(plane[:,0])
        y_max = max(plane[:,1])
        # We check the upper and lower bounds of the rectangle.
        if((point[0] > x_min and point[0] < x_max) and (point[1] > y_min and point[1] < y_max)):
            # Point lies in rectangle
            # Check height difference between nearest
            perp_point = getPointPerpendicular(point,equation)
            # We say that 'up' is towards the z-axis in a positive direction.
            vec_up = np.array([0,0,1])
            diff_vec = point - perp_point
            diff = np.dot(vec_up,diff_vec)
            if(diff < 0):
                # Point lies below plane.
                return True


def testNegativeHeightFilter():
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        # Define a 3D plane
        A,B,C = np.random.randn(3,3) * 50
        AB = B - A #vecA
        AC = C - A #vecB
        normal_vector = np.cross(AB, AC) #cross of vecA, vecB = vecC
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        d = -np.dot(normal_vector,A) # k
        corners = np.array([A,B,C])

        equation = [normal_vector[0],normal_vector[1],normal_vector[2],d]
        x_min = min(corners[:,0])
        y_min = min(corners[:,1])
        x_max = max(corners[:,0])
        y_max = max(corners[:,1])

        # Define the x and y coordinates of the plane
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 10))

        # Calculate the z coordinate of the plane using the plane equation
        zz = (-normal_vector[0]*xx - normal_vector[1]*yy - d)*1./normal_vector[2]
        # Generate 50 additional random points
        other_points = np.random.randn(600, 3) * 50
        filtered_points = []
        for point in other_points:
              on_plane = negativeHeightFilter(point,np.array([A,B,C]),equation)
              if(on_plane):
                filtered_points.append(point)
        
        cloud1_set = set(map(tuple, other_points))
        cloud2_set = set(map(tuple, filtered_points))

        # Find common points
        common_points = cloud1_set.intersection(cloud2_set)

        # Remove common points (i.e seabed)
        other_points = np.array([point for point in cloud1_set if point not in common_points])
        filtered_points = np.array(filtered_points)
        if(len(other_points) < 1):
                x_other = []
                y_other = []
                z_other = []
        else:
                x_other = other_points[:,0]
                y_other = other_points[:,1]
                z_other = other_points[:,2]
        if(len(filtered_points) < 1):
                x_filtered = []
                y_filtered = []
                z_filtered = []
        else:
                x_filtered = filtered_points[:,0]
                y_filtered = filtered_points[:,1]
                z_filtered = filtered_points[:,2]
        x_corners = corners[:,0]
        y_corners = corners[:,1]
        z_corners = corners[:,2]

        surf = ax.plot_surface(xx, yy, zz, alpha=0.5)
        surf._facecolors2d  = surf._facecolor3d
        surf._edgecolors2d  = surf._edgecolor3d
        ax.scatter(x_other,y_other,z_other,c='red',s=1)
        ax.scatter(x_filtered,y_filtered,z_filtered,c='blue',s=1)
        ax.scatter(x_corners,y_corners,z_corners,c='green',s=1)
        #tikzplotlib.save("NSF.tex")
        plt.show()

if __name__ == "__main__":
    testDownsampling()
    testNegativeHeightFilter()


