import numpy as np
import matplotlib.pyplot as plt


def getPointPerpendicular(point,plane):
        # point = [x,y,z]
        # plane = [a,b,c,d]
        # lambda: a*(x + lambda*a) + b*(y + lambda*b) + c*(z + lambda*c) + d = 0
        # (a^2 + b^2 + c^2)*lambda + a*x + b*y + c*z + d = 0
        x,y,z = point[0],point[1],point[2]
        a,b,c,d = plane[0],plane[1],plane[2],plane[3]
        delta = -(a*x + b*y + c*z + d)/(a*a + b*b + c*c)
        perp_point =  point + [delta*a,delta*b,delta*c]
        return perp_point

def negativeHeightFilter(point,plane,equation):
        x,y = point[0],point[1]
        # First check if point (x,y) lies on the horizontal area defined by the seabed plane.
        pointM = np.array([x,y])
        pointA = np.array([plane[0][0],plane[0][1]])
        pointB = np.array([plane[1][0],plane[1][1]])
        pointC = np.array([plane[2][0],plane[2][1]])
        # https://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not
        vectorAB = pointB-pointA
        vectorAM = pointM-pointA
        vectorBC = pointC-pointB
        vectorBM = pointM-pointB
        # 0 <= dot(AB,AM) <= dot(AB,AB) && 0 <= dot(BC,BM) <= dot(BC,BC)
        if (np.dot(vectorAB,vectorAM) >= 0 and np.dot(vectorAB,vectorAB) >= np.dot(vectorAB,vectorAM)) and (np.dot(vectorBC,vectorBM) >= 0 and np.dot(vectorBC,vectorBC) >= np.dot(vectorBC,vectorBM)):
            # Point lies in rectangle
            # Check height difference between nearest
            perp_point = getPointPerpendicular(point,equation)
            # We say that 'up' is towards the z-axis in a positive direction.
            vec_up = np.array([0,0,1])
            diff_vec = point - perp_point
            diff = np.dot(vec_up,diff_vec)
            if(diff < 0):
                # Do as above with filtering out seabed, make two sets
                return True


def testNegativeHeightFilter():
        # Define a 3D plane
        A,B,C = np.random.randn(3,3) * 10
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
        other_points = np.random.randn(50, 3) * 10
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

        x_other = other_points[:,0]
        y_other = other_points[:,1]
        z_other = other_points[:,2]

        x_filtered = filtered_points[:,0]
        y_filtered = filtered_points[:,1]
        z_filtered = filtered_points[:,2]

        x_corners = corners[:,0]
        y_corners = corners[:,1]
        z_corners = corners[:,2]

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(xx, yy, zz, alpha=0.5)
        ax.scatter(x_other,y_other,z_other,c='red')
        ax.scatter(x_filtered,y_filtered,z_filtered,c='yellow')
        ax.scatter(x_corners,y_corners,z_corners,c='green')
        plt.show()

if __name__ == "__main__":
    testNegativeHeightFilter()


