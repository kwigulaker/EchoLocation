import numpy as np
import open3d as o3d
from mayavi import mlab
from pyntcloud import PyntCloud
from collections import Counter

def plot_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    x = np.array(pcd.points)[:, 0]  # x position of point
    y = np.array(pcd.points)[:, 1]  # y position of point
    z = np.array(pcd.points)[:, 2]  # z position of point
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1920, 1080))

    # Colour correcting based on height
    mlab.points3d(x,y,z,z,          # Values used for Color
                        mode="point",
                        colormap='gnuplot', # 'bone', 'copper', 'gnuplot'
                        #color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                        figure=fig,
                        )
    mlab.show()
    sys.exit()

def plot_pcd_clusters(filename,clusters,centroids):
    pcd = o3d.io.read_point_cloud(filename)
    x = np.array(pcd.points)[:, 0]  # x position of point
    y = np.array(pcd.points)[:, 1]  # y position of point
    z = np.array(pcd.points)[:, 2]  # z position of point
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1920, 1080))
    cluster_ids = np.loadtxt(clusters)
    if(centroids is None):
        unique_ids = np.unique(cluster_ids)
        num_clusters = len(unique_ids)
    else:
        num_clusters =len(np.loadtxt(centroids))

    lut = np.zeros((len(cluster_ids), 4))
    print("Number of points:" + str(len(cluster_ids)))
    print("Number of clusters:" + str(num_clusters))
    for id in range(len(cluster_ids)):
        f = cluster_ids[id]/num_clusters
        lut[id,:] = [255*(1-f),0,255*f,255]
                
    # Count points per cluster
    unique, counts = np.unique(cluster_ids, return_counts=True)
    print("Points per cluster: ")
    for i in range(len(unique)):
        print("id:" + str(unique[i]) + ", count:" + str(counts[i]))

    p3d = mlab.points3d(x, y, z, z, mode='point',figure=fig)
    #Lookup table for colour-correcting based on cluster
    p3d.module_manager.scalar_lut_manager.lut.number_of_colors = num_clusters
    p3d.module_manager.scalar_lut_manager.lut.table = lut
    mlab.show()
    sys.exit()

def plot_pcd_seabed(filename,inliers):
    pcd = o3d.io.read_point_cloud(filename)
    x = np.array(pcd.points)[:, 0]  # x position of point
    y = np.array(pcd.points)[:, 1]  # y position of point
    z = np.array(pcd.points)[:, 2]  # z position of point
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1920, 1080))

    # Colour correcting based on height
    #mlab.points3d(x,y,z,z,          # Values used for Color
    #                    mode="point",
    #                    colormap='gnuplot', # 'bone', 'copper', 'gnuplot'
    #                    color=(1,0,0),
    #                    figure=fig,
    #                    )

    # Plotting seabed plane
    print("Plotting PCD plus seabed...")
    try:
        inliers = np.loadtxt(inliers).astype(int)
        seabed = pcd.select_by_index(inliers)
        x_sb = np.array(seabed.points)[:, 0]  # x position of point
        y_sb = np.array(seabed.points)[:, 1]  # y position of point
        z_sb = np.array(seabed.points)[:, 2]  # z position of point
        #Plotting without seabed.
        mlab.points3d(x,y,z,z,          # Values used for Color
                            mode="point",
                            color=(0,0,1),
                            figure=fig,
                            )
        #Plotting seabed.
        mlab.points3d(x_sb,y_sb,z_sb,z_sb,          # Values used for Color
                            mode="point",
                            color=(1,0,0),
                            figure=fig,
                            )
    except Exception as e:
            f = open('log.txt', 'w')
            f.write('An exceptional thing happed - %s' % e)
            f.close()
    mlab.show()
    sys.exit()


#plot_pcd_seabed("../../data/xyz/identifiable_objects/0002_20210607_130418.utm.xyz.xyz","../../data/seabed/0002_20210607_130418_inliers.txt") 2 moorings small

plot_pcd_seabed("../../data/xyz/unknown_objects/0016_20210607_135027.utm.xyz.xyz","../../data/seabed/0016_20210607_135027_inliers.txt")


