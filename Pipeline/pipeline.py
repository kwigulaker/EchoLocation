from pcd_preprocess import PCD
from cluster import *
from gcn import *

params = {
    ## Downsampling parameters
    "threshold_random": 7000,
    "voxel_size": 10,

    ## Seabed approximation parameters
    "threshold_inliers": 0.05,
    "seabed_lower_bounds": 2000,
    "neighbor_radius": 0.70,

    ## Clustering parameters
    "noise_threshold": 50,
    "dbscan_eps": 2,
    "dbscan_min_samples": 10,
    "distance_threshold": 2,
    "hdbscan_min_samples": 50

}


if __name__ == "__main__":
    ## PIPELINE EXAMPLE
    # $ python kmall_to_xyz.py -xyz -c utm -e .xyz ../../data/source/field/kmall ../../data/source/field/xyz
    #test_pcd = PCD("../EM2040/data/report/field/ship/0022_20230505_081316_Pingeline.utm.xyz.xyz")
    test_pcd = PCD("../EM2040/data/demo/shipwreck/original/0009_20220802_110307.utm.xyz.xyz")
    #test_pcd = PCD("0002_20220708_112438_Shipname_inliers.xyz")
    test_pcd.find_seabed_ransac(True,params["threshold_inliers"],params["seabed_lower_bounds"]) # Seabed filtering
    test_pcd.generateGraphNN(neighbor_radius=params["neighbor_radius"])
    test_pcd.plot2D(True,False) # Plotting
    test_pcd = PCD("../EM2040/data/demo/shipwreck/original/0009_20220802_110307.utm.xyz.xyz")
    #test_pcd = PCD("0002_20220708_112438_Shipname_inliers.xyz")
    test_pcd.find_seabed_ransac(True,params["threshold_inliers"],params["seabed_lower_bounds"]) # Seabed filtering
    test_pcd.generateGraphDelaunay()
    test_pcd.plot2D(True,False) # Plotting
    #getConnectedComponents(test_pcd) # Clustering
    #clusterDBScan(test_pcd)
    #test_pcd.plot2D(False,True) # Plotting
    #np.savetxt("seabed_inliers.xyz",test_pcd.seabed) # Save seabed inliers
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_outliers_test.xyz",test_pcd.outliers) # Save seabed outliers
    #test_pcd.writeToClusters("../EM2040/data/clusters/all/0009_20220802_110307/0",params["noise_threshold"]) # Save outliers as multiple clustered pointclouds
    #predictOnFiles("../EM2040/data/report/field/ship/clusters",'best.pt')

    ## RANSAC TESTING
    #inlier_test = 0.10
    #test_pcd = PCD("../EM2040/data/demo/moorings_large/original/0001_20210607_130222.utm.xyz.xyz")
    #test_pcd.find_seabed_ransac(False,inlier_test,params["seabed_lower_bounds"]) # Seabed filtering
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_inliers_0100.xyz",test_pcd.seabed) # Save seabed inliers
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_outliers_0100.xyz",test_pcd.outliers) # Save seabed outliers
    #inlier_test = 0.125
    #test_pcd = PCD("../EM2040/data/demo/moorings_large/original/0001_20210607_130222.utm.xyz.xyz")
    #test_pcd.find_seabed_ransac(False,inlier_test,params["seabed_lower_bounds"]) # Seabed filtering
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_inliers_0125.xyz",test_pcd.seabed) # Save seabed inliers
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_outliers_0125.xyz",test_pcd.outliers) # Save seabed outliers
    #inlier_test = 0.15
    #est_pcd = PCD("../EM2040/data/demo/moorings_large/original/0001_20210607_130222.utm.xyz.xyz")
    #test_pcd.find_seabed_ransac(False,inlier_test,params["seabed_lower_bounds"]) # Seabed filtering
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_inliers_015.xyz",test_pcd.seabed) # Save seabed inliers
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_outliers_015.xyz",test_pcd.outliers) # Save seabed outliers
    #inlier_test = 0.175
    #test_pcd = PCD("../EM2040/data/demo/moorings_large/original/0001_20210607_130222.utm.xyz.xyz")
    #test_pcd.find_seabed_ransac(False,inlier_test,params["seabed_lower_bounds"]) # Seabed filtering
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_inliers_0175.xyz",test_pcd.seabed) # Save seabed inliers
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_outliers_0175.xyz",test_pcd.outliers) # Save seabed outliers
    #inlier_test = 0.2
    #test_pcd = PCD("../EM2040/data/demo/moorings_large/original/0001_20210607_130222.utm.xyz.xyz")
    #test_pcd.find_seabed_ransac(False,inlier_test,params["seabed_lower_bounds"]) # Seabed filtering
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_inliers_02.xyz",test_pcd.seabed) # Save seabed inliers
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_outliers_02.xyz",test_pcd.outliers) # Save seabed outliers

    ## SEABED INLIER THRESH TESTING
    #test_seabed = 1000
    #test_pcd = PCD("../EM2040/data/demo/moorings_large/original/0001_20210607_130222.utm.xyz.xyz")
    #test_pcd.find_seabed_ransac(True,params["threshold_inliers"],test_seabed) # Seabed filtering
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_inliers_3000_nsf.xyz",test_pcd.seabed) # Save seabed inliers
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_outliers_3000_nsf.xyz",test_pcd.outliers) # Save seabed outliers

    #test_seabed = 3500
    #test_pcd = PCD("../EM2040/data/demo/moorings_large/original/0001_20210607_130222.utm.xyz.xyz")
    #test_pcd.find_seabed_ransac(True,params["threshold_inliers"],test_seabed) # Seabed filtering
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_inliers_5000_nsf.xyz",test_pcd.seabed) # Save seabed inliers
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_outliers_5000_nsf.xyz",test_pcd.outliers) # Save seabed outliers

    #test_seabed = 7000
    #test_pcd = PCD("../EM2040/data/demo/moorings_large/original/0001_20210607_130222.utm.xyz.xyz")
    #test_pcd.find_seabed_ransac(True,params["threshold_inliers"],test_seabed) # Seabed filtering
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_inliers_7000_nsf.xyz",test_pcd.seabed) # Save seabed inliers
    #np.savetxt("../EM2040/data/report/moorings_large/seabed_outliers_7000_nsf.xyz",test_pcd.outliers) # Save seabed outliers

    ## DOWNSAMPLING
    #test_seabed = 1000
    #test_pcd = PCD("../EM2040/data/demo/shipwreck/original/0009_20220802_110307.utm.xyz.xyz")
    #test_pcd.find_seabed_ransac(True,params["threshold_inliers"],test_seabed) # Seabed filtering
    #np.savetxt("../EM2040/data/report/shipwreck/seabed_inliers_ds.xyz",test_pcd.seabed) # Save seabed inliers
    #np.savetxt("../EM2040/data/report/shipwreck/seabed_outliers_ds.xyz",test_pcd.outliers) # Save seabed outliers