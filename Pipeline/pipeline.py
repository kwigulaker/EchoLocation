from pcd_preprocess import PCD
from cluster import *
from gcn import *

params = {
    ## Downsampling parameters
    "threshold_random": 7000,
    "voxel_size": 10,

    ## Seabed approximation parameters
    "threshold_inliers": 0.15,
    "seabed_lower_bounds": 8000,
    "neighbor_radius": 0.75,

    ## Clustering parameters
    "dbscan_eps": 2,
    "dbscan_min_samples": 10,
    "distance_threshold": 2,
    "hdbscan_min_samples": 10,

    ## Classification parameters
}


if __name__ == "__main__":
    test_pcd = PCD("../EM2040/data/demo/shipwreck/original/0009_20220802_110307.utm.xyz.xyz")
    test_pcd.find_seabed_ransac(True,params["threshold_inliers"],params["seabed_lower_bounds"]) # Seabed filtering
    test_pcd.generateGraphNN(neighbor_radius=params["neighbor_radius"])
    getConnectedComponents(test_pcd) # Clustering
    test_pcd.plot2D(False,True) # Plotting
    np.savetxt("../EM2040/data/demo/shipwreck/0009_20220802_110307_seabed_inliers.xyz",test_pcd.seabed) # Save seabed inliers
    test_pcd.writeToClusters("../EM2040/data/demo/shipwreck/clusters/0016_20210607_135027") # Save outliers as multiple clustered pointclouds
    predictOnFiles("../EM2040/data/demo/shipwreck/clusters",'best.pt')