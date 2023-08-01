import numpy as np
import matplotlib.pyplot as plt
from pcd_preprocess import PCD

test_pcd = PCD("../EM2040/data/clusters/shipwrecks_xyz/0010_20220802_110640_1_1.xyz")
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(test_pcd.outliers[:,0], test_pcd.outliers[:,1], test_pcd.outliers[:,2], c=test_pcd.outliers[:,2],cmap='gnuplot',s=0.1)
plt.show()