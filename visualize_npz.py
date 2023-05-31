import numpy as np
# import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from mpl_toolkits.mplot3d import axes3d

a = np.load("D:/Projects/Datasets/shapenet/02747177/1b7d468a27208ee3dad910e221d16b18/input_4.npz")
gt_file = np.load('D:/Projects/Datasets/shapenet\\03207941/5d17e90f512a3dc7df3a1b0d597ce76e\\gt.npz')

print(a.files)

voxels = a['voxels']
tsdf = a['tsdf']

print(type(voxels))
print((type(tsdf)))

_truncation = 2.5
gt_orig = tsdf * 32
gt = tsdf * 32

gt[np.where(gt>_truncation)] = _truncation
gt[np.where(gt<-1*_truncation)] = -1*_truncation

print(gt.shape)

x = np.arange(gt.shape[0])[:, None, None]
y = np.arange(gt.shape[1])[None, :, None]
z = np.arange(gt.shape[2])[None, None, :]
x, y, z = np.broadcast_arrays(x, y, z)

c = np.tile(gt.ravel()[:, None], [1, 3])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=y.squeeze())
plt.show()


# ################################
# tmp = tsdf.shape[0]
# x = np.arange(tsdf.shape[0])[:, None, None]
# y = np.arange(tsdf.shape[1])[None, :, None]
# z = np.arange(tsdf.shape[2])[None, None, :]
# x, y, z = np.broadcast_arrays(x, y, z)
#
# c = np.tile(tsdf.ravel()[:, None], [1, 3])
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=y.squeeze())
# plt.show()

#################################


# x = np.arange(voxels.shape[0])[:, None, None]
# y = np.arange(voxels.shape[1])[None, :, None]
# z = np.arange(voxels.shape[2])[None, None, :]
# x, y, z = np.broadcast_arrays(x, y, z)
#
# c = np.tile(voxels.ravel()[:, None], [1, 3])
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=c)
# plt.show()

#################################################

    # img = Image.fromarray(b)
    # if img.mode != 'RGB':
    #     img = img.convert('RGB')
    
    # name = "pics/file_{}".format(i)
    # img.save(name+".jpeg")


# plt.imshow(b)

# pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
# pcd_o3d.points = o3d.utility.Vector3dVector(a['tsdf'][0][0])  # set pcd_np as the point cloud points


# # Visualize:
# o3d.visualization.draw_geometries([pcd_o3d])