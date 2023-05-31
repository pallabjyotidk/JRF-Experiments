import numpy as np
# import open3d as o3d
import os, glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from mpl_toolkits.mplot3d import axes3d

# gt_file = np.load('D:/Projects/Datasets/shapenet\\03207941/5d17e90f512a3dc7df3a1b0d597ce76e\\gt.npz')

# bbox = [[4, 27], [0, 31], [4, 27]]
# model_path = 'D:/Projects/Datasets/shapenet\\03207941/5d17e90f512a3dc7df3a1b0d597ce76e'
# print(a.files)
#
a = np.load('D:/Projects/Datasets/shapenet/03207941/5d17e90f512a3dc7df3a1b0d597ce76e/gt.npz')
voxels = a['voxels']
# tsdf = a['tsdf']

_truncation = 2.5
_res = 32
_data_pairs = []
_mask_names = []
_bbox = []

model_path = 'D:/Projects/Datasets/shapenet/03207941/ba66302db9cfa0147286af1ad775d13a'
prior_path = 'D:/Projects/JRF/related_code/shape_completion/PatchComplete/priors'

gt_file = os.path.join(model_path, "gt.npz")


def vis(obj, bbox):
    obj = np.array(obj)
    x = np.arange(obj.shape[0])[:, None, None]
    y = np.arange(obj.shape[1])[None, :, None]
    z = np.arange(obj.shape[2])[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)

    c = np.tile(obj.ravel()[:, None], [1, 3])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x.ravel(), y.ravel(), z.ravel())
    ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=c)
    ax.plot([2, 29], [0, 31], [7, 27])
    # ax.legend()
    plt.show()


def _get_data():
    priors = []
    for prior_file in glob.iglob(os.path.join(prior_path, "*.npy")):
        with open(prior_file, 'rb') as data:
            prior = np.load(data)
            prior[np.where(prior > _truncation)] = _truncation
            prior[np.where(prior < -1 * _truncation)] = -1 * _truncation
            priors.append(prior)
    return priors

def get_bbox(gt, _res=32):
    x1 = 0
    x2 = _res - 1
    y1 = 0
    y2 = _res - 1
    z1 = 0
    z2 = _res - 1
    for i in range(_res):
        tmp = gt[i, :, :]
        tmp_1 = np.where(gt[i, :, :] <= 0)[0]
        tmp_2 = gt[0]
        if len(np.where(gt[i, :, :] <= 0)[0]) > 0:
            x1 = i
            break
    for i in range(_res):
        if len(np.where(gt[:, i, :] <= 0)[0]) > 0:
            y1 = i
            break
    for i in range(_res):
        if len(np.where(gt[:, :, i] <= 0)[0]) > 0:
            z1 = i
            break
    for i in range(_res - 1, 0, -1):
        if len(np.where(gt[i, :, :] <= 0)[0]) > 0:
            x2 = i
            break
    for i in range(_res - 1, 0, -1):
        if len(np.where(gt[:, i, :] <= 0)[0]) > 0:
            y2 = i
            break
    for i in range(_res - 1, 0, -1):
        if len(np.where(gt[:, :, i] <= 0)[0]) > 0:
            z2 = i
            break
    bbox = [[x1, x2], [y1, y2], [z1, z2]]
    return bbox

with np.load(gt_file, 'rb') as data:
    gt = data["tsdf"] * _res # convert to voxel unit
    print(type(gt))
    gt[np.where(gt>_truncation)] = _truncation
    gt[np.where(gt<-1*_truncation)] = -1*_truncation
    bbox = get_bbox(gt)
    priors = _get_data()
    vis(voxels, bbox)

for input_file in glob.glob(os.path.join(model_path, "input*.npz")):
    with np.load(input_file, 'rb') as data:
        inputs = data["tsdf"] * _res
        inputs[np.where(inputs>_truncation)] = _truncation
        inputs[np.where(inputs<-1 * _truncation)] = -1 * _truncation
        _data_pairs.append([inputs, gt])
        _mask_names.append(input_file)
        _bbox.append(np.array(bbox))

# print(_bbox)

# print(type(voxels))
# print((type(tsdf)))

# _truncation = 2.5
# gt_orig = tsdf * 32
# gt = tsdf * 32

# gt[np.where(gt>_truncation)] = _truncation
# gt[np.where(gt<-1*_truncation)] = -1*_truncation
#
# print(gt.shape)



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