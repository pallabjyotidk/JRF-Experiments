# python voxel_vis.py
import os
import glob
import copy
import vis_numpy as np
import torch
from torch.utils.data import Dataset
from testFile import ReadOFF

class ShapenetDataset(Dataset):
    def __init__(self, file_name, data_path, res=32, truncation=2.5):
        # path parameters
        self._file = file_name
        self._data_path = data_path
        # read data
        self._data_pairs = []
        self._mask_names = []
        self._bbox = []
        self._truncation = truncation
        self._res = res
        self._read_data()

    def get_bbox(self, gt):
        x1 = 0
        x2 = self._res -1
        y1 = 0
        y2 = self._res -1
        z1 = 0
        z2 = self._res -1
        for i in range(self._res):
            if len(np.where(gt[i, :, :]<= 0)[0]) > 0:
                x1 = i
                break
        for i in range(self._res):
            if len(np.where(gt[:, i, :]<= 0)[0]) > 0:
                y1 = i
                break
        for i in range(self._res):
            if len(np.where(gt[:, :, i]<= 0)[0]) > 0:
                z1 = i
                break
        for i in range(self._res - 1, 0, -1):
            if len(np.where(gt[i, :, :]<= 0)[0]) > 0:
                x2 = i
                break
        for i in range(self._res - 1, 0, -1):
            if len(np.where(gt[:, i, :]<= 0)[0]) > 0:
                y2 = i
                break
        for i in range(self._res - 1, 0, -1):
            if len(np.where(gt[:, :, i]<= 0)[0]) > 0:
                z2 = i
                break
        bbox = [[x1, x2], [y1, y2], [z1, z2]]
        return bbox

    def _read_data(self):
        with open(self._file, 'r') as data:
            print (self._file)
            gt_files = data.readlines()
            for gt_file in gt_files:
                model_path = os.path.join(self._data_path, gt_file.split('\n')[0])
                gt_file = os.path.join(model_path, "gt.npz")
                with np.load(gt_file, 'rb') as data:
                    gt = data["tsdf"] * self._res # convert to voxel unit
                    gt[np.where(gt>self._truncation)] = self._truncation
                    gt[np.where(gt<-1*self._truncation)] = -1*self._truncation
                    bbox = self.get_bbox(gt)
                for input_file in glob.glob(os.path.join(model_path, "input*.npz")):
                    with np.load(input_file, 'rb') as data:
                        inputs = data["tsdf"] * self._res
                        inputs[np.where(inputs>self._truncation)] = self._truncation
                        inputs[np.where(inputs<-1 * self._truncation)] = -1 * self._truncation
                        self._data_pairs.append([inputs, gt])
                        self._mask_names.append(input_file)
                        self._bbox.append(np.array(bbox))
        # print (len(self._data_pairs))

    def __len__(self):
        return len(self._data_pairs)

    def __getitem__(self, idx):
        name = self._mask_names[idx]
        input_sdf = copy.deepcopy(self._data_pairs[idx][0])
        gt_sdf = copy.deepcopy(self._data_pairs[idx][1])
        bbox = torch.from_numpy(self._bbox[idx]).unsqueeze(0).float()
        # get final data
        input_sdf = torch.from_numpy(input_sdf).unsqueeze(0).float()
        gt_sdf = torch.from_numpy(gt_sdf).unsqueeze(0).float()
        return [input_sdf, gt_sdf, bbox, name]


data_path = 'D:/Projects/JRF/random/patchComplete/data/shapenet'
train_file = 'txt_files/shapenet_train.txt'

# trainSet = ShapenetDataset(train_file, data_path)

vertices, faces = ReadOFF.readFile('D:/Projects/JRF/random/patchComplete/data/shapenet/03207941/3bee2133518d4a2c1f3168043fdd9df6/gt.off')
print(len(vertices), len(faces))
print(len(vertices[0]), len(faces[0]))
print(vertices[0], faces[0])
