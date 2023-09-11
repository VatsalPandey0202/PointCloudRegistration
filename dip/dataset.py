import os
import numpy as np
import torch.utils.data as data
import torch
import h5py
import random
from scipy.spatial.transform import Rotation as rotation


class Dataset(data.Dataset):

    def __init__(self,dataset, aug=True):
        self.dataset = dataset
        if self.dataset == 'train':
            self.hf_patches = h5py.File("data/patches_lrf/train.hdf5", 'r')
        else:
            self.hf_patches = h5py.File("data/patches_lrf/test.hdf5", 'r')
        self.length = len(list(self.hf_patches.keys()))
        self.hf_patches.close()
        self.do_data_aug = aug

    def __getitem__(self, index):
        if self.dataset == 'train':
            self.hf_patches = h5py.File("data/patches_lrf/train.hdf5", 'r')
            self.hf_points = h5py.File("data/points_lrf/train.hdf5", 'r')
            self.hf_rotations = h5py.File("data/rotations_lrf/train.hdf5", 'r')
            self.hf_lrfs = h5py.File("data/lrfs/train.hdf5", 'r')
            
        else:
            self.hf_patches = h5py.File("data/patches_lrf/test.hdf5", 'r')
            self.hf_points = h5py.File("data/points_lrf/test.hdf5", 'r')
            self.hf_rotations = h5py.File("data/rotations_lrf/test.hdf5", 'r')
            self.hf_lrfs = h5py.File("data/lrfs/test.hdf5", 'r')
        
        patches = np.asarray(self.hf_patches[str(index)])
        frag1_batch = patches[0]
        frag2_batch = patches[1]

        rotations = np.asarray(self.hf_rotations[str(index)])
        R1 = rotations[0]
        R2 = rotations[1]


        frag1_batch = torch.Tensor(frag1_batch)
        frag2_batch = torch.Tensor(frag2_batch)

        points = np.asarray(self.hf_points[str(index)])
        fps_pcd1_pts = torch.Tensor(points[0])
        fps_pcd2_pts = torch.Tensor(points[1])

        lrfs = np.asarray(self.hf_lrfs[str(index)])
        lrf1 = torch.Tensor(lrfs[0])
        lrf2 = torch.Tensor(lrfs[1])

        self.hf_patches.close()
        self.hf_points.close()
        self.hf_rotations.close()
        self.hf_lrfs.close()


        return frag1_batch, frag2_batch, fps_pcd1_pts, fps_pcd2_pts, torch.Tensor(R1), torch.Tensor(R2), lrf1, lrf2


    def __len__(self):
        return self.length


    def get_length(self):
        return self.length