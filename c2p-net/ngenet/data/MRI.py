import numpy as np
import os
import pickle
from glob import glob
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
CUR = os.path.dirname(os.path.abspath(__file__))
from ngenet.utils import npy2pcd, get_correspondences, normal
from sklearn.model_selection import train_test_split


class MRIDataset(Dataset):
    def __init__(self, root, split, aug, overlap_radius, noise_scale=0.005):
        super().__init__()

        self.root = root
        self.split = split
        self.aug = aug
        self.noise_scale = noise_scale
        self.overlap_radius = overlap_radius
        self.max_points = 3000

        assert self.split in ['train','val','test']

        if self.split == 'train':
            # Load the data from the pickle file
            with open('../DataPreparation/RANSACData/RANSACTraincropped.pickle', 'rb') as f:
                data = pickle.load(f)

            # Get the indices for train, test, and validation sets
            indices = np.arange(len(data['source']))  # assume all keys have the same length
            train_indices, test_val_indices = train_test_split(indices, test_size=0.4, random_state=42)
            test_indices, val_indices = train_test_split(test_val_indices, test_size=0.5, random_state=42)

            # Create the train, test, and validation sets for each key in the dictionary
            train_data, test_data, val_data = {}, {}, {}
            for key in data.keys():
                train_data[key] = [data[key][i] for i in train_indices]
                test_data[key] = [data[key][i] for i in test_indices]
                val_data[key] = [data[key][i] for i in val_indices]

            with open('./train_data.pickle', 'wb') as f:
                pickle.dump(train_data, f)
            with open('./test_data.pickle', 'wb') as f:
                pickle.dump(test_data, f)
            with open('./val_data.pickle', 'wb') as f:
                pickle.dump(val_data, f)
        
        with open(f'./{split}_data.pickle', 'rb') as f:
            self.infos = pickle.load(f)

    def __len__(self):
        return len(self.infos['source'])


    def __getitem__(self, item):
        # get pointcloud
        src_points = np.array(self.infos['source'][item])
        tgt_points = np.array(self.infos['target'][item])
        T = np.array(self.infos['transformation'][item])

        # for gpu memory
        if (src_points.shape[0] > self.max_points):
            idx = np.random.permutation(src_points.shape[0])[:self.max_points]
            src_points = src_points[idx]
        if (tgt_points.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_points.shape[0])[:self.max_points]
            tgt_points = tgt_points[idx]

        

        coors = get_correspondences(npy2pcd(src_points),
                                    npy2pcd(tgt_points),
                                    T,
                                    self.overlap_radius)
        
        src_feats = np.ones_like(src_points[:, :1], dtype=np.float32)
        tgt_feats = np.ones_like(tgt_points[:, :1], dtype=np.float32)

        src_pcd, tgt_pcd = normal(npy2pcd(src_points)), normal(npy2pcd(tgt_points))
        src_normals = np.array(src_pcd.normals).astype(np.float32) 
        tgt_normals = np.array(tgt_pcd.normals).astype(np.float32)

        pair = dict(
            src_points=src_points,
            tgt_points=tgt_points,
            src_feats=src_feats,
            tgt_feats=tgt_feats,
            src_normals=src_normals,
            tgt_normals=tgt_normals,
            transf=T,
            coors=coors,
            src_points_raw=src_points,
            tgt_points_raw=tgt_points)
        return pair
