import copy
import h5py
import math
import numpy as np
import os
import torch
import pickle
from torch.utils.data import Dataset
import sys
import open3d as o3d
from sklearn.model_selection import train_test_split
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)

from utils import  random_select_points, shift_point_cloud, jitter_point_cloud, \
    generate_random_rotation_matrix, generate_random_tranlation_vector, \
    transform, random_crop, shuffle_pc, random_scale_point_cloud, flip_pc



class RANSACOriginal(Dataset):
    def __init__(self, split, npts, ao=False,normal=False):
        super(RANSACOriginal, self).__init__()
        assert split in ['train', 'test']
        self.split = split
        self.npts = npts
        self.ao = ao # Asymmetric Objects
        self.normal = normal
        if self.split == 'train':
            with open('../../DataPreparation/RANSACData/RANSACTrainoriginal.pickle', 'rb') as f: #../../Data/Train.pickle
                data = pickle.load(f)
            # Get the indices for the train and test sets
            indices = np.arange(len(data['source']))  # assume all keys have the same length
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

            # Create the train and test sets for each key in the dictionary
            train_data, test_data = {}, {}
            for key in data.keys():
                train_data[key] = [data[key][i] for i in train_indices]
                test_data[key] = [data[key][i] for i in test_indices]
            self.data = train_data
            with open('../RANSACTestoriginal.pickle', 'wb') as f:
                pickle.dump(test_data, f)
        else:
            with open('../RANSACTestoriginal.pickle', 'rb') as f:
                self.data = pickle.load(f)

    def compose(self, item):
        src_cloud = np.array(self.data['source'][item]).astype('float32')
        tgt_cloud = np.array(self.data['target'][item]).astype('float32')

        src_cloud_normal = np.array(self.data['src_normals'][item]).astype('float32')
        tgt_cloud_normal = np.array(self.data['tgt_normals'][item]).astype('float32')

        T = np.array(self.data['transformation'][item]).astype('float32')
        R = T[:3,:3]
        t = T[:3,3]
        
        #RANSAC alignment
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)

        R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)
        
        src_cloud = np.concatenate([src_cloud, src_cloud_normal],axis=-1)
        tgt_cloud = np.concatenate([tgt_cloud, tgt_cloud_normal],axis=-1)
        
        src_cloud = random_select_points(src_cloud, m=self.npts)
        tgt_cloud = random_select_points(tgt_cloud, m=self.npts)

        tgt_cloud, src_cloud = shuffle_pc(tgt_cloud), shuffle_pc(src_cloud)
        
        return src_cloud, tgt_cloud, R, t

    def __getitem__(self, item):
        src_cloud, tgt_cloud, R, t = self.compose(item=item)
        if not self.normal:
            tgt_cloud, src_cloud = tgt_cloud[:, :3], src_cloud[:, :3]
        return tgt_cloud, src_cloud, R, t

    def __len__(self):
        return len(self.data['source'])
    

class RANSACCropped(Dataset):
    def __init__(self, split, npts, ao=False, normal=False):
        super(RANSACOriginal, self).__init__()
        assert split in ['train', 'test']
        self.split = split
        self.npts = npts
        self.ao = ao # Asymmetric Objects
        self.normal = normal
        if self.split == 'train':
            with open('../../DataPreparation/RANSACData/RANSACTraincropped.pickle', 'rb') as f: #../../Data/Train.pickle
                data = pickle.load(f)
            # Get the indices for the train and test sets
            indices = np.arange(len(data['source']))  # assume all keys have the same length
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

            # Create the train and test sets for each key in the dictionary
            train_data, test_data = {}, {}
            for key in data.keys():
                train_data[key] = [data[key][i] for i in train_indices]
                test_data[key] = [data[key][i] for i in test_indices]
            self.data = train_data
            with open('../RANSACTestcropped.pickle', 'wb') as f:
                pickle.dump(test_data, f)
        else:
            with open('../RANSACTestcropped.pickle', 'rb') as f:
                self.data = pickle.load(f)

    def compose(self, item):
        src_cloud = np.array(self.data['source'][item]).astype('float32')
        tgt_cloud = np.array(self.data['target'][item]).astype('float32')

        src_cloud_normal = np.array(self.data['src_normals'][item]).astype('float32')
        tgt_cloud_normal = np.array(self.data['tgt_normals'][item]).astype('float32')

        T = np.array(self.data['transformation'][item]).astype('float32')
        R = T[:3,:3]
        t = T[:3,3]
        
        #RANSAC alignment
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)

        R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)
        
        src_cloud = np.concatenate([src_cloud, src_cloud_normal],axis=-1)
        tgt_cloud = np.concatenate([tgt_cloud, tgt_cloud_normal],axis=-1)
        
        src_cloud = random_select_points(src_cloud, m=self.npts)
        tgt_cloud = random_select_points(tgt_cloud, m=self.npts)

        tgt_cloud, src_cloud = shuffle_pc(tgt_cloud), shuffle_pc(src_cloud)
        
        return src_cloud, tgt_cloud, R, t

    def __getitem__(self, item):
        src_cloud, tgt_cloud, R, t = self.compose(item=item)
        if not self.normal:
            tgt_cloud, src_cloud = tgt_cloud[:, :3], src_cloud[:, :3]
        return tgt_cloud, src_cloud, R, t

    def __len__(self):
        return len(self.data['source'])
    

class RigidCPDOriginal(Dataset):
    def __init__(self, split, npts, ao=False, normal=False):
        super(RANSACOriginal, self).__init__()
        assert split in ['train', 'test']
        self.split = split
        self.npts = npts
        self.ao = ao # Asymmetric Objects
        self.normal = normal
        if self.split == 'train':
            with open('../../DataPreparation/CPDData/CPDTrainRigidoriginal.pickle', 'rb') as f: #../../Data/Train.pickle
                data = pickle.load(f)
            # Get the indices for the train and test sets
            indices = np.arange(len(data['source']))  # assume all keys have the same length
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

            # Create the train and test sets for each key in the dictionary
            train_data, test_data = {}, {}
            for key in data.keys():
                train_data[key] = [data[key][i] for i in train_indices]
                test_data[key] = [data[key][i] for i in test_indices]
            self.data = train_data
            with open('../CPDTestRigidoriginal.pickle', 'wb') as f:
                pickle.dump(test_data, f)
        else:
            with open('../CPDTestRigidoriginal.pickle', 'rb') as f:
                self.data = pickle.load(f)

    def compose(self, item):
        src_cloud = np.array(self.data['source'][item]).astype('float32')
        tgt_cloud = np.array(self.data['target'][item]).astype('float32')

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(src_cloud)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(tgt_cloud)

        src_cloud_normal = np.array(source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))).astype('float32')
        tgt_cloud_normal = np.array(target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))).astype('float32')

        T = np.array(self.data['transformation'][item]).astype('float32')
        R = T[:3,:3]
        t = T[:3,3]
        
        #Alignment
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)

        R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)
        
        src_cloud = np.concatenate([src_cloud, src_cloud_normal],axis=-1)
        tgt_cloud = np.concatenate([tgt_cloud, tgt_cloud_normal],axis=-1)
        
        src_cloud = random_select_points(src_cloud, m=self.npts)
        tgt_cloud = random_select_points(tgt_cloud, m=self.npts)

        tgt_cloud, src_cloud = shuffle_pc(tgt_cloud), shuffle_pc(src_cloud)
        
        return src_cloud, tgt_cloud, R, t

    def __getitem__(self, item):
        src_cloud, tgt_cloud, R, t = self.compose(item=item)
        if not self.normal:
            tgt_cloud, src_cloud = tgt_cloud[:, :3], src_cloud[:, :3]
        return tgt_cloud, src_cloud, R, t

    def __len__(self):
        return len(self.data['source'])
    
class RigidCPDCropped(Dataset):
    def __init__(self, split, npts, ao=False, normal=False):
        super(RANSACOriginal, self).__init__()
        assert split in ['train', 'test']
        self.split = split
        self.npts = npts
        self.ao = ao # Asymmetric Objects
        self.normal = normal
        if self.split == 'train':
            with open('../../DataPreparation/CPDData/CPDTrainRigidcropped.pickle', 'rb') as f: #../../Data/Train.pickle
                data = pickle.load(f)
            # Get the indices for the train and test sets
            indices = np.arange(len(data['source']))  # assume all keys have the same length
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

            # Create the train and test sets for each key in the dictionary
            train_data, test_data = {}, {}
            for key in data.keys():
                train_data[key] = [data[key][i] for i in train_indices]
                test_data[key] = [data[key][i] for i in test_indices]
            self.data = train_data
            with open('../CPDTestRigidcropped.pickle', 'wb') as f:
                pickle.dump(test_data, f)
        else:
            with open('../CPDTestRigidcropped.pickle', 'rb') as f:
                self.data = pickle.load(f)

    def compose(self, item):
        src_cloud = np.array(self.data['source'][item]).astype('float32')
        tgt_cloud = np.array(self.data['target'][item]).astype('float32')

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(src_cloud)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(tgt_cloud)

        src_cloud_normal = np.array(source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))).astype('float32')
        tgt_cloud_normal = np.array(target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))).astype('float32')

        T = np.array(self.data['transformation'][item]).astype('float32')
        R = T[:3,:3]
        t = T[:3,3]
        
        #RANSAC alignment
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)

        R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)
        
        src_cloud = np.concatenate([src_cloud, src_cloud_normal],axis=-1)
        tgt_cloud = np.concatenate([tgt_cloud, tgt_cloud_normal],axis=-1)
        
        src_cloud = random_select_points(src_cloud, m=self.npts)
        tgt_cloud = random_select_points(tgt_cloud, m=self.npts)

        tgt_cloud, src_cloud = shuffle_pc(tgt_cloud), shuffle_pc(src_cloud)
        
        return src_cloud, tgt_cloud, R, t

    def __getitem__(self, item):
        src_cloud, tgt_cloud, R, t = self.compose(item=item)
        if not self.normal:
            tgt_cloud, src_cloud = tgt_cloud[:, :3], src_cloud[:, :3]
        return tgt_cloud, src_cloud, R, t

    def __len__(self):
        return len(self.data['source'])

class NonRigidCPDOriginal(Dataset):
    def __init__(self, split, npts, ao=False, normal=False):
        super(RANSACOriginal, self).__init__()
        assert split in ['train', 'test']
        self.split = split
        self.npts = npts
        self.ao = ao # Asymmetric Objects
        self.normal = normal
        if self.split == 'train':
            with open('../../DataPreparation/CPDData/CPDTrainNon-Rigidoriginal.pickle', 'rb') as f: #../../Data/Train.pickle
                data = pickle.load(f)
            # Get the indices for the train and test sets
            indices = np.arange(len(data['source']))  # assume all keys have the same length
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

            # Create the train and test sets for each key in the dictionary
            train_data, test_data = {}, {}
            for key in data.keys():
                train_data[key] = [data[key][i] for i in train_indices]
                test_data[key] = [data[key][i] for i in test_indices]
            self.data = train_data
            with open('../CPDTestNonRigidoriginal.pickle', 'wb') as f:
                pickle.dump(test_data, f)
        else:
            with open('../CPDTestNonRigidoriginal.pickle', 'rb') as f:
                self.data = pickle.load(f)

    def compose(self, item):
        src_cloud = np.array(self.data['source'][item]).astype('float32')
        tgt_cloud = np.array(self.data['target'][item]).astype('float32')

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(src_cloud)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(tgt_cloud)

        src_cloud_normal = np.array(source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))).astype('float32')
        tgt_cloud_normal = np.array(target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))).astype('float32')

        T = np.array(self.data['transformation'][item]).astype('float32')
        R = T[:3,:3]
        t = T[:3,3]
        
        #RANSAC alignment
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)

        R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)
        
        src_cloud = np.concatenate([src_cloud, src_cloud_normal],axis=-1)
        tgt_cloud = np.concatenate([tgt_cloud, tgt_cloud_normal],axis=-1)
        
        src_cloud = random_select_points(src_cloud, m=self.npts)
        tgt_cloud = random_select_points(tgt_cloud, m=self.npts)

        tgt_cloud, src_cloud = shuffle_pc(tgt_cloud), shuffle_pc(src_cloud)
        
        return src_cloud, tgt_cloud, R, t

    def __getitem__(self, item):
        src_cloud, tgt_cloud, R, t = self.compose(item=item)
        if not self.normal:
            tgt_cloud, src_cloud = tgt_cloud[:, :3], src_cloud[:, :3]
        return tgt_cloud, src_cloud, R, t

    def __len__(self):
        return len(self.data['source'])
    
class NonRigidCPDCropped(Dataset):
    def __init__(self, split, npts, ao=False, normal=False):
        super(RANSACOriginal, self).__init__()
        assert split in ['train', 'test']
        self.split = split
        self.npts = npts
        self.ao = ao # Asymmetric Objects
        self.normal = normal
        if self.split == 'train':
            with open('../../DataPreparation/CPDData/CPDTrainNon-Rigidcropped.pickle', 'rb') as f: #../../Data/Train.pickle
                data = pickle.load(f)
            # Get the indices for the train and test sets
            indices = np.arange(len(data['source']))  # assume all keys have the same length
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

            # Create the train and test sets for each key in the dictionary
            train_data, test_data = {}, {}
            for key in data.keys():
                train_data[key] = [data[key][i] for i in train_indices]
                test_data[key] = [data[key][i] for i in test_indices]
            self.data = train_data
            with open('../CPDTestNonRigidcropped.pickle', 'wb') as f:
                pickle.dump(test_data, f)
        else:
            with open('../CPDTestNonRigidcropped.pickle', 'rb') as f:
                self.data = pickle.load(f)

    def compose(self, item):
        src_cloud = np.array(self.data['source'][item]).astype('float32')
        tgt_cloud = np.array(self.data['target'][item]).astype('float32')

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(src_cloud)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(tgt_cloud)

        src_cloud_normal = np.array(source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))).astype('float32')
        tgt_cloud_normal = np.array(target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))).astype('float32')

        T = np.array(self.data['transformation'][item]).astype('float32')
        R = T[:3,:3]
        t = T[:3,3]
        
        #RANSAC alignment
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)

        R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)
        
        src_cloud = np.concatenate([src_cloud, src_cloud_normal],axis=-1)
        tgt_cloud = np.concatenate([tgt_cloud, tgt_cloud_normal],axis=-1)
        
        src_cloud = random_select_points(src_cloud, m=self.npts)
        tgt_cloud = random_select_points(tgt_cloud, m=self.npts)

        tgt_cloud, src_cloud = shuffle_pc(tgt_cloud), shuffle_pc(src_cloud)
        
        return src_cloud, tgt_cloud, R, t

    def __getitem__(self, item):
        src_cloud, tgt_cloud, R, t = self.compose(item=item)
        if not self.normal:
            tgt_cloud, src_cloud = tgt_cloud[:, :3], src_cloud[:, :3]
        return tgt_cloud, src_cloud, R, t

    def __len__(self):
        return len(self.data['source'])


class AffineCPDOriginal(Dataset):
    def __init__(self, split, npts, ao=False, normal=False):
        super(RANSACOriginal, self).__init__()
        assert split in ['train', 'test']
        self.split = split
        self.npts = npts
        self.ao = ao # Asymmetric Objects
        self.normal = normal
        if self.split == 'train':
            with open('../../DataPreparation/CPDData/CPDTrainAffineoriginal.pickle', 'rb') as f: #../../Data/Train.pickle
                data = pickle.load(f)
            # Get the indices for the train and test sets
            indices = np.arange(len(data['source']))  # assume all keys have the same length
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

            # Create the train and test sets for each key in the dictionary
            train_data, test_data = {}, {}
            for key in data.keys():
                train_data[key] = [data[key][i] for i in train_indices]
                test_data[key] = [data[key][i] for i in test_indices]
            self.data = train_data
            with open('../CPDTestAffineoriginal.pickle', 'wb') as f:
                pickle.dump(test_data, f)
        else:
            with open('../CPDTestAffineoriginal.pickle', 'rb') as f:
                self.data = pickle.load(f)

    def compose(self, item):
        src_cloud = np.array(self.data['source'][item]).astype('float32')
        tgt_cloud = np.array(self.data['target'][item]).astype('float32')

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(src_cloud)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(tgt_cloud)

        src_cloud_normal = np.array(source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))).astype('float32')
        tgt_cloud_normal = np.array(target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))).astype('float32')

        T = np.array(self.data['transformation'][item]).astype('float32')
        R = T[:3,:3]
        t = T[:3,3]
        
        #RANSAC alignment
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)

        R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)
        
        src_cloud = np.concatenate([src_cloud, src_cloud_normal],axis=-1)
        tgt_cloud = np.concatenate([tgt_cloud, tgt_cloud_normal],axis=-1)
        
        src_cloud = random_select_points(src_cloud, m=self.npts)
        tgt_cloud = random_select_points(tgt_cloud, m=self.npts)

        tgt_cloud, src_cloud = shuffle_pc(tgt_cloud), shuffle_pc(src_cloud)
        
        return src_cloud, tgt_cloud, R, t

    def __getitem__(self, item):
        src_cloud, tgt_cloud, R, t = self.compose(item=item)
        if not self.normal:
            tgt_cloud, src_cloud = tgt_cloud[:, :3], src_cloud[:, :3]
        return tgt_cloud, src_cloud, R, t

    def __len__(self):
        return len(self.data['source'])
    
class AffineCPDCropped(Dataset):
    def __init__(self, split, npts, ao=False, normal=False):
        super(RANSACOriginal, self).__init__()
        assert split in ['train', 'test']
        self.split = split
        self.npts = npts
        self.ao = ao # Asymmetric Objects
        self.normal = normal
        if self.split == 'train':
            with open('../../DataPreparation/CPDData/CPDTrainAffinecropped.pickle', 'rb') as f: #../../Data/Train.pickle
                data = pickle.load(f)
            # Get the indices for the train and test sets
            indices = np.arange(len(data['source']))  # assume all keys have the same length
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

            # Create the train and test sets for each key in the dictionary
            train_data, test_data = {}, {}
            for key in data.keys():
                train_data[key] = [data[key][i] for i in train_indices]
                test_data[key] = [data[key][i] for i in test_indices]
            self.data = train_data
            with open('../CPDTestAffinecropped.pickle', 'wb') as f:
                pickle.dump(test_data, f)
        else:
            with open('../CPDTestAffinecropped.pickle', 'rb') as f:
                self.data = pickle.load(f)

    def compose(self, item):
        src_cloud = np.array(self.data['source'][item]).astype('float32')
        tgt_cloud = np.array(self.data['target'][item]).astype('float32')

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(src_cloud)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(tgt_cloud)

        src_cloud_normal = np.array(source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))).astype('float32')
        tgt_cloud_normal = np.array(target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))).astype('float32')

        T = np.array(self.data['transformation'][item]).astype('float32')
        R = T[:3,:3]
        t = T[:3,3]
        
        #RANSAC alignment
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)

        R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
        src_cloud = transform(src_cloud, R, t)
        src_cloud_normal = transform(src_cloud_normal, R)
        
        src_cloud = np.concatenate([src_cloud, src_cloud_normal],axis=-1)
        tgt_cloud = np.concatenate([tgt_cloud, tgt_cloud_normal],axis=-1)
        
        src_cloud = random_select_points(src_cloud, m=self.npts)
        tgt_cloud = random_select_points(tgt_cloud, m=self.npts)

        tgt_cloud, src_cloud = shuffle_pc(tgt_cloud), shuffle_pc(src_cloud)
        
        return src_cloud, tgt_cloud, R, t

    def __getitem__(self, item):
        src_cloud, tgt_cloud, R, t = self.compose(item=item)
        if not self.normal:
            tgt_cloud, src_cloud = tgt_cloud[:, :3], src_cloud[:, :3]
        return tgt_cloud, src_cloud, R, t

    def __len__(self):
        return len(self.data['source'])