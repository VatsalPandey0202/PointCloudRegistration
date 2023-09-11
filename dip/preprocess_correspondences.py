import numpy as np
import open3d as o3d
import os
import h5py
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help= 'Dataset to use.', default='RANSAC', choices=['RANSAC','RigidCPD','AffineCPD','Non-RigidCPD'])
parser.add_argument('--type', type=str, help= 'Dataset to use.', default='original', choices=['cropped','original'])
args = parser.parse_args()

assert args.dataset in ['RANSAC','RigidCPD','AffineCPD','Non-RigidCPD']

# destination directory of preprocessed 3DMatch training data
dest_dir = 'data/'
if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)

if not os.path.isdir(os.path.join(dest_dir, 'correspondences')):
    os.mkdir(os.path.join(dest_dir, 'correspondences'))

if args.dataset == 'RANSAC':
    with open(f"../DataPreparation/RANSACData/RANSACTrain{args.type}.pickle", 'rb') as f:
        trn_data = pickle.load(f)
elif args.dataset == 'RigidCPD':
    with open(f"../DataPreparation/CPDData/CPDTrainRigid{args.type}.pickle", 'rb') as f:
        trn_data = pickle.load(f)
elif args.dataset == 'AffineCPD':
    with open(f"../DataPreparation/CPDData/CPDTrainAffine{args.type}.pickle", 'rb') as f:
        trn_data = pickle.load(f)
elif args.dataset == 'Non-RigidCPD':
    with open(f"../DataPreparation/CPDData/CPDTrainNon-Rigid{args.type}.pickle", 'rb') as f:
        trn_data = pickle.load(f)

hf_corr = h5py.File(os.path.join(dest_dir, 'correspondences', '{}.hdf5'.format('train')), 'w')

if args.type == 'original':
    threshold = 0.7
else : threshold = 0.03

for i in range(0,len(trn_data['source'])):
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(trn_data['source'][i])
        target.points = o3d.utility.Vector3dVector(trn_data['target'][i])
        # find correspondences
        result = o3d.pipelines.registration.registration_icp(source.transform(trn_data['transformation'][i]), target, threshold, np.eye(4),
                                                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
        if np.array(result.correspondence_set).shape[0]==0:
             print("no correspondece train")
             exit()
        pcd1_overlap_idx = np.asarray(result.correspondence_set)[:, 0]
        pcd2_overlap_idx = np.asarray(result.correspondence_set)[:, 1]
        hf_corr.create_dataset('{}'.format(i),
                            data=np.asarray(result.correspondence_set),
                            compression='gzip')


hf_corr.close()

if args.dataset == 'RANSAC':
    with open(f"../DataPreparation/RANSACData/RANSACTest{args.type}.pickle", 'rb') as f:
        tst_data = pickle.load(f)
elif args.dataset == 'RigidCPD':
    with open(f"../DataPreparation/CPDData/CPDTestRigid{args.type}.pickle", 'rb') as f:
        tst_data = pickle.load(f)
elif args.dataset == 'AffineCPD':
    with open(f"../DataPreparation/CPDData/CPDTestAffine{args.type}.pickle", 'rb') as f:
        tst_data = pickle.load(f)
elif args.dataset == 'Non-RigidCPD':
    with open(f"../DataPreparation/CPDData/CPDTestNon-Rigid{args.type}.pickle", 'rb') as f:
        tst_data = pickle.load(f)


hf_corr = h5py.File(os.path.join(dest_dir, 'correspondences', '{}.hdf5'.format('test')), 'w')
for i in range(0,len(tst_data['source'])):
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(tst_data['source'][i])
        target.points = o3d.utility.Vector3dVector(tst_data['target'][i])
        # find correspondences
        result = o3d.pipelines.registration.registration_icp(source.transform(tst_data['transformation'][i]), target, threshold, np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPoint())
        pcd1_overlap_idx = np.asarray(result.correspondence_set)[:, 0]
        pcd2_overlap_idx = np.asarray(result.correspondence_set)[:, 1]
        if np.array(result.correspondence_set).shape[0]==0:
             print("no correspondece test")
             exit()
        hf_corr.create_dataset('{}'.format(i),
                            data=np.asarray(result.correspondence_set),
                            compression='gzip')


hf_corr.close()
