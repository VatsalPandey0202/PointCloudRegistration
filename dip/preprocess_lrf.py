import numpy as np
import open3d as o3d
import os
import h5py
from torch_cluster import fps
import torch
from lrf import lrf
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help= 'Dataset to use.', default='RANSAC', choices=['RANSAC','RigidCPD','AffineCPD','Non-RigidCPD'])
parser.add_argument('--type', type=str, help= 'Dataset to use.', default='original', choices=['cropped','original'])
args = parser.parse_args()

assert args.dataset in ['RANSAC','RigidCPD','AffineCPD','Non-RigidCPD']

'''
Notes:
1. LRF and patch preprocessing assumes that correspondences are given or preprocessed
2. to preprocess correspondences use preprocess_correspondences.py
'''

do_save = True

# training and testing parameters for 3DMatch
pts_to_sample = 2048
batch_size = 256
patch_size = 256
if args.type == 'original':
    lrf_kernel = 3.0 * np.sqrt(3) 
    voxel_size = 1.0 
else: 
    lrf_kernel = 0.5 * np.sqrt(3) 
    voxel_size = 0.01 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dest_dir = 'data/'

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

# load preprocessed correspondences, i.e. set of indices of the corresponding 3D points between two point clouds
hf_corrs = h5py.File(os.path.join(dest_dir, 'correspondences', '{}.hdf5'.format('train')), 'r')
corrs_to_test = np.asarray(list(hf_corrs.keys()))

if do_save:
    if not os.path.isdir(os.path.join(dest_dir, 'patches_lrf')):
        os.mkdir(os.path.join(dest_dir, 'patches_lrf'))
    if not os.path.isdir(os.path.join(dest_dir, 'points_lrf')):
        os.mkdir(os.path.join(dest_dir, 'points_lrf'))
    if not os.path.isdir(os.path.join(dest_dir, 'rotations_lrf')):
        os.mkdir(os.path.join(dest_dir, 'rotations_lrf'))
    if not os.path.isdir(os.path.join(dest_dir, 'lrfs')):
        os.mkdir(os.path.join(dest_dir, 'lrfs'))

    hf_patches = h5py.File(os.path.join(dest_dir, 'patches_lrf', '{}.hdf5'.format('train')), 'w')
    hf_points = h5py.File(os.path.join(dest_dir, 'points_lrf', '{}.hdf5'.format('train')), 'w')
    hf_rotations = h5py.File(os.path.join(dest_dir, 'rotations_lrf', '{}.hdf5'.format('train')), 'w')
    hf_lrfs = h5py.File(os.path.join(dest_dir, 'lrfs', '{}.hdf5'.format('train')), 'w')

for j in range(len(corrs_to_test)):

    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(trn_data['source'][j])
    target.points = o3d.utility.Vector3dVector(trn_data['target'][j])

    T = np.asarray(trn_data['transformation'][j])
    T_inv = np.linalg.inv(T)
    corrs = np.asarray(hf_corrs[str(j)])

    # select only corresponding points
    pcd1_corr = source.select_by_index(corrs[:, 0])#select_down_sample
    pcd2_corr = target.select_by_index(corrs[:, 1])#select_down_sample(corrs[:, 1])

    pcd1_corr = pcd1_corr.voxel_down_sample(voxel_size)
    pcd2_corr = pcd2_corr.voxel_down_sample(voxel_size)


    # apply ground truth transformation to bring them in the same reference frame
    pcd1_corr.transform(T)
    
    # FPS
    tensor_pcd1_frag = torch.Tensor(np.asarray(pcd1_corr.points)).to(device)
    fps_pcd1_idx = fps(tensor_pcd1_frag,
                        ratio=batch_size / tensor_pcd1_frag.shape[0],
                        random_start=True)

    _pcd2_frag_tree = o3d.geometry.KDTreeFlann(pcd2_corr)

    fps_pcd1_pts = np.asarray(pcd1_corr.points)[fps_pcd1_idx.cpu()]

    fps_pcd2_idx = torch.empty(fps_pcd1_idx.shape, dtype=int)

    # find nearest neighbors on the other point cloud
    for i, pt in enumerate(fps_pcd1_pts):
        _, patch_idx, _ = _pcd2_frag_tree.search_knn_vector_xd(pt, 1)
        fps_pcd2_idx[i] = patch_idx[0]

    # transform point clouds back to their reference frame using ground truth
    # (this is important because the network must learn when point clouds are in their original reference frame)
    pcd1_corr.transform(T_inv)

    # extract patches and compute LRFs
    patches1_batch = np.empty((batch_size, 3, patch_size))
    patches2_batch = np.empty((batch_size, 3, patch_size))

    lrfs1_batch = np.empty((batch_size, 4, 4))
    lrfs2_batch = np.empty((batch_size, 4, 4))

    frag1_lrf = lrf(pcd=source,
                    pcd_tree=o3d.geometry.KDTreeFlann(source),
                    patch_size=patch_size,
                    lrf_kernel=lrf_kernel,
                    viz=False)

    frag2_lrf = lrf(pcd=target,
                    pcd_tree=o3d.geometry.KDTreeFlann(target),
                    patch_size=patch_size,
                    lrf_kernel=lrf_kernel,
                    viz=False)

    # GET PATCHES USING THE FPS POINTS
    for i in range(len(fps_pcd1_idx)):

        pt1 = np.asarray(pcd1_corr.points)[fps_pcd1_idx.cpu()[i]]
        pt2 = np.asarray(pcd2_corr.points)[fps_pcd2_idx.cpu()[i]]

        frag1_lrf_pts, _, lrf1 = frag1_lrf.get(pt1)
        frag2_lrf_pts, _, lrf2 = frag2_lrf.get(pt2)

        patches1_batch[i] = frag1_lrf_pts.T
        patches2_batch[i] = frag2_lrf_pts.T

        lrfs1_batch[i] = lrf1
        lrfs2_batch[i] = lrf2

    if do_save:
        hf_lrfs.create_dataset('{}'.format(j),
                                    data=np.asarray([lrfs1_batch, lrfs2_batch]),
                                    compression='gzip')

        hf_patches.create_dataset('{}'.format(j),
                                    data=np.asarray([patches1_batch, patches2_batch]),
                                    compression='gzip')

        hf_points.create_dataset('{}'.format(j),
                                    data=np.asarray([np.asarray(pcd1_corr.points)[fps_pcd1_idx.cpu()],
                                                    np.asarray(pcd2_corr.points)[fps_pcd2_idx.cpu()]]),
                                    compression='gzip')

        hf_rotations.create_dataset('{}'.format(j),
                                    data=np.asarray([T[:3, :3], T_inv[:3, :3]]),
                                    compression='gzip')

if do_save:
    hf_patches.close()
    hf_points.close()
    hf_rotations.close()
    hf_lrfs.close()

print("train complete")

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


# load preprocessed correspondences, i.e. set of indices of the corresponding 3D points between two point clouds
hf_corrs = h5py.File(os.path.join(dest_dir, 'correspondences', '{}.hdf5'.format('test')), 'r')
corrs_to_test = np.asarray(list(hf_corrs.keys()))

if do_save:
    if not os.path.isdir(os.path.join(dest_dir, 'patches_lrf')):
        os.mkdir(os.path.join(dest_dir, 'patches_lrf'))
    if not os.path.isdir(os.path.join(dest_dir, 'points_lrf')):
        os.mkdir(os.path.join(dest_dir, 'points_lrf'))
    if not os.path.isdir(os.path.join(dest_dir, 'rotations_lrf')):
        os.mkdir(os.path.join(dest_dir, 'rotations_lrf'))
    if not os.path.isdir(os.path.join(dest_dir, 'lrfs')):
        os.mkdir(os.path.join(dest_dir, 'lrfs'))

    hf_patches = h5py.File(os.path.join(dest_dir, 'patches_lrf', '{}.hdf5'.format('test')), 'w')
    hf_points = h5py.File(os.path.join(dest_dir, 'points_lrf', '{}.hdf5'.format('test')), 'w')
    hf_rotations = h5py.File(os.path.join(dest_dir, 'rotations_lrf', '{}.hdf5'.format('test')), 'w')
    hf_lrfs = h5py.File(os.path.join(dest_dir, 'lrfs', '{}.hdf5'.format('test')), 'w')

for j in range(len(corrs_to_test)):

    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(tst_data['source'][j])
    target.points = o3d.utility.Vector3dVector(tst_data['target'][j])

    T = np.asarray(trn_data['transformation'][j])
    T_inv = np.linalg.inv(T)
    
    source = source.voxel_down_sample(voxel_size)
    target = target.voxel_down_sample(voxel_size)

    # pick 5000 random points from the two point clouds
    inds1 = np.random.choice(np.asarray(source.points).shape[0], pts_to_sample, replace=False)
    inds2 = np.random.choice(np.asarray(target.points).shape[0], pts_to_sample, replace=False)

    pcd1_pts = np.asarray(source.points)[inds1]
    pcd2_pts = np.asarray(target.points)[inds2]

    frag1_lrf = lrf(pcd=source,
                    pcd_tree=o3d.geometry.KDTreeFlann(source),
                    patch_size=patch_size,
                    lrf_kernel=lrf_kernel,
                    viz=False)

    frag2_lrf = lrf(pcd=target,
                    pcd_tree=o3d.geometry.KDTreeFlann(target),
                    patch_size=patch_size,
                    lrf_kernel=lrf_kernel,
                    viz=False)
    
    patches1_batch = np.empty((pcd1_pts.shape[0], 3, patch_size))
    patches2_batch = np.empty((pcd2_pts.shape[0], 3, patch_size))

    lrfs1_batch = np.empty((pcd1_pts.shape[0], 4, 4))
    lrfs2_batch = np.empty((pcd2_pts.shape[0], 4, 4))

    for i in range(pcd1_pts.shape[0]):
        frag1_lrf_pts, _, lrf1 = frag1_lrf.get(pcd1_pts[i])
        frag2_lrf_pts, _, lrf2 = frag2_lrf.get(pcd2_pts[i])

        patches1_batch[i] = frag1_lrf_pts.T
        patches2_batch[i] = frag2_lrf_pts.T

        lrfs1_batch[i] = lrf1
        lrfs2_batch[i] = lrf2

    if do_save:
        hf_lrfs.create_dataset('{}'.format(j),
                                    data=np.asarray([lrfs1_batch, lrfs2_batch]),
                                    compression='gzip')

        hf_patches.create_dataset('{}'.format(j),
                                    data=np.asarray([patches1_batch, patches2_batch]),
                                    compression='gzip')

        hf_points.create_dataset('{}'.format(j),
                                    data=np.asarray([np.asarray(pcd1_corr.points)[fps_pcd1_idx.cpu()],
                                                    np.asarray(pcd2_corr.points)[fps_pcd2_idx.cpu()]]),
                                    compression='gzip')

        hf_rotations.create_dataset('{}'.format(j),
                                    data=np.asarray([T[:3, :3], T_inv[:3, :3]]),
                                    compression='gzip')

if do_save:
    hf_patches.close()
    hf_points.close()
    hf_rotations.close()
    hf_lrfs.close()

   



