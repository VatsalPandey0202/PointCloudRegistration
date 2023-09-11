# %%
# IMPORTS FOR NGENET AND DEFORMATION PYRAMID
import os
from glob import glob
from pickle import load, dump
import argparse

import numpy as np
import open3d as o3d
import torch
import trimesh as trm
import yaml
import time
from scipy.spatial.distance import cdist
from statistics import mean
from easydict import EasyDict as edict

from deformationpyramid.model.geometry import *
from deformationpyramid.model.loss import compute_truncated_chamfer_distance
from deformationpyramid.model.registration import Registration
from tqdm import tqdm
from ngenet.data.MRI import MRIDataset
from ngenet.data import get_dataloader
from ngenet.models import NgeNet, architectures, vote
from ngenet.utils import (decode_config, execute_global_registration, get_blue,
                          get_correspondences, get_yellow, npy2feat, npy2pcd, pcd2npy,
                          setup_seed, to_tensor, vis_plys)
from deformationpyramid.utils.benchmark_utils import setup_seed
from deformationpyramid.utils.tiktok import Timers

# %%
def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])
yaml.add_constructor('!join', join)

parser = argparse.ArgumentParser()

def transform_point_cloud(point_cloud, transformation_matrix):
    # Add homogeneous coordinates to the point cloud
    homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # Apply the transformation matrix
    transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T

    # Remove homogeneous coordinates
    transformed_points = transformed_points[:, :3]

    return transformed_points.astype(float)


parser.add_argument('--checkpoint', type=str, default='trainResults/mri/checkpoints/best_loss.pth', required=False, help='path to NgeNet checkpoint')
parser.add_argument('--ngenet_config_path', type=str, default='config/MRI.yaml', help='which configuration file to use for NgeNet')
parser.add_argument('--ndp_config_path', type=str, default='config/NDP.yaml', help='which configuration file to use for NDP')
parser.add_argument('--vis', action='store_true',  default=False, help='visualize output while running in an extra window')
parser.add_argument('--no_cuda', action='store_true',  default=False, help='disable cuda (cpu only)')
args = parser.parse_args()


# %%
setup_seed(22)
config = decode_config(args.ngenet_config_path)
config = edict(config)
config.architecture = architectures[config.dataset]
config.num_workers = 0



test_dataset_real = MRIDataset(
            root=config.root,
            split='test',
            aug=False,
            overlap_radius=config.overlap_radius
            )

test_dataloader, neighborhood_limits = get_dataloader(
    config=config,
    dataset=test_dataset_real,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=False,
    neighborhood_limits=None
)


with open(args.ndp_config_path,'r') as f:
    p_config = yaml.load(f, Loader=yaml.Loader)

p_config = edict(p_config)


# %%
model_rigid = NgeNet(config)
use_cuda = not args.no_cuda
if use_cuda:
    model = model_rigid.cuda()
    if args.checkpoint != None:
        model_rigid.load_state_dict(torch.load(args.checkpoint))
    p_config.device = torch.cuda.current_device()
else:
    if args.checkpoint != None:
        model_rigid.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    config.device = torch.device('cpu')
model_rigid.eval()

fmr_threshold = 0.05
rmse_threshold = 0.2

transformations = []
displ, displ_ngenet = [], []

dist_thresh_maps = {
    '10000': config.first_subsampling_dl,
    '5000': config.first_subsampling_dl,
    '2500': config.first_subsampling_dl * 1.5,
    '1000': config.first_subsampling_dl * 1.5,
    '500': config.first_subsampling_dl * 1.5,
    '250': config.first_subsampling_dl * 2,
}
model_nonrigid = Registration(p_config)
timer = Timers()

# %%
src = []
tgt = []
for pair_ind, inputs in enumerate(tqdm(test_dataloader)):
    if use_cuda:
        for k, v in inputs.items():
            if isinstance(v, list):
                for i in range(len(v)):
                    inputs[k][i] = inputs[k][i].cuda()
            else:
                inputs[k] = inputs[k].cuda()
    with torch.no_grad():
        
        batched_feats_h, batched_feats_m, batched_feats_l = model_rigid(inputs)
        stack_points = inputs['points']
        stack_points_raw = inputs['batched_points_raw']
        stack_lengths = inputs['stacked_lengths']
        coords_src = stack_points[0][:stack_lengths[0][0]]
        coords_tgt = stack_points[0][stack_lengths[0][0]:]
        coords_src_raw = stack_points_raw[:stack_lengths[0][0]]
        coords_tgt_raw = stack_points_raw[stack_lengths[0][0]:]
        feats_src_h = batched_feats_h[:stack_lengths[0][0]]
        feats_tgt_h = batched_feats_h[stack_lengths[0][0]:]
        feats_src_m = batched_feats_m[:stack_lengths[0][0]]
        feats_tgt_m = batched_feats_m[stack_lengths[0][0]:]
        feats_src_l = batched_feats_l[:stack_lengths[0][0]]
        feats_tgt_l = batched_feats_l[stack_lengths[0][0]:]

        source_npy = coords_src.detach().cpu().numpy()
        target_npy = coords_tgt.detach().cpu().numpy()

        source_feats_h = feats_src_h[:, :-2].detach().cpu().numpy()
        target_feats_h = feats_tgt_h[:, :-2].detach().cpu().numpy()
        source_feats_m = feats_src_m.detach().cpu().numpy()
        target_feats_m = feats_tgt_m.detach().cpu().numpy()
        source_feats_l = feats_src_l.detach().cpu().numpy()
        target_feats_l = feats_tgt_l.detach().cpu().numpy() 
        
        after_vote = vote(
            source_npy=source_npy, 
            target_npy=target_npy, 
            source_feats=[source_feats_h, source_feats_m, source_feats_l], 
            target_feats=[target_feats_h, target_feats_m, target_feats_l], 
            voxel_size=config.first_subsampling_dl,
            use_cuda=use_cuda)
        source_npy, target_npy, source_feats_npy, target_feats_npy = after_vote

        source, target = npy2pcd(source_npy), npy2pcd(target_npy)
        
        source_feats, target_feats = npy2feat(source_feats_h), npy2feat(target_feats_h)
        pred_T, estimate, result = execute_global_registration(
            source=source,
            target=target,
            source_feats=source_feats,
            target_feats=target_feats,
            voxel_size=dist_thresh_maps['10000']
        )
        
        transformations.append(pred_T)
    
    corrs_pred = np.unique(np.asarray(result.correspondence_set).T[0])

    raw_coords_src = torch.tensor(pcd2npy(estimate)).to(coords_src.device)
    raw_coords_tgt = torch.tensor(pcd2npy(target)).to(coords_tgt.device)

    coords_src_raw = coords_src_raw.cpu().detach().numpy()
    coords_tgt_raw = coords_tgt_raw.cpu().detach().numpy()
   
    model_nonrigid.load_pcds(raw_coords_src.float(), raw_coords_tgt.float(), inds=corrs_pred, search_radius=0.0375)
    warped_pcd, hist, iter_cnt, timer = model_nonrigid.register(visualize=args.vis, timer = timer)
    warped_pcd = warped_pcd.cpu().numpy()
    src.append(warped_pcd)
    tgt.append(pcd2npy(target))
    # # Create an Open3D PointCloud object
    # pcd = o3d.geometry.PointCloud()
    # # Set the point cloud data from the NumPy array
    # pcd.points = o3d.utility.Vector3dVector(warped_pcd)
    # # Save the PointCloud object as a .pcd file
    # o3d.io.write_point_cloud(f"test_output/point_cloud{pair_ind}.pcd", pcd)
data = {'src':src,'tgt':tgt}
import pickle
# Save the dictionary as a pickle file
with open(f'test_output.pickle', 'wb') as f:
    pickle.dump(data, f)