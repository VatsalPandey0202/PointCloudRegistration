import numpy as np
import torch
import open3d as o3d
from network import PointNetFeature
import copy
from lrf import lrf
import pickle

model_root = './chkpts'

patch_size = 256
dim = 32*2
perc = 5
batch_size = 500
pts_to_sample = 2048

# dataset specific parameters
voxel_size = 1.0
lrf_kernel = 3.0 * np.sqrt(3)

net = PointNetFeature(dim=dim, l2norm=True, tnet=True)

with open("../DataPreparation/RANSACData/RANSACTestoriginal.pickle", 'rb') as f:
    tst_data = pickle.load(f)

# Loading Model
state_dict = torch.load('chkpts/best_dip.pt')
new_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        k = k.replace('module.', '')
    new_state_dict[k] = v
net.load_state_dict(new_state_dict)
net.cuda()
net.eval()

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

print('*****************************')
print('demo MRI dataset')
print('*****************************')

print('point clouds in their original reference frame ...')
print('... press \'q\' to continue ...')
src = []
tgt = []
tsrc = []
for i in range(len(tst_data['source'])):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(np.asarray(tst_data['source'][i]))
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(np.asarray(tst_data['target'][i]))

    pcd1.paint_uniform_color([128 / 255, 128 / 255, 128 / 255])
    pcd2.paint_uniform_color([0 / 255, 76 / 255, 153 / 255])

    pcd1 = pcd1.voxel_down_sample(voxel_size)
    pcd2 = pcd2.voxel_down_sample(voxel_size)

    pcd1.estimate_normals()
    pcd2.estimate_normals()

    # random sampling of points
    inds1 = np.random.choice(np.asarray(pcd1.points).shape[0], pts_to_sample, replace=False)
    print()
    print('randomly sampled {} out of {} from pcd1 to compute DIPs'.format(pts_to_sample, np.asarray(pcd1.points).shape[0]))
    inds2 = np.random.choice(np.asarray(pcd2.points).shape[0], pts_to_sample, replace=False)
    print('randomly sampled {} out of {} from pcd2 to compute DIPs'.format(pts_to_sample, np.asarray(pcd2.points).shape[0]))

    pcd1_pts = np.asarray(pcd1.points)[inds1]
    pcd2_pts = np.asarray(pcd2.points)[inds2]

    # compute DIPs

    # set viz=True to visualise patch kernels and their respective LRF
    frag1_lrf = lrf(pcd=pcd1,
                    pcd_tree=o3d.geometry.KDTreeFlann(pcd1),
                    patch_size=patch_size,
                    lrf_kernel=lrf_kernel,
                    viz=False)

    frag2_lrf = lrf(pcd=pcd2,
                    pcd_tree=o3d.geometry.KDTreeFlann(pcd2),
                    patch_size=patch_size,
                    lrf_kernel=lrf_kernel,
                    viz=False)

    patches1 = np.empty((pcd1_pts.shape[0], 3, patch_size))
    patches2 = np.empty((pcd2_pts.shape[0], 3, patch_size))

    print()
    print('computing patches and LRFs ...')
    for i in range(pcd1_pts.shape[0]):
        frag1_lrf_pts, _, _ = frag1_lrf.get(pcd1_pts[i])
        frag2_lrf_pts, _, _ = frag2_lrf.get(pcd2_pts[i])

        patches1[i] = frag1_lrf_pts.T
        patches2[i] = frag2_lrf_pts.T
    print('... done')

    pcd1_desc = np.empty((patches1.shape[0], dim))
    pcd2_desc = np.empty((patches2.shape[0], dim))

    pcd1_mx = np.empty((patches1.shape[0], 256))
    pcd2_mx = np.empty((patches2.shape[0], 256))

    pcd1_amx = np.empty((patches1.shape[0], 256), dtype=int)
    pcd2_amx = np.empty((patches2.shape[0], 256), dtype=int)

    print()
    print('processing patches with PointNet ...')
    for b in range(int(np.ceil(patches1.shape[0] / batch_size))):

        i_start = b * batch_size
        i_end = (b + 1) * batch_size
        if i_end > pts_to_sample:
            i_end = pts_to_sample

        pcd1_batch = torch.Tensor(patches1[i_start:i_end]).cuda()
        with torch.no_grad():
            f, mx1, amx1 = net(pcd1_batch)
        pcd1_desc[i_start:i_end] = f.cpu().detach().numpy()[:i_end-i_start]
        pcd1_mx[i_start:i_end] = mx1.cpu().detach().numpy().squeeze()[:i_end - i_start]
        pcd1_amx[i_start:i_end] = amx1.cpu().detach().numpy().squeeze()[:i_end - i_start]

        pcd2_batch = torch.Tensor(patches2[i_start:i_end]).cuda()
        with torch.no_grad():
            f, mx2, amx2 = net(pcd2_batch)
        pcd2_desc[i_start:i_end] = f.cpu().detach().numpy()[:i_end-i_start]
        pcd2_mx[i_start:i_end] = mx2.cpu().detach().numpy().squeeze()[:i_end - i_start]
        pcd2_amx[i_start:i_end] = amx2.cpu().detach().numpy().squeeze()[:i_end - i_start]

    mag_pcd1_mx = np.linalg.norm(pcd1_mx, axis=1)
    mag_pcd2_mx = np.linalg.norm(pcd2_mx, axis=1)

    good_pcd1_desc = mag_pcd1_mx > np.percentile(mag_pcd1_mx, perc)
    good_pcd2_desc = mag_pcd2_mx > np.percentile(mag_pcd2_mx, perc)

    pcd1_desc = pcd1_desc[good_pcd1_desc]
    pcd2_desc = pcd2_desc[good_pcd2_desc]

    print('... done')

    # ransac
    pcd1_dsdv = o3d.pipelines.registration.Feature()
    pcd2_dsdv = o3d.pipelines.registration.Feature()

    pcd1_dsdv.data = pcd1_desc.T
    pcd2_dsdv.data = pcd2_desc.T

    pcd1_pts = pcd1_pts[good_pcd1_desc]
    pcd2_pts = pcd2_pts[good_pcd2_desc]

    _pcd1 = o3d.geometry.PointCloud()
    _pcd1.points = o3d.utility.Vector3dVector(pcd1_pts)
    _pcd2 = o3d.geometry.PointCloud()
    _pcd2.points = o3d.utility.Vector3dVector(pcd2_pts)

    print()
    print('estimating transformation with RANSAC ...')

    est_result12 =  execute_global_registration(_pcd1, _pcd2, pcd1_dsdv, pcd2_dsdv, voxel_size)
    print('... done')

    pcd1a = copy.deepcopy(pcd1)
    pcd2a = copy.deepcopy(pcd2)

    print()
    print('estimated transformation to align pcd1 to pcd2:')
    print(est_result12.transformation)

    pcd1a.transform(est_result12.transformation)

    print()
    print('point clouds in a common reference frame after estimated transformation ...')
    print('... press \'q\' to continue ...')

    src.append(np.array(pcd1.points))
    tgt.append(np.array(pcd2.points))
    tsrc.append(np.array(pcd1a.points))

data = {'source':src,'target':tgt,'tsource':tsrc}

# Save the dictionary as a pickle file
with open('DIPResults.pickle', 'wb') as f:
    pickle.dump(data, f)