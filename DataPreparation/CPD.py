import open3d as o3d
from probreg import cpd
from probreg import callbacks
import time
import numpy as np
import pickle
import argparse
from Augment import *

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', type=bool, help= 'Use CUDA', default=True)
#parser.add_argument('--type', type=str, help= 'Rigid, Affine or Non-Rigid', default='Rigid')
#parser.add_argument('--crop_type', type=str, help= 'Dataset to use.', default='original', choices=['cropped','original'])
args = parser.parse_args()

#assert args.type in ['Rigid','Non-Rigid', 'Affine']

if args.use_cuda:
    import cupy as cp
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
else:
    cp = np
    to_cpu = lambda x: x

for c in ['original']:
    with open(f"RANSACData/RANSACTrain{c}.pickle", 'rb') as f:
        trn_data = pickle.load(f)
    for ty in ['Rigid','Non-Rigid', 'Affine']:
        sources = []
        targets = []
        tr = []
        for i in range(len(trn_data['source'])):
            source= o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(np.asarray(trn_data['source'][i], dtype=np.float32))
            source.transform(np.asarray(trn_data['transformation'][i]))
            source = cp.asarray(source.points, dtype=np.float32)

            target = cp.asarray(trn_data['target'][i], dtype=np.float32)

            if ty == 'Rigid':
                acpd = cpd.RigidCPD(source, use_cuda=args.use_cuda)
            elif ty == 'Non-Rigid':
                acpd = cpd.NonRigidCPD(source, use_cuda=args.use_cuda)
            else:
                acpd = cpd.AffineCPD(source, use_cuda=args.use_cuda)
            
            start = time.time()
            tf_param, _, _ = acpd.registration(target)
            elapsed = time.time() - start
            print("time: ", elapsed, i)

            ts = tf_param.transform(source)

            # Disorient the point clouds.
            R = generate_random_rotation_matrix()
            t = generate_random_tranlation_vector()
            T = np.eye(4)
            T[:3, :3] = np.array(R)
            T[:3, 3] = np.array(t)
            
            ts = transform(ts.get(), R, t)
            T = np.linalg.inv(T)

            sources.append(ts)
            targets.append(target.get())
            tr.append(T)

        data = {'source':sources,'target':targets, 'transformation': tr}

        # Save the dictionary as a pickle file
        with open(f'CPDData/CPDTrain{ty}{c}.pickle', 'wb') as f:
            pickle.dump(data, f)