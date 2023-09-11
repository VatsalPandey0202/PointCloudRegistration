import numpy as np
import glob
from scipy.spatial.distance import directed_hausdorff
import open3d as o3d
from scipy.spatial.distance import cdist
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors

# Set the random seed to 42
random.seed(42)

def hausdorffDistance(original_cloud,aug_cloud):
    """
    Computes the Hausdorff distance between two point clouds
    """
    # Load the original and augmented point clouds as numpy arrays
    original_pc = np.asarray(original_cloud.points)
    augmented_pc = np.asarray(aug_cloud.points)
    # Compute the Hausdorff distance between the original and augmented point clouds
    hd1 = directed_hausdorff(original_pc, augmented_pc)[0]
    hd2 = directed_hausdorff(augmented_pc, original_pc)[0]
    return max(hd1,hd2)

def chamfer_distance(p1, p2):
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(np.asarray(p1.points,dtype='float32').reshape(-1,3))
    y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(np.asarray(p2.points,dtype='float32').reshape(-1,3))
    min_x_to_y = y_nn.kneighbors(np.asarray(p1.points,dtype='float32').reshape(-1,3))[0]
    min_y_to_x = x_nn.kneighbors(np.asarray(p2.points,dtype='float32').reshape(-1,3))[0]
    chamfer_dist = np.mean(min_x_to_y) + np.mean(min_y_to_x)
    return chamfer_dist

def check_all(tof_path,pc_path, original_TOF, original_PC):
    #Read and check all TOF point clouds
    paths = []
    cd = []
    hd = []
    avg = []
    for f in glob.glob(tof_path):
        pcd = o3d.io.read_point_cloud(str(f))
        if pcd.is_empty(): 
            exit()
        else: 
            centroid = pcd.get_center()#np.mean(np.asarray(pcd.points), axis=0)
            #pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)-centroid)
            pcd.translate(-centroid)
            hds = hausdorffDistance(original_TOF,pcd)
            cds = chamfer_distance(original_TOF,pcd)
            cd.append(cds)
            hd.append(hds)
            paths.append(str(f))
            avg.append((hds+cds)/2)

    df = pd.DataFrame({'Path': paths, 'Chamfer': cd, 'Hausdorff': hd, 'Average': avg})
    df = df.sort_values(by='Average',ascending=False)
    df.to_excel('Xlsx/TOF.xlsx', index=False)
    tof_avg = sum(avg)/len(avg)
    #Read and check all PC point clouds
    paths = []
    cd = []
    hd = []
    avg = []
    for f in glob.glob(pc_path):
        pcd = o3d.io.read_point_cloud(str(f))
        if pcd.is_empty(): 
            exit()
        else: 
            centroid = pcd.get_center()
            pcd.translate(-centroid)
            hds = hausdorffDistance(original_PC,pcd)
            cds = chamfer_distance(original_PC,pcd)
            cd.append(cds)
            hd.append(hds)
            paths.append(str(f))
            avg.append((hds+cds)/2)

    df = pd.DataFrame({'Path': paths, 'Chamfer': cd, 'Hausdorff': hd, 'Average': avg})
    df = df.sort_values(by='Average',ascending=False)
    df.to_excel('Xlsx/PC.xlsx', index=False)
    pc_avg = sum(avg)/len(avg)
    return tof_avg, pc_avg





