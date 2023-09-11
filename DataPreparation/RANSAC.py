import glob
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import pickle
import gc
import random
import copy
random.seed(42)


def preprocess_point_cloud(pcd, voxel_size):

    radius_normal = voxel_size * 4 
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 7 
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd,pcd_fpfh

def prepare_dataset(voxel_size,path1,path2):

    print(":: Load two point clouds.")
    source = o3d.io.read_point_cloud(path1)
    target = o3d.io.read_point_cloud(path2)

    source,source_fpfh = preprocess_point_cloud(source, voxel_size)
    target,target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source, target, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 4.0 # 1.5

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

def refine_registration(source, target, result_ransac, distance_threshold):
    
    print(":: Point-to-point ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

def ransac(crop=False):
    tof_train_path = "AugmentedData/TOF/Train/*.pcd"
 
    pc_train_path = "AugmentedData/PC/Train/*.pcd"


    tof_train_files = []
    pc_train_files = []
 

    for f in glob.glob(tof_train_path):
        tof_train_files.append(str(f))

    for f in glob.glob(pc_train_path):
        pc_train_files.append(str(f))

    np.random.shuffle(np.asarray(tof_train_files))
    np.random.shuffle(np.asarray(pc_train_files))
 

    if len(tof_train_files)>len(pc_train_files):
        tof_train_files = random.sample(tof_train_files,len(pc_train_files))
    elif len(tof_train_files)<len(pc_train_files):
        pc_train_files = random.sample(pc_train_files,len(tof_train_files))

    print("TOF & PC Train size",len(tof_train_files),len(pc_train_files))

    
    if crop:
        distance_threshold = 0.02
        x = 'cropped'
    else:
        distance_threshold = 0.02 
        x = 'original'

    sources = []
    targets = []
    inlier_ratio = []
    inlier_rmse = []
    transformation = []
    correspondence = []
    src_normals = []
    tgt_normals = []
    for i in range(len(tof_train_files)):
        source, target, source_fpfh, target_fpfh = prepare_dataset(0.01,pc_train_files[i],tof_train_files[i])
        result_ransac = execute_global_registration(source, target, source_fpfh, target_fpfh, 0.01)
        result_icp = refine_registration(source, target, result_ransac, distance_threshold)
        if len(result_icp.correspondence_set)<1000:
            continue
        sources.append(np.array(source.points))
        targets.append(np.array(target.points))
        src_normals.append(np.array(source.normals))
        tgt_normals.append(np.array(target.normals))
        inlier_ratio.append(len(result_icp.correspondence_set)/len(target.points))
        inlier_rmse.append(float(result_icp.inlier_rmse))
        transformation.append(np.array(result_icp.transformation))
        correspondence.append(np.array(result_icp.correspondence_set))

        del result_ransac
        del result_icp
        gc.collect()

    data = {'source':sources,'target':targets,'src_normals':src_normals,'tgt_normals':tgt_normals,'transformation':transformation,'inlier_rmse':inlier_rmse,'inlier_ratio':inlier_ratio, 'correspondence':correspondence}

    # Save the dictionary as a pickle file
    with open(f'RANSACData/RANSACTrain{x}.pickle', 'wb') as f:
        pickle.dump(data, f)

    del sources,targets,inlier_ratio,transformation,correspondence,src_normals,tgt_normals,data
    gc.collect()

    
    # sources = []
    # targets = []
    # inlier_ratio = []
    # inlier_rmse = []
    # transformation = []
    # correspondence = []
    # src_normals = []
    # tgt_normals = []

    
    # for i in range(len(tof_test_files)):
    #     source, target, source_fpfh, target_fpfh = prepare_dataset(voxel_size,pc_test_files[i],tof_test_files[i],unit_cube)
    #     result_ransac = execute_global_registration(source, target,source_fpfh, target_fpfh, voxel_size)
    #     result_icp = refine_registration(source, target, voxel_size, result_ransac, unit_cube)
    #     if len(result_icp.correspondence_set)<1000:
    #         continue
    #     sources.append(np.array(source.points))
    #     targets.append(np.array(target.points))
    #     src_normals.append(np.array(source.normals))
    #     tgt_normals.append(np.array(target.normals))
    #     inlier_ratio.append(len(result_icp.correspondence_set)/len(target.points))
    #     inlier_rmse.append(float(result_icp.inlier_rmse))
    #     transformation.append(np.array(result_icp.transformation))
    #     correspondence.append(np.array(result_icp.correspondence_set))
    
    #     gc.collect()

    # data = {'source':sources,'target':targets,'src_normals':src_normals,'tgt_normals':tgt_normals,'transformation':transformation,'inlier_rmse':inlier_rmse,'inlier_ratio':inlier_ratio,'correspondence':correspondence}

    # # Save the dictionary as a pickle file
    # with open(f'RANSACData/RANSACTest{x}.pickle', 'wb') as f:
    #     pickle.dump(data, f)

    # del sources,targets,inlier_ratio,transformation,correspondence,src_normals,tgt_normals,data
    # gc.collect()
    # return 0


