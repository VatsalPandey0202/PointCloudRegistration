import shutil
import pandas as pd
import random

# Set the random seed to 42
random.seed(42)


def copy(tof_path,pc_path,tof_dist_thresh=None,pc_dist_thresh=None):
    tof_df = pd.read_excel(tof_path)  
    tof_dist_thresh = tof_dist_thresh
    for index, row in tof_df.iterrows():
        if float(row['Average']) < tof_dist_thresh:
            break
        else:
            source = str(row['Path'])
            file_name = source.split('/')[-1]
            if 'rot' in str(file_name) or 'trans' in str(file_name):
                continue
            else:
                destination = "AugmentedData/TOF/Test/"+str(file_name)
                shutil.move(source, destination)

    pc_df = pd.read_excel(pc_path)  
    pc_dist_thresh = pc_dist_thresh
    for index, row in pc_df.iterrows():
        if float(row['Average']) < pc_dist_thresh:
            break
        else:
            source = str(row['Path'])
            file_name = source.split('/')[-1]
            if 'rot' in str(file_name) or 'trans' in str(file_name):
                continue
            else:
                destination = "AugmentedData/PC/Test/"+str(file_name)
                shutil.move(source, destination)

            









'''
tof_test = "/workspace/Storage_redundent/PointCloudRegistration/Data/AugmentedData/TOF/Test/*.pcd"
tof_test_files = []
for f in glob.glob(tof_test):
    file = str(f).split("/")[-1]
    tof_test_files.append(file)

tof_monte = "/workspace/Storage_redundent/PointCloudRegistration/Data/MonteCarloSimulation/TOF/*.pcd"
tof_monte_files = []
for f in glob.glob(tof_monte):
    file = str(f).split("/")[-1]
    if file in tof_test_files:
        continue
    else: 
        dst = "/workspace/Storage_redundent/PointCloudRegistration/Data/AugmentedData/TOF/Train/"+file
        shutil.copy(str(f),dst)



pc_test = "/workspace/Storage_redundent/PointCloudRegistration/Data/AugmentedData/PC/Test/*.pcd"
pc_test_files = []
for f in glob.glob(pc_test):
    file = str(f).split("/")[-1]
    pc_test_files.append(file)

pc_monte = "/workspace/Storage_redundent/PointCloudRegistration/Data/MonteCarloSimulation/PC/*.pcd"
pc_monte_files = []
for f in glob.glob(pc_monte):
    file = str(f).split("/")[-1]
    if file in pc_test_files:
        continue
    else: 
        dst = "/workspace/Storage_redundent/PointCloudRegistration/Data/AugmentedData/PC/Train/"+file
        shutil.copy(str(f),dst)

'''
