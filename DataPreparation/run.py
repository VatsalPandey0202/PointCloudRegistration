import os
import argparse
from Augment import *
from QualityCheck import *
from Copy import *
from Pointnet import *
from RANSAC import ransac, preprocess_point_cloud, execute_global_registration
import glob
import gc
import random
import copy as cp
import time

# Set the random seed to 42
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--voxel_size', type=float, help= 'The size of voxels for downsampling', default=0.01)
parser.add_argument('--tof_dist_thresh', type=float, help= 'Distance threshold for TOF data', default=20.0)
parser.add_argument('--pc_dist_thresh', type=float, help= 'Distance threshold for TOF data', default=20.0)
parser.add_argument('--confidence_thresh', type=float, help= 'Confidence threshold to accept test data', default=0.80)
parser.add_argument('--use_avg', type=bool, help= 'Use average as threshold', default=True)
parser.add_argument('--crop', type=bool, help= 'Crop TOF before augmentation', default=True)
parser.add_argument('--range', type=float, help= 'Crop range', default=0)
parser.add_argument('--N', type=int, help= 'Number of points to sample from mesh', default=20000)
args = parser.parse_args()

#Deleting previous TOF and PC files
data_path = "AugmentedData/*/*/*.pcd"
for f in glob.glob(data_path):
    try:
        os.remove(str(f))
    except:
        print("No file found")
        continue

data_path = "AugmentedData/RejectedData/*.pcd"
for f in glob.glob(data_path):
    try:
        os.remove(str(f))
    except:
        print("No file found")
        continue

def to_unit_cube(pcd):
    m = np.mean(pcd, axis=0, keepdims=True)  # [N, D] -> [1, D]
    v = pcd - m
    s = np.max(np.abs(v))
    v = v / s * 0.5
    return v

def scale_mesh(mesh, scale_fac = None):
    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    # Step 2: Compute center of bounding box
    center = (min_bound + max_bound) / 2

    if scale_fac:
        # Step 4: Translate mesh to origin
        mesh.translate(-center)

        # Step 5: Scale mesh to unit cube
        mesh.scale(scale_fac, center=(0,0,0))
    else:
        # Step 3: Compute scale factor
        scale = 2.0 / max(max_bound - min_bound)

        # Step 4: Translate mesh to origin
        mesh.translate(-center)

        # Step 5: Scale mesh to unit cube
        mesh.scale(scale, center=(0,0,0))

    return mesh

# Loading Mesh
tof_mesh = o3d.io.read_triangle_mesh("OriginalData/TOF_ww25_Cow_higherRes.obj")
pc_mesh = o3d.io.read_triangle_mesh("OriginalData/PCMRI_ww25_Cow_v4_final.obj")

#Scaling mesh to unit cube
tof_mesh = scale_mesh(tof_mesh)
pc_mesh = scale_mesh(pc_mesh)
pc_mesh = scale_mesh(pc_mesh, 1/1.7)#Scaling again to match TOF mesh. Experiment with different scaling factors.

# Converting to point cloud by sampling 50% points
tof_pcd = tof_mesh.sample_points_uniformly(int(len(tof_mesh.vertices)*0.5))
pc_pcd = pc_mesh.sample_points_uniformly(int(len(pc_mesh.vertices)*0.5))

tof_pcd = tof_pcd.voxel_down_sample(0.01)
pc_pcd = pc_pcd.voxel_down_sample(0.01)

pc_pcd,source_fpfh = preprocess_point_cloud(pc_pcd, 0.01)
tof_pcd,target_fpfh = preprocess_point_cloud(tof_pcd, 0.01)
result_ransac = execute_global_registration(pc_pcd, tof_pcd, source_fpfh, target_fpfh, 0.01)

pc_pcd = pc_pcd.transform(result_ransac.transformation)

if args.crop:
    xmin = min(np.array(pc_pcd.points)[:,0])-args.range
    xmax = max(np.array(pc_pcd.points)[:,0])+args.range
    ymin = min(np.array(pc_pcd.points)[:,1])-args.range
    ymax = max(np.array(pc_pcd.points)[:,1])+args.range
    zmin = min(np.array(pc_pcd.points)[:,2])-args.range
    zmax = max(np.array(pc_pcd.points)[:,2])+args.range
    # Define bounding box
    bbox_min = [xmin, ymin, zmin] 
    bbox_max = [xmax, ymax, zmax] 
    bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)

    # Crop point cloud
    tof_pcd = tof_pcd.crop(bbox)

    print(f'No of points in TOF after cropping {len(tof_pcd.points)}')

tof_pcd = tof_mesh.sample_points_poisson_disk(number_of_points=3000, pcl=tof_pcd)
pc_pcd = pc_mesh.sample_points_poisson_disk(number_of_points=3000, pcl=pc_pcd)


if args.crop:
    x = 'cropped'
else: x = 'original'


print(f'size of tof after voxel downsampling {len(tof_pcd.points)} and size of pc after voxel downsampling {len(pc_pcd.points)}')

min_point = min(len(pc_pcd.points),len(tof_pcd.points))
print("Minimum no of points: ",min_point)

# Save the sampled points as a point cloud file
o3d.io.write_point_cloud(f"OriginalData/TOF_voxel_{args.voxel_size}_points_{len(tof_pcd.points)}_{x}.pcd", tof_pcd)
o3d.io.write_point_cloud(f"OriginalData/PC_voxel_{args.voxel_size}_points_{len(pc_pcd.points)}_{x}.pcd", pc_pcd)

tof_path = "AugmentedData/TOF/Train/"
pc_path = "AugmentedData/PC/Train/"

print("Rotating Point Clouds...")
for i in range(200):
     R = generate_random_rotation_matrix()
     pc = transform(np.array(pc_pcd.points), R)
     tof = transform(np.array(tof_pcd.points), R)
     pcd = o3d.geometry.PointCloud()
     pcd.points = o3d.utility.Vector3dVector(pc)
     o3d.io.write_point_cloud(f'AugmentedData/PC/Train/PC_Rot_{i}.pcd',pcd)
     pcd.points = o3d.utility.Vector3dVector(tof)
     o3d.io.write_point_cloud(f'AugmentedData/TOF/Train/TOF_Rot_{i}.pcd',pcd)

print("Translating Rotated TOF Clouds...")
i = 0
for f in glob.glob(tof_path+'*.pcd'):
    t = generate_random_tranlation_vector()
    pcd = o3d.io.read_point_cloud(str(f))
    pcd.points = o3d.utility.Vector3dVector(transform(np.array(pcd.points), None, t))
    o3d.io.write_point_cloud(f'AugmentedData/TOF/Train/TOF_Trans_{i}.pcd',pcd)
    i+=1
print("Translating Rotated PC Clouds...")
i = 0
for f in glob.glob(pc_path+'*.pcd'):
    t = generate_random_tranlation_vector()
    pcd = o3d.io.read_point_cloud(str(f))
    pcd.points = o3d.utility.Vector3dVector(transform(np.array(pcd.points), None, t))
    o3d.io.write_point_cloud(f'AugmentedData/PC/Train/PC_Trans_{i}.pcd',pcd)
    i+=1

print("Jittering ALL TOF Clouds...")
i = 0
for f in glob.glob(tof_path+'*.pcd'):
    pcd = o3d.io.read_point_cloud(str(f))
    pcd.points = o3d.utility.Vector3dVector(jitter_point_cloud(np.array(pcd.points)))
    o3d.io.write_point_cloud(f'AugmentedData/TOF/Train/TOF_Jitter_{i}.pcd',pcd)
    i+=1
print("Jittering ALL PC Clouds...")
i = 0
for f in glob.glob(pc_path+'*.pcd'):
    pcd = o3d.io.read_point_cloud(str(f))
    pcd.points = o3d.utility.Vector3dVector(jitter_point_cloud(np.array(pcd.points)))
    o3d.io.write_point_cloud(f'AugmentedData/PC/Train/PC_Jitter_{i}.pcd',pcd)
    i+=1

#Transalating to origin
center = tof_pcd.get_center()
tof_pcd.translate(-center)
center = pc_pcd.get_center()
pc_pcd.translate(-center)



print("Check similarity using distance measure")
tof_avg,pc_avg = check_all(tof_path+"*.pcd",pc_path+"*.pcd",tof_pcd, pc_pcd)

print("TOF average distance",tof_avg)
print("PC average distance",pc_avg)

print("Copying files to test...")
if args.use_avg:
    print("Using average value")
    copy("Xlsx/TOF.xlsx","Xlsx/PC.xlsx", tof_avg, pc_avg) 
else:
    print("Using threshold value")
    copy("Xlsx/TOF.xlsx","Xlsx/PC.xlsx", args.tof_dist_thresh,args.pc_dist_thresh)


print("Counting training and testing samples...")
count = 0
for f in glob.glob("AugmentedData/TOF/Train/*.pcd"):
    count += 1
print(f"Training TOF sample:{count}")

count = 0
for f in glob.glob("AugmentedData/TOF/Test/*.pcd"):
    count += 1
print(f"Test TOF sample:{count}")
if count == 0:
    print("No TOF test samples")
    exit()

count = 0
for f in glob.glob("AugmentedData/PC/Train/*.pcd"):
    count += 1
print(f"Training PC sample:{count}")

count = 0
for f in glob.glob("AugmentedData/PC/Test/*.pcd"):
    count += 1
print(f"Test PC sample:{count}")
if count == 0:
    print("No PC test samples")
    exit()

print("Trainig Pointnet...")
train_test(min_point, 16, args.crop)

print("Rejecting less confident and incorrectly predicted samples...")
df = pd.read_excel("Xlsx/PointNet.xlsx")  
for index, row in df.iterrows():
    source = str(row['Path'])
    file_name = source.split('/')[-1]
    if row['Correct']=='No':
        destination = "AugmentedData/RejectedData/"+str(file_name)
        shutil.move(source, destination)
    elif float(row['Argmax'])<args.confidence_thresh:
        destination = "AugmentedData/RejectedData/"+str(file_name)
        shutil.move(source, destination)


print("Counting training and testing samples after pointnet...")
count = 0
for f in glob.glob("AugmentedData/TOF/Train/*.pcd"):
    count += 1
print(f"Training TOF sample:{count}")
count = 0
for f in glob.glob("AugmentedData/TOF/Test/*.pcd"):
    count += 1
print(f"Test TOF sample:{count}")
count = 0
for f in glob.glob("AugmentedData/PC/Train/*.pcd"):
    count += 1
print(f"Training PC sample:{count}")
count = 0
for f in glob.glob("AugmentedData/PC/Test/*.pcd"):
    count += 1
print(f"Test PC sample:{count}")

print("Moving TOF files from test to train to increase size of training data")
# Define source and destination folders
src_folder = "AugmentedData/TOF/Test"
dest_folder = "AugmentedData/TOF/Train"

# Get a list of all the files in the source folder
file_list = os.listdir(src_folder)
# Define the number of files to move
num_files_to_move = int(0.5*len(file_list))
# Randomly select files to move
files_to_move = random.sample(file_list, num_files_to_move)
# Move the files
for file_name in files_to_move:
    src_file_path = os.path.join(src_folder, file_name)
    dest_file_path = os.path.join(dest_folder, file_name)
    shutil.move(src_file_path, dest_file_path)

print("Moving PC files from test to train to increase size of training data")
# Define source and destination folders
src_folder = "AugmentedData/PC/Test"
dest_folder = "AugmentedData/PC/Train"

# Get a list of all the files in the source folder
file_list = os.listdir(src_folder)
# Define the number of files to move
num_files_to_move = int(1.0*len(file_list))
# Randomly select files to move
files_to_move = random.sample(file_list, num_files_to_move)
# Move the files
for file_name in files_to_move:
    src_file_path = os.path.join(src_folder, file_name)
    dest_file_path = os.path.join(dest_folder, file_name)
    shutil.move(src_file_path, dest_file_path)


print("Final count")
count = 0
for f in glob.glob("AugmentedData/TOF/Train/*.pcd"):
    count += 1
print(f"TOF sample:{count}")

count = 0
for f in glob.glob("AugmentedData/PC/Train/*.pcd"):
    count += 1
print(f"PC sample:{count}")

print("Perform RANSAC+ICP")
ransac(args.crop)
gc.collect()