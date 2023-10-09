#### Data Preparation
This folder is a part of pipeline for data augmentation and point cloud registration. It performs various tasks related to processing point cloud data, augmenting it, and then running point cloud registration algorithms like RANSAC and ICP.

Here is an explanation of the code and how to run it with arguments:

1. **Importing Libraries**: The script starts by importing necessary libraries and modules, including `os`, `argparse` for command-line arguments, and several custom modules for data augmentation, quality checking, copying files, and point cloud registration.

2. **Argument Parsing**: The `argparse.ArgumentParser()` is used to define command-line arguments. These arguments control various aspects of the data processing and augmentation, such as voxel size, distance thresholds, cropping, and more.

3. **Deleting Previous Files**: It removes previously generated point cloud files from specific directories (`AugmentedData/*/*/*.pcd` and `AugmentedData/RejectedData/*.pcd`).

4. **Scaling Mesh**: It scales the input 3D meshes to a unit cube to ensure consistency in size.

5. **Loading Mesh**: It loads the original 3D meshes from files ("TOF_ww25_Cow_higherRes.obj" and "PCMRI_ww25_Cow_v4_final.obj").

6. **Sampling Points**: The script samples points from the meshes to create point clouds. These point clouds are further downsampled using voxel downsampling.

7. **Point Cloud Registration**: It performs point cloud registration using RANSAC and ICP to align the two point clouds ("pc_pcd" and "tof_pcd").

8. **Data Augmentation**: The script performs various data augmentation operations, including rotation, translation, and jittering, to create augmented point cloud data for training.

9. **Quality Checking**: It checks the quality of the augmented data using distance measures and possibly rejects or copies files based on user-defined thresholds or average values.

10. **Training PointNet**: It train a neural network (likely a PointNet) using the augmented data.

11. **Moving Files**: Some files are moved from the test set to the training set to increase the size of the training data.

12. **RANSAC+ICP**: It performs RANSAC+ICP registration again, possibly after augmenting the data.



#### Setup

# Define the Conda environment name
```shell
conda_env_name= myenv
```
# Create a Conda environment with Python 3.8 (you can change the version if needed)
```shell
conda create -n $conda_env_name python=3.8
```
# Activate the Conda environment
```shell
conda activate $conda_env_name
```

# Install packages from requirements.txt using pip
```shell
pip install -r requirements.txt
```
echo "Conda environment $conda_env_name created and activated."






#### Run
Execute run.py to perform data augmentation and generate sample pairs that are registered using RANSAC+ICP.
```shell
python run.py --voxel_size 0.01 --tof_dist_thresh 20.0 --pc_dist_thresh 20.0 --confidence_thresh 0.80 --use_avg True --crop True --range 0 --N 20000
```
Here's an explanation of each argument:

1. `--voxel_size` (Type: `float`, Default: `0.01`):
   - This argument specifies the size of voxels for downsampling the point clouds. It controls the granularity of the point cloud data.

2. `--tof_dist_thresh` (Type: `float`, Default: `20.0`):
   - This argument sets the distance threshold for TOF (Time-of-Flight) data. It is used for quality checking and removing sample not similar to original data.

3. `--pc_dist_thresh` (Type: `float`, Default: `20.0`):
   - Similar to `--tof_dist_thresh`, this argument sets the distance threshold for the PC (Point Cloud) data. It serves the same purpose but for a different point cloud.

4. `--confidence_thresh` (Type: `float`, Default: `0.80`):
   - This argument defines a confidence threshold for accepting test data. The script may use this threshold for quality checking or filtering out data points with confidence scores below this value.

5. `--use_avg` (Type: `bool`, Default: `True`):
   - A boolean argument that controls whether to use the average as a threshold. If set to `True`, it suggests using an average value for some operation; otherwise, it may use other threshold values.

6. `--crop` (Type: `bool`, Default: `True`):
   - Another boolean argument that determines whether to crop the TOF data before augmentation. If set to `True`, the script will crop the data based on specified range values.

7. `--range` (Type: `float`, Default: `0`):
   - This argument specifies the range for cropping the TOF data. It is effective when `--crop` is set to `True`. It controls the extent of the cropping operation.

8. `--N` (Type: `int`, Default: `20000`):
   - An integer argument that defines the number of points to sample from a mesh. It determines how many points are randomly selected from the 3D mesh to create a point cloud.

