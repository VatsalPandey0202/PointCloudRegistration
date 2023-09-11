Non-rigid Point Cloud Registration for Middle Ear Diagnostics (C2P-Net)
================================================

In this project, we aim to improve the interpretation of OCT scans of middle ear. We propose C2P-Net: a two-staged non-rigid registration pipeline for complete to partial point clouds, which are sampled from ex-vivo and in-vivo OCT models of middle ear, respectively. To overcome the lack of labeled training data, a fast and effective generation pipeline in Blender3D is designed to simulate middle ear shapes and extract in-vivo noisy and partial point clouds.

![Transformation GIF](documentation/transformation.gif)

## Environment
- python=3.7 (because of open3d)
- Blender=3.1.2
- torch=1.13.1
- vtk=9.2.5
- **open3d=0.10.0.0**
  
See `requirements.txt` for more details.
<!-- pytorch= -->



## Middle Ear Shape Simulation
We simulate synthetic in-vivo middle ear shape variants as training dataset in Blender. Before performing simulation, a config file need to be adapted to a new environment:

```yaml
output:
  # folder where the simulate ear shapes will be saved.
  folder: /folder/path/to/trainset/
  # folder for samples used for testing the neural network
  folder_test: /folder/path/to/testset/

# path to the file of non-rigid and rigid deformation operations 
def_ops_path: /path/to/project/c2p-net/ear_simulation/Configs/deformation_ops/large_rigid.yml

blender:
  # path to blender executable file
  exe_path: /path/to/blender-3.1.2-linux-x64/blender
  # path to the middle ear scene
  scene_path: /path/to/project/c2p-net/ear_simulation/middle_ear_scene.blend
  # path to the simuation script
  python_path: /path/to/project/c2p-net/ear_simulation/Scripts/simulation.py

# how many variants will be generated
num: 100000
num_test: 500

```

Run simulation:
```bash
cd ear_simulation/Scripts/
python3 simulation_master.py ../Configs/setup_larger_deformation_with_rigid.yml
```

## C2P-Net
### Structure
![Structure of C2P-Net](./documentation/nn_structure.png)

### Inference on custom dataset
Prepare dataset for inference:
```bash
cd mesh_dataset/
python3 read_any_data.py /path/to/testFiles/*.stl path/to/ex-vivo/.stl
cd ..
```
The specification for the dataset path should be in glob format.  
Run inference:
```bash
python3 testScript.py --checkpoint /path/to/NgeNet/checkpoint
```
All other arguments, which can be seen with `-h`, are working by default. 

### Train NgeNet
```bash
python3 trainNgeNet.py /path/to/NgeNet/config
```
The config which was used in the paper is here: `config/eardataset.yaml`  
Every training and model parameter can be adjusted in these config files.

# TODO
README  
add a .GIF which shows the transformation process rendered in blender
