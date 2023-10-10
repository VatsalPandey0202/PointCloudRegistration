# Distinctive 3D local deep descriptors
Distinctive 3D local deep descriptors (DIPs) are rotation-invariant compact 3D descriptors computed using a PointNet-based deep neural network.
DIPs can be used to register point clouds without requiring an initial alignment. DIPs are generated from point-cloud patches that are canonicalised with respect to their estimated local reference frame (LRF).


## Installation
```
git clone https://github.com/VatsalPandey0202/PointCloudRegistration.git
cd dip
pip install -r requirements.txt
pip install torch-cluster==1.4.5 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
cd torch-nndistance
python build.py install
```

## Preprocessing

Preprocessing can be used to generate patches and LRFs for training. 
This will greatly reduce training time.
Preprocessing requires two steps: 
the first step computes point correspondences between point-cloud pairs using the [Iterative Closest Point algoritm](http://www.open3d.org/docs/0.8.0/python_api/open3d.registration.registration_icp.html); 
the second step produces patches along with their LRF.
To preprocess training data, run *preprocess_correspondences.py* and *preprocess_lrf.py*.
Just make sure that datasets paths in the code is set.

## Training

Training requires preprocessed data, i.e. patches and LRFs (it would be too slow to extract and compute them at each iteration during training).
To train set the variable *dataset_root* in *train.py*.
Then run
```
python train.py
```
Training generates checkpoints in the *chkpts* directory and the training logs in the *logs* directory. Logs can be monitored through tensorboard by running
```
tensorboard --logdir=logs
```

## Demo using pretrained model

Run
```
python demo.py

```
