# [Point Cloud Registration using Representative Overlapping Points (ROPNet)](https://arxiv.org/abs/2107.02583)

## Abstract

ROPNet uses discriminative features for registration that transforms partial-to-partial registration into partial-to-complete registration. A context-guided module is used which uses an encoder to extract global features for predicting point overlap score. To better find representative overlapping points, it uses the extracted global features for coarse alignment. A Transformer is used to enrich point features and remove non-representative points based on point overlap score and feature matching. A similarity matrix is built in a partial-to-complete mode, and finally, weighted SVD is adopted to estimate a transformation matrix. Though the medthod has been tested on a simple ModelNet40 dataset, I have modified the network architecture so that it can fit a complex data like TOF & PC MRI.

## Model Training

```
cd src/
python train.py
```
For the arguments check 'ROPNet/src/configs/arguments.py'.

## Model Evaluation

```
cd src/
python eval.py  --unseen --noise  --cuda --checkpoint work_dirs/models/min_rot_error.pth
```

## Registration Visualization

```
cd src/
python vis.py --unseen --noise  --checkpoint work_dirs/models/min_rot_error.pth
```
