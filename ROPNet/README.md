# [Point Cloud Registration using Representative Overlapping Points (ROPNet)](https://arxiv.org/abs/2107.02583)

## Abstract

ROPNet uses discriminative features for registration that transforms partial-to-partial registration into partial-to-complete registration. A context-guided module is used which uses an encoder to extract global features for predicting point overlap score. To better find representative overlapping points, it uses the extracted global features for coarse alignment. A Transformer is used to enrich point features and remove non-representative points based on point overlap score and feature matching. A similarity matrix is built in a partial-to-complete mode, and finally, weighted SVD is adopted to estimate a transformation matrix. Though the medthod has been tested on a simple ModelNet40 dataset, I have modified the network architecture so that it can fit a complex data like TOF & PC MRI.

## Model Training

```
cd src/
python train.py --root your_data_path/modelnet40_ply_hdf5_2048/ --noise --unseen
```

## Model Evaluation

```
cd src/
python eval.py --root your_data_path/modelnet40_ply_hdf5_2048/  --unseen --noise  --cuda --checkpoint work_dirs/models/min_rot_error.pth
```

## Registration Visualization

```
cd src/
python vis.py --root your_data_path/modelnet40_ply_hdf5_2048/  --unseen --noise  --checkpoint work_dirs/models/min_rot_error.pth
```


## Citation

If you find our work is useful, please consider citing:

```
@article{zhu2021point,
  title={Point Cloud Registration using Representative Overlapping Points},
  author={Zhu, Lifa and Liu, Dongrui and Lin, Changwei and Yan, Rui and G{\'o}mez-Fern{\'a}ndez, Francisco and Yang, Ninghua and Feng, Ziyong},
  journal={arXiv preprint arXiv:2107.02583},
  year={2021}
}
```

and

```
@article{zhu2021deep,
  title={Deep Models with Fusion Strategies for MVP Point Cloud Registration},
  author={Zhu, Lifa and Lin, Changwei and Liu, Dongrui and Li, Xin and G{\'o}mez-Fern{\'a}ndez, Francisco},
  journal={arXiv preprint arXiv:2110.09129},
  year={2021}
}
```

## Acknowledgements

We thank the authors of [RPMNet](https://github.com/yewzijian/RPMNet), [PCRNet](https://github.com/vinits5/pcrnet_pytorch), [OverlapPredator](https://github.com/overlappredator/OverlapPredator), [PCT](https://github.com/MenghaoGuo/PCT) and [PointNet++](https://github.com/charlesq34/pointnet2) for open sourcing their methods.

We also thank the third-party code [PCReg.PyTorch](https://github.com/zhulf0804/PCReg.PyTorch) and [Pointnet2.PyTorch](https://github.com/zhulf0804/Pointnet2.PyTorch).
