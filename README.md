# PointCloudRegistration
This repository contains the code for my Master thesis: Deep Learning based Co-Registration of Point Clouds. The work contains three different methods that were used to register point clouds from TOF and PC MRI.

## Data Preparation
This folder contains the code to generate TOF-PC sample pairs from a single pair that will be used by the three methods. Further explanation is given in the folder itself.

## ROPNet
This is an end to end rigid registration method. Check the folder for the usage.

## DIP
This is a feature learning based rigid registration method. Check the folder for the usage.

## C2P Net
This method first uses GC-Net for rigid registration followed by NDP for non-rigid regsitration. Check the folder for the usage.

## References
[Point Cloud Registration using Representative Overlapping Points (ROPNet)](https://github.com/zhulf0804/ROPNet)
