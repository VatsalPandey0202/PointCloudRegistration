Non-rigid Point Cloud Registration for TOF-PC MRI (C2P-Net)
================================================

This code performs the registration of TOF-PC MRI using C2P-Net: a two-staged non-rigid registration pipeline for complete to partial point clouds.

## Environment
- python=3.7 (because of open3d)
- torch=1.13.1
- vtk=9.2.5
- **open3d=0.10.0.0**
  
See `requirements.txt` for more details.
<!-- pytorch= -->


### Train NgeNet
```bash
python3 trainNgeNet.py /path/to/NgeNet/config
```


