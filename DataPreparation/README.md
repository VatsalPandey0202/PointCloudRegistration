Execute run.py to perform data augmentation and generate sample pairs that are registered using RANSAC+ICP.

\```python
import os
import argparse
# ... (insert the rest of your code here)
gc.collect()
\```

``` python script_name.py --voxel_size 0.01 --tof_dist_thresh 20.0 --pc_dist_thresh 20.0 --confidence_thresh 0.80 --use_avg True --crop True --range 0 --N 20000 ```

