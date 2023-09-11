import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import torch_nndistance as NND
from dataset import Dataset
from losses import hardest_contrastive
from network import PointNetFeature
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.loss import chamfer_distance

'''
Notes:
1. training assumes that data (patches, LRFs) are preprocessed with preprocess_3dmatch_lrf_train.py
(it would be too slow to process this info during training)
2. batch size, patch size, lrf kernel size are defined in the preprocessing step
'''

chkpt_dir = './chkpts'
if not os.path.isdir(chkpt_dir):
    os.mkdir(chkpt_dir)


do_data_augmentation = False # activate/deactivate data augmentation
l2norm = True # activate/deactivate LRN
tnet = True # activate/deactivate TNet

dataset_to_train = None # [0, 1]
nepochs = 200
dim = 32*2

model = PointNetFeature(dim=dim, l2norm=l2norm, tnet=tnet)
device_ids = [0] # change this according to your GPU setup, e.g. if you have only one GPU -> device_ids = [0]
net = nn.DataParallel(model, device_ids=device_ids).cuda()
net.train()

train_dataset = Dataset('train',False)
test_dataset = Dataset('train',False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

#optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001, nesterov=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=40,
                                                                     T_mult=2,
                                                                     eta_min=1e-6,
                                                                     last_epoch=-1)

# logger    
log_dir_root = './logs'
if not os.path.isdir(log_dir_root):
    os.mkdir(log_dir_root)

date = datetime.now().timetuple()
log_dir = os.path.join(log_dir_root, '{}.{:02d}.{:02d}.{:02d}.{:02d}.{:02d}'.format(date[0], date[1], date[2], date[3], date[4], date[5]))

writer = SummaryWriter(log_dir=log_dir)

dists_eval = np.empty((test_dataset.get_length(),))
min_val_loss = np.inf
for e in range(nepochs):
    for frag1_batch, frag2_batch, _, _, R1, R2, lrf1, lrf2 in train_dataloader:

        '''
        TRAINING
        '''
        frag1_batch = frag1_batch.squeeze().cuda()
        frag2_batch = frag2_batch.squeeze().cuda()
        R1 = R1.squeeze().cuda()
        R2 = R2.squeeze().cuda()
        lrf1 = lrf1.squeeze().cuda()
        lrf2 = lrf2.squeeze().cuda()

        optimizer.zero_grad()

        f1, xtrans1, trans1, f2, xtrans2, trans2 = net(frag1_batch, frag2_batch)
        
        # chamfer loss
        lchamf,_ = chamfer_distance(xtrans1.transpose(2, 1).contiguous(),xtrans2.transpose(2, 1).contiguous())
        
        # hardest-contrastive loss
        lcontrastive, a, b, c = hardest_contrastive(f1, f2)

        # combination of losses
        loss = lcontrastive + lchamf
        loss.backward()
        optimizer.step()

    '''
    VALIDATION
    '''
    net.eval()
    j = 0
    with torch.no_grad():
        for frag1_batch, frag2_batch, _, _, R1, R2, lrf1, lrf2 in test_dataloader:
            frag1_batch = frag1_batch.squeeze().cuda()
            frag2_batch = frag2_batch.squeeze().cuda()
            R1 = R1.squeeze().cuda()
            R2 = R2.squeeze().cuda()
            lrf1 = lrf1.squeeze().cuda()
            lrf2 = lrf2.squeeze().cuda()
            #print(frag1_batch.shape,frag2_batch.shape)
            f1, _, trans1, f2, _, trans2 = net(frag1_batch, frag2_batch)

            # hardest-contrastive loss
            lcontrastive, a, b, c = hardest_contrastive(f1, f2)
            # chamfer loss
            lchamf,_ = chamfer_distance(xtrans1.transpose(2, 1).contiguous(),xtrans2.transpose(2, 1).contiguous())
            # combination of losses
            dists_eval[j] = lcontrastive + lchamf
            j += 1
   

    net.train()
    print(f"Epoch:{e} Train loss:{loss.item()} Val loss:{np.mean(dists_eval)}")
    if e == nepochs-1:
        torch.save(net.state_dict(), '/workspace/Storage_redundent/PointCloudRegistration/dip/chkpts/final_dip.pt')
    if np.mean(dists_eval)<min_val_loss:
        min_val_loss = np.mean(dists_eval)
        torch.save(net.state_dict(), '/workspace/Storage_redundent/PointCloudRegistration/dip/chkpts/best_dip.pt')


      

    scheduler.step()