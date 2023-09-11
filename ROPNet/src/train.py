import argparse
import json
import numpy as np
import open3d
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys

ROOT = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(ROOT)))

from data import RANSACCropped,RANSACOriginal,RigidCPDCropped,RigidCPDOriginal,NonRigidCPDCropped,NonRigidCPDOriginal,AffineCPDCropped,AffineCPDOriginal
from models import ROPNet
from loss import cal_loss
from metrics import compute_metrics, summary_metrics, print_train_info
from utils import time_calc, inv_R_t, batch_transform, setup_seed, square_dists
from configs import train_config_params as config_params


global_min_loss, global_min_rot = float('inf'),float('inf')


# empty the CUDA cache
torch.cuda.empty_cache()

def save_summary(writer, loss_all, cur_r_isotropic, cur_r_mse, global_step, tag,
                 lr=None):
    for k, v in loss_all.items():
        loss = np.mean(v.item())
        writer.add_scalar(f'{k}/{tag}', loss, global_step)
    cur_r_mse = np.mean(cur_r_mse)
    writer.add_scalar(f'RError/{tag}', cur_r_mse, global_step)
    cur_r_isotropic = np.mean(cur_r_isotropic)
    writer.add_scalar(f'rotError/{tag}', cur_r_isotropic, global_step)
    if lr is not None:
        writer.add_scalar('Lr', lr, global_step)


@time_calc
def train_one_epoch(train_loader, model, loss_fn, optimizer, epoch, log_freq, writer):
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    global test_min_loss, test_min_r_mse_error, test_min_rot_error
    for step, (tgt_cloud, src_cloud, gtR, gtt) in enumerate(tqdm(train_loader)):
        np.random.seed((epoch + 1) * (step + 1))
        tgt_cloud, src_cloud, gtR, gtt = tgt_cloud.cuda(), src_cloud.cuda(), \
                                         gtR.cuda(), gtt.cuda()

        optimizer.zero_grad()
        results = model(src=src_cloud,
                        tgt=tgt_cloud,
                        num_iter=2,
                        train=True)
        pred_Ts = results['pred_Ts']
        pred_src = results['pred_src']
        x_ol = results['x_ol']
        y_ol = results['y_ol']
        inv_R, inv_t = inv_R_t(gtR, gtt)
        gt_transformed_src = batch_transform(src_cloud[..., :3], inv_R,
                                             inv_t)
        dists = square_dists(gt_transformed_src, tgt_cloud[..., :3])
        loss_all = loss_fn(gt_transformed_src=gt_transformed_src,
                           pred_transformed_src=pred_src,
                           dists=dists,
                           x_ol=x_ol,
                           y_ol=y_ol)

        loss = loss_all['total']
        loss.backward()
        optimizer.step()

        R, t = pred_Ts[-1][:, :3, :3], pred_Ts[-1][:, :3, 3]
        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
        global_step = epoch * len(train_loader) + step + 1

        if global_step % log_freq == 0:
            save_summary(writer, loss_all, cur_r_isotropic, cur_r_mse,
                         global_step, tag='train',
                         lr=optimizer.param_groups[0]['lr'])

        losses.append(loss.item())
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic)
        t_isotropic.append(cur_t_isotropic)
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    # empty the CUDA cache
    #torch.cuda.empty_cache()
    return results


@time_calc
def test_one_epoch(test_loader, model, loss_fn, epoch, log_freq, writer, tag):
    model.eval()
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        for step, (tgt_cloud, src_cloud, gtR, gtt) in enumerate(
                tqdm(test_loader)):
            tgt_cloud, src_cloud, gtR, gtt = tgt_cloud.cuda(), src_cloud.cuda(), \
                                             gtR.cuda(), gtt.cuda()

            results = model(src=src_cloud,
                            tgt=tgt_cloud,
                            num_iter=2)
            pred_Ts = results['pred_Ts']
            pred_src = results['pred_src']
            x_ol = results['x_ol']
            y_ol = results['y_ol']
            inv_R, inv_t = inv_R_t(gtR, gtt)
            gt_transformed_src = batch_transform(src_cloud[..., :3], inv_R,
                                                 inv_t)
            dists = square_dists(gt_transformed_src, tgt_cloud[..., :3])
            loss_all = loss_fn(gt_transformed_src=gt_transformed_src,
                               pred_transformed_src=pred_src,
                               dists=dists,
                               x_ol=x_ol,
                               y_ol=y_ol)
            loss = loss_all['total']

            R, t = pred_Ts[-1][:, :3, :3], pred_Ts[-1][:, :3, 3]
            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
            global_step = epoch * len(test_loader) + step + 1
            if global_step % log_freq == 0:
                save_summary(writer, loss_all, cur_r_isotropic, cur_r_mse,
                             global_step, tag=tag)

            losses.append(loss.item())
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic)
            t_isotropic.append(cur_t_isotropic)
    model.train()
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    # empty the CUDA cache
    #torch.cuda.empty_cache()
    return results


def main():
    args = config_params()
    print(args)

    setup_seed(args.seed)
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)
        with open(os.path.join(args.saved_path, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, ensure_ascii=False, indent=2)
    summary_path = os.path.join(args.saved_path, 'summary')
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    checkpoints_path = os.path.join(args.saved_path, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    if args.data_type == 'RANSACOriginal':
        train_set = RANSACOriginal(split='train', npts=args.npts, ao=args.ao, normal=args.normal)
    elif args.data_type == 'RANSACCropped':
        train_set = RANSACCropped(split='train', npts=args.npts, ao=args.ao, normal=args.normal)
    elif args.data_type == 'RigidCPDOriginal':
        train_set = RigidCPDOriginal(split='train', npts=args.npts, ao=args.ao, normal=args.normal)
    elif args.data_type == 'RigidCPDCropped':
        train_set = RigidCPDCropped(split='train', npts=args.npts, ao=args.ao, normal=args.normal)
    elif args.data_type == 'NonRigidCPDOriginal':
        train_set = NonRigidCPDOriginal(split='train', npts=args.npts, ao=args.ao, normal=args.normal)
    elif args.data_type == 'NonRigidCPDCropped':
        train_set = NonRigidCPDCropped(split='train', npts=args.npts, ao=args.ao, normal=args.normal)
    elif args.data_type == 'AffineCPDOriginal':
        train_set = AffineCPDOriginal(split='train', npts=args.npts, ao=args.ao, normal=args.normal)
    elif args.data_type == 'AffineCPDCropped':
        train_set = AffineCPDCropped(split='train', npts=args.npts, ao=args.ao, normal=args.normal)
    
    
    

    splits=KFold(n_splits=5,shuffle=True,random_state=42)

    training_loss = {i: [] for i in range(5)}
    validation_loss = {i: [] for i in range(5)}

    global global_min_loss
    global global_min_rot

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_set)))):

        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(train_set, batch_size=args.batchsize,
                                shuffle=False, num_workers=args.num_workers, drop_last = True, sampler = train_sampler)
        val_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=False,
                                num_workers=args.num_workers, drop_last = True, sampler=test_sampler)
        
  
        model = ROPNet(args)
        model = model.cuda()
        loss_fn = cal_loss

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=0.001

        epoch = 0
        if args.resume:
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                        T_0=40,
                                                                        T_mult=2,
                                                                        eta_min=1e-6,
                                                                        last_epoch=-1)

        writer = SummaryWriter(summary_path)

        for i in tqdm(range(epoch)):
            for _ in train_loader:
                pass
            for _ in val_loader:
                pass
            scheduler.step()

        val_min_loss, val_min_r_mse_error, val_min_rot_error = float('inf'), float('inf'), float('inf')
        

        for epoch in range(epoch, args.epoches):
            print('=' * 20, epoch + 1, '=' * 20)
            train_results = train_one_epoch(train_loader=train_loader,
                                            model=model,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            epoch=epoch,
                                            log_freq=args.log_freq,
                                            writer=writer)
            print_train_info(train_results)
            training_loss[fold].append(train_results)

            val_results = test_one_epoch(test_loader=val_loader,
                                        model=model,
                                        loss_fn=loss_fn,
                                        epoch=epoch,
                                        log_freq=args.log_freq,
                                        writer=writer,tag='Val')
            print_train_info(val_results)
            validation_loss[fold].append(val_results)

            val_loss, val_r_error, val_rot_error = \
                val_results['loss'], val_results['r_mse'], \
                val_results['r_isotropic']
            
            
            if val_loss < val_min_loss:
                saved_path = os.path.join(checkpoints_path,
                                        f"min_loss_val_{fold+1}.pth")
                torch.save(model.state_dict(), saved_path)
                val_min_loss = val_loss

            if val_rot_error < val_min_rot_error:
                saved_path = os.path.join(checkpoints_path,
                                        f"min_val_rot_error_{fold+1}.pth")
                torch.save(model.state_dict(), saved_path)
                val_min_rot_error = val_rot_error

            if val_loss < global_min_loss:
                saved_path = os.path.join(checkpoints_path,
                                        f"min_loss.pth")
                torch.save(model.state_dict(), saved_path)
                global_min_loss = val_loss

            if val_rot_error < global_min_rot:
                saved_path = os.path.join(checkpoints_path,
                                        f"min_rot_error.pth")
                torch.save(model.state_dict(), saved_path)
                global_min_rot = val_rot_error

            scheduler.step()

    import pickle
    with open('../train_loss.pickle', 'wb') as f:
            pickle.dump(training_loss, f)
    with open('../val_loss.pickle', 'wb') as f:
            pickle.dump(validation_loss, f)


if __name__ == '__main__':
    main()
