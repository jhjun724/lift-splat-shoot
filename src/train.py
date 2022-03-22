"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
import datetime
from tensorboardX import SummaryWriter
import numpy as np
import os

from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, MultiLoss, get_batch_iou, get_val_info
from .lr_scheduler import WarmupPolyLR


def train(args):
    grid_conf = {
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
    }
    data_aug_conf = {
        'resize_lim': args.resize_lim,
        'final_dim': args.final_dim,
        'rot_lim': args.rot_lim,
        'H': args.base_size[0], 'W': args.base_size[1],
        'rand_flip': args.rand_flip,
        'bot_pct_lim': args.bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': args.ncams,
    }

    model = compile_model(grid_conf, data_aug_conf, outC=4)

    if not torch.cuda.is_available() | args.no_cuda:
        model.cpu()
        device = torch.device('cpu')
    elif args.distributed:  # multi-GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda')
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            # output_device=args.local_rank,
            # find_unused_parameters=True
        )
    else:                   # single-GPU
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda')
        model.to(device)

    # device = torch.device('cuda')
    # model = torch.nn.DataParallel(model, device_ids=[0,1,2]).cuda()
    # args.batch_size *= torch.cuda.device_count()
    # args.lr *= torch.cuda.device_count()
    # args.local_rank = None

    trainloader, valloader = compile_data(args,
                                        data_aug_conf=data_aug_conf,
                                        grid_conf=grid_conf,
                                        parser_name='segmentationdata')

    iters_per_epoch = len(trainloader.dataset) // (args.batch_size * args.ngpus)
    max_iters = args.nepochs * iters_per_epoch
    if args.local_rank==0:
        print('==================================================')
        print('iterations per epoch:', iters_per_epoch)
        print('total iterations:', max_iters)
        print('==================================================')

    opt = torch.optim.Adam(model.parameters(),
                        lr=args.lr*args.ngpus,
                        weight_decay=args.weight_decay)
    lr_scheduler = WarmupPolyLR(opt,
                                max_iters=max_iters,
                                power=0.9,
                                warmup_factor=args.warmup_factor,
                                warmup_iters=args.warmup_iters,
                                warmup_method=args.warmup_method)
    loss_fn = SimpleLoss(args.pos_weight).to(device)
    # loss_fn = MultiLoss().to(device)
    writer = SummaryWriter(logdir=args.log_dir)
    val_step = 1000 if args.version == 'mini' else 10000

    model.train()

    iter = 0
    val_info = {'loss':0.0, 'iou':0.0}
    best_iou = 0.0
    start_time = time()
    print('Start training. Total epochs: {:d}, Total iterations: {:d}, Start time: {}'
        .format(args.nepochs, max_iters, datetime.datetime.now().strftime("%H:%M:%S")))

    for epoch in range(args.nepochs):
        np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            iter = iter + 1
            t0 = time()
            opt.zero_grad()

            if torch.cuda.is_available():
                imgs = imgs.to(device)
                rots = rots.to(device)
                trans = trans.to(device)
                intrins = intrins.to(device)
                post_rots = post_rots.to(device)
                post_trans = post_trans.to(device)
                binimgs = binimgs.to(device)

            preds = model(imgs, rots, trans, intrins, post_rots, post_trans,)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            lr_scheduler.step()
            t1 = time()
            if iter % 10 == 0:
                print('Epochs: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Elapsed Time: {} || Estimated Time: {}'.format(
                    epoch, args.nepochs, iter, max_iters, opt.param_groups[0]['lr'], loss, 
                    str(datetime.timedelta(seconds=int(t1 - start_time))),
                    str(datetime.timedelta(seconds=int((t1 - start_time)/iter*(max_iters-iter))))
                ))
                writer.add_scalar('train/loss', loss, iter)

            if iter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, iter)
                writer.add_scalar('train/epoch', epoch, iter)
                writer.add_scalar('train/step_time', t1 - t0, iter)

        if (epoch+1) % 10 == 0:
            val_info = get_val_info(model, valloader, loss_fn, device)
            print('VAL', val_info)
            writer.add_scalar('val/loss', val_info['loss'], iter)
            writer.add_scalar('val/iou', val_info['iou'], iter)

        if (epoch+1) % 20 == 0 or (epoch+1) == args.nepochs:
            model.eval()
            if val_info['iou'] > best_iou:
                best_iou = val_info['iou']
                best_name = os.path.join(args.log_dir, "best_model_test.pt")
                print('saving', best_name)
                torch.save(model.state_dict(), best_name)
            mname = os.path.join(args.log_dir, "model_{}epochs_test.pt".format(epoch+1))
            print('saving', mname)
            torch.save(model.state_dict(), mname)
            model.train()

    print('End training. End time: {}'.format(datetime.datetime.now().strftime("%H:%M:%S")))
