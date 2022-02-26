"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import argparse
import os

import src

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--dataset', type=str, default='nuscenes',
                        choices=['nuscenes', 'argoverse'],
                        help='dataset name (default: nuscenes)')
    parser.add_argument('--version', type=str, default='trainval',
                        choices=['mini', 'trainval'],
                        help='version name (default: mini)')
    parser.add_argument('--dataroot', type=str, default='/home/user/data/Dataset/nuscenes',
                        help='directory of dataset')
    parser.add_argument('--base-size', type=list, default=(900, 1600),
                        help='base image size (H x W)')
    parser.add_argument('--crop-size', type=list, default=(128, 352),
                        help='crop image size')
    parser.add_argument('--ncams', type=int, default=5,
                        help='number of cameras')
    parser.add_argument('--resize-lim', type=tuple, default=(0.193, 0.225))
    parser.add_argument('--bot-pct-lim', type=tuple, default=(0.0, 0.22))
    parser.add_argument('--rot-lim', type=tuple, default=(-5.4, 5.4))
    parser.add_argument('--rand-flip', action='store_true', default=True)
    parser.add_argument('--max-grad-norm', type=float, default=5.0)
    parser.add_argument('--pos-weight', type=float, default=2.13)
    parser.add_argument('--xbound', type=list, default=(-50.0, 50.0, 0.5))
    parser.add_argument('--ybound', type=list, default=(-50.0, 50.0, 0.5))
    parser.add_argument('--zbound', type=list, default=(-10.0, 10.0, 20.0))
    parser.add_argument('--dbound', type=list, default=(4.0, 45.0, 1.0))
    # training hyper params
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size per gpu for training (default: 8)')
    parser.add_argument('--nworkers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads per gpu')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--nepochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-7, metavar='M',
                        help='w-decay (default: 1e-7)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='shuffle the dataset')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--master_addr', type=str, default='')
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='./runs',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--modelf', type=str, default='./runs/model2000.pt',
                        help='model for visualization')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    args.distributed = True if "WORLD_SIZE" in os.environ else False
    args.ngpus = int(os.environ["WORLD_SIZE"] if "WORLD_SIZE" in os.environ else 1)
    args.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"] if args.ngpus > 1 else [0]
    args.gpu_ids = [d for d in args.gpu_ids if d != ',']
    args.gpu_ids = [int(d) for d in args.gpu_ids]

    args.ngpus = args.ngpus if args.ngpus <= len(args.gpu_ids) else len(args.gpu_ids)
    args.gpu_ids = args.gpu_ids[:args.ngpus]
    args.local_rank = int(os.environ["LOCAL_RANK"] if "LOCAL_RANK" in os.environ else 0)
    
    return args

if __name__ == '__main__':
    args = parse_args()
    print('ngpus: {} || gpu_id: {} || rank: {}'.format(
        args.ngpus, args.gpu_ids[args.local_rank], args.local_rank
    ))

    src.train.train(args)
    # src.explore.viz_model_preds(args, viz_train=False, gpuid=3)
    # src.explore.viz_vector_gt(args, viz_train=False)
    # src.explore.viz_pixel_gt(version='trainval', viz_train=True, gpuid=3)