import pathlib
from collections import defaultdict
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from common.args import Args
from common.utils import save_reconstructions
from data import transforms
from data.dmri_data import SliceData
from models.subsampling_model import Subsampling_Model
import scipy.io as sio

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = Subsampling_Model(
        out_directions=args.out_directions,
        dir_decimation_rate=args.dir_decimation_rate,
        direction_learning=args.direction_learning,
        initialization=args.initialization,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model

def reconstructe():
    args = create_arg_parser().parse_args(sys.argv[1:])
    args.checkpoint = f'summary/{args.test_name}/best_model.pt'

    model = load_model(args.checkpoint)
    theta = model.subsampling.theta.data.detach().cpu().numpy()
    phi = model.subsampling.phi.data.detach().cpu().numpy()
    sio.savemat(f'summary/{args.test_name}/bvecs.mat', {'phi': phi, 'theta': theta})


def create_arg_parser():
    parser = Args()
    parser.add_argument('--test-name', type=str, default='new0.2/3/random_0.0001', help='name for the output dir')
    parser.add_argument('--data-split', choices=['val', 'test'],default='val',
                        help='Which data partition to run on: "val" or "test"')
    parser.add_argument('--checkpoint', type=pathlib.Path,default='summary/test/checkpoint/best_model.pt',
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path,default='summary/test/rec',
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')

    return parser


if __name__ == '__main__':
    reconstructe()
