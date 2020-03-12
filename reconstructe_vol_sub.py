import pathlib
from collections import defaultdict
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="6"
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from common.args import Args
from common.utils import save_reconstructions
from data import transforms
from data.dmri_data import SliceData
from models.subsampling_model import Subsampling_Model
import h5py
import scipy.io as sio

class DataTransform:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, img, fname, slice):
        image = transforms.to_tensor(img)
        image = transforms.center_crop(image.permute(2,0,1), (self.resolution, self.resolution))
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        return image, mean, std, fname

def create_data_loaders(args,i):
    data = SliceData(
        root=args.data_path / f'vol{i}',
        transform=DataTransform(args.resolution),
        sample_rate=1.
    )
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
    )
    return data_loader

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

def eval(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (input, mean, std, fnames) in data_loader:
            input = input.to(args.device)
            recons = model.subsampling(input)
            recons=recons.to('cpu').squeeze(1)
            slice_n = recons.shape[1]
            for i in range(recons.shape[0]):
                recons[i] = recons[i] * std[i] + mean[i]
                reconstructions[fnames[i]].append(recons[i].numpy())

    reconstructions = {
        fname: np.stack([pred for pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions,slice_n

def save_vol(reconstructions, out_dir, slice_n):
    pathlib.Path(out_dir).mkdir(exist_ok=True)
    vol_data=np.zeros((144, 144, len(reconstructions), slice_n))
    for fname, recons in reconstructions.items():
        slice = fname[8:fname.find('.')]
        vol_data[:,:,int(slice),:]=recons.squeeze().transpose(1,2,0)
    name=fname[:6]
    with h5py.File(out_dir +'/'+ name+'_sub.h5', 'w') as f:
        f.create_dataset('reconstruction', data=vol_data)

def reconstructe():
    args = create_arg_parser().parse_args(sys.argv[1:])
    args.checkpoint = f'summary/{args.test_name}/best_model.pt'
    args.out_dir = f'summary/{args.test_name}/rec_vol'
    model = load_model(args.checkpoint)

    for i in range(1, args.num_vol + 1):
        data_loader = create_data_loaders(args,i)
        reconstructions, slice_n = eval(args, model, data_loader)
        save_vol(reconstructions, args.out_dir, slice_n)
    theta = model.subsampling.theta.data.detach().cpu().numpy()
    phi = model.subsampling.phi.data.detach().cpu().numpy()
    sio.savemat(f'summary/{args.test_name}/bvecs.mat', {'phi': phi, 'theta': theta})


def create_arg_parser():
    parser = Args()
    parser.add_argument('--test-name', type=str, default='new0.2/3/random_0.0001', help='name for the output dir')
    parser.add_argument('--checkpoint', type=pathlib.Path,default='summary/test/checkpoint/best_model.pt',
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path,default='summary/test/rec',
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--num_vol', type=int, default=2, help='num of volumes to reconstruct')

    return parser


if __name__ == '__main__':
    reconstructe()
