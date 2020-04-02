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
from common.evaluate import psnr, ssim
# from skimage.measure import compare_psnr, compare_ssim
class DataTransform:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, img, fname, slice):
        image = transforms.to_tensor(img)
        image = transforms.center_crop(image.permute(2,0,1), (self.resolution, self.resolution))
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        target = image
        return image, target, mean, std


def create_data_loaders(args):
    data = SliceData(
        root=args.data_path / f'{args.data_split}',
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
    psnrl = []
    with torch.no_grad():
        for (input, target, mean, std) in data_loader:
            input = input.to(args.device)
            recons = model(input).to('cpu').squeeze(1)
            target=target.cpu()
            psnrl.append(psnr(target.numpy(), recons.numpy()))

    print(f'PSNR: {np.mean(psnrl):.2f}')
    return


def reconstructe():
    args = create_arg_parser().parse_args(sys.argv[1:])
    args.checkpoint = f'summary/{args.test_name}/best_model.pt'
    args.out_dir = f'summary/{args.test_name}/rec'
    print(f'summary/{args.test_name}')

    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    eval(args, model, data_loader)


def create_arg_parser():
    parser = Args()
    parser.add_argument('--test-name', type=str, default='test', help='name for the output dir')
    parser.add_argument('--data-split', choices=['val', 'test'],default='val',
                        help='Which data partition to run on: "val" or "test"')
    parser.add_argument('--checkpoint', type=pathlib.Path,default='summary/test/checkpoint/best_model.pt',
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path,default='summary/test/rec',
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=30, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')

    return parser


if __name__ == '__main__':
    reconstructe()
