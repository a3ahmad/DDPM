import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.datasets import DatasetFolder
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import vgg

import pytorch_lightning as pl
from pytorch_lightning.plugins import DeepSpeedPlugin

import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

import argparse
from glob import glob
import os

from model import DDPM


def npy2torch(path):
    return torch.FloatTensor(np.load(path))


class FacesFolder(DatasetFolder):

    def __init__(self, root, target_transform=None, is_valid_file=None):
        super(FacesFolder, self).__init__(root, npy2torch, extensions="npy",
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        target = self.loader(path)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target


def ParseArgs():
    global args
    parser = argparse.ArgumentParser(description='StyleGLO training script')
    parser.add_argument(
        '--T',
        type=int,
        default=1000,
        help='Number of time steps')
    parser.add_argument(
        '--B1',
        type=float,
        default=1e-4,
        help='Starting beta')
    parser.add_argument(
        '--BT',
        type=float,
        default=0.02,
        help='Ending beta')
    parser.add_argument(
        '--training_path',
        default='/data/allfaces/',
        help='Location of training data')
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-6,
        help="Specify the learning-rate")
    parser.add_argument('--with_cuda', default=False, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        help='mini-batch size (default: 1)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=0,
                    help='local rank passed from distributed launcher')
    return parser.parse_args()


if __name__ == '__main__':
    args = ParseArgs()

    model = DDPM(args)

    train = FacesFolder(
            root=args.training_path,
            target_transform=T.Compose([
                #T.Lambda(lambda x: TF.convert_image_dtype(x)),
                T.Normalize(
                    (127.5, 127.5, 127.5),
                    (128, 128, 128)),
            ]))

    trainer = pl.Trainer(
                gpus=1,
                precision=16,
                plugins=DeepSpeedPlugin(
                        stage=3,
                        #partition_activations=True,
                        allgather_bucket_size=5e8,
                        reduce_bucket_size=5e8,
                        cpu_offload=True))

    trainer.fit(model, DataLoader(train, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True))
