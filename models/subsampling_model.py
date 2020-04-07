import torch
from torch import nn
from models.rec_models.unet_model import UnetModel
import data.transforms as transforms
import numpy as np
import scipy.io as sio
from util.sphere_interp_torch import *
from dipy.core.sphere import disperse_charges, HemiSphere


class Subsampling_Layer(nn.Module):
    def __init__(self, total_directions, dir_decimation_rate, direction_learning, initialization):
        super().__init__()

        self.total_directions = total_directions
        self.dir_decimation_rate = dir_decimation_rate
        self.sample_directions = total_directions//dir_decimation_rate

        # init invB matrix for dwi to sph in the forward pass
        mat = sio.loadmat('/home/tomerweiss/Datasets/dMRI/dir90.mat')
        self.orig_phi = torch.tensor(mat['phi'],device='cuda').float()
        self.orig_theta = torch.tensor(mat['theta'],device='cuda').float()
        self.sh_order = 12
        self.smooth = 0.006
        Ba, m, n = sph_harm_basis_torch(self.sh_order, self.orig_theta, self.orig_phi, device='cuda')
        L = -n * (n + 1)
        self.invB = smooth_pinv(Ba, np.sqrt(self.smooth) * L.float())

        self.initialize_dir(self.sample_directions, direction_learning, initialization)

    def forward(self, input):
        with torch.no_grad():
            self.theta.data = self.theta.clamp(0, np.pi / 2)
            self.phi.data = self.phi.clamp(-np.pi, np.pi)
        input = input.permute(0, 2, 3, 1)
        dwi_sh = torch.matmul(input, self.invB.t())  #dwi2sh(input,self.orig_theta,self.orig_phi,self.sh_order,self.smooth)
        output = sh2dwi(dwi_sh, self.theta, self.phi, self.sh_order, self.smooth)
        return output.permute(0, 3, 1, 2)

    def initialize_dir(self, dir_num,direction_learning,initialization):
        theta = np.pi * np.random.rand(dir_num)
        phi = 2 * np.pi * np.random.rand(dir_num)
        hsph_initial = HemiSphere(theta=theta, phi=phi)
        if initialization == 'uniform':
            hsph_updated, potential = disperse_charges(hsph_initial, 5000)                                               # optimize
            self.theta = torch.nn.Parameter(torch.tensor(hsph_updated.theta).float(), requires_grad=bool(int(direction_learning)))
            self.phi = torch.nn.Parameter(torch.tensor(hsph_updated.phi).float(), requires_grad=bool(int(direction_learning)))
        if initialization == 'random':
            self.theta = torch.nn.Parameter(torch.tensor(hsph_initial.theta).float(), requires_grad=bool(int(direction_learning)))     # random
            self.phi = torch.nn.Parameter(torch.tensor(hsph_initial.phi).float(), requires_grad=bool(int(direction_learning)))
        return

    def get_directions(self):
        return self.phi, self.theta

    def __repr__(self):
        return f'Subsampling_Layer'

class Subsampling_Model(nn.Module):
    def __init__(self, out_directions,  dir_decimation_rate, direction_learning, initialization, chans, num_pool_layers, drop_prob):
        super().__init__()

        self.subsampling = Subsampling_Layer(out_directions, dir_decimation_rate, direction_learning, initialization)
        self.reconstruction_model = UnetModel(out_directions // dir_decimation_rate, out_directions, chans, num_pool_layers, drop_prob)

    def forward(self, input):
        input  = self.subsampling(input)
        output = self.reconstruction_model(input)
        return output

    def get_directions(self):
        return self.subsampling.get_directions()
