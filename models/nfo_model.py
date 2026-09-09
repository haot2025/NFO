from re import I
import torch
from .base_model import BaseModel

###### choose network ######
from models.NFO import net_nfo as networks

import torch.nn
import torch.nn.functional
import torch.optim
import os
import nibabel as nib
import sys
import numpy as np
from tqdm import trange


class nfoModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        if is_train:
            pass

        return parser

    def __init__(self, opt):
        """Initialize the SMC GAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(nfoModel, self).__init__(opt)
        self.loss_names = ['loss_total']

        if self.isTrain:
            self.model_names = [opt.model]
        else:
            self.model_names = [opt.model]

        # define networks
        self.nfo = networks.define_network(init_type=opt.init_type,
                                              init_gain=opt.init_gain,
                                              gpu_ids=self.gpu_ids)
        
        if self.isTrain:
            self.optimizer_names = ['optimizer_1']
            self.optimizer_1 = torch.optim.Adam(self.nfo.parameters(), eps=1e-7,
                                                lr=opt.lr)
            self.optimizers.append(self.optimizer_1)
            self.l2loss = torch.nn.MSELoss()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        if self.opt.isTrain == True:
            if input['fodgt'].dtype != torch.float32:
                input['fodgt'] = input['fodgt'].float()
            if input['fodlr'].dtype != torch.float32:
                input['fodlr'] = input['fodlr'].float()

            self.fodgt = input['fodgt'].to(self.device)
            self.fodlr = input['fodlr'].to(self.device)

    def set_input_for_test(self, fodlr, brain_mask, fod_affine, fod_header):
        """ data preparation for FOD super resolution inference

        Input:
            fodlr: low angular resolution FOD array
            brain_mask: brain mask array
            fod_affine: used to indicate which affine we should use when saving super resolved fod image
            fod_header: used to indicate which header info we should use when saving super resolved fod image

        """
        self.fod_affine = fod_affine
        self.fod_header = fod_header

        # Move data to torch tensor
        self.brain_mask = torch.from_numpy(
            brain_mask.astype(np.float32)).to(self.device)

        self.brain_mask = torch.nn.functional.pad(self.brain_mask, (5, 5, 5, 5, 5, 5), "constant", 0)
        brain_mask = self.brain_mask.cpu().numpy()  # copy zero-padded brain mask tensor to brain mask array
        index_mask = np.where(brain_mask)
        self.index_mask = np.asarray(index_mask)
        self.index_length = len(self.index_mask[0])

        # Get normalised low resolution FOD images
        fodlr = torch.from_numpy(fodlr.copy()).to(self.device)
        fodlr = torch.nn.functional.pad(fodlr, (0, 0, 5, 5, 5, 5, 5, 5), "constant", 0)
        self.normalised_fodlr = fodlr  # (fodlr - self.fodlr_mean) / self.fodlr_std

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
        """
        self.fodpred = self.nfo(self.fodlr)

    def backward(self):
        """Calculate the loss """
        self.loss_total = self.l2loss(self.fodpred, self.fodgt)
        self.loss_total.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()
        self.optimizer_1.zero_grad()
        self.backward()
        self.optimizer_1.step()

    @torch.no_grad()
    def conventional_test(self, sr_fod_path):
        """Perform FOD super resolution following FOD-Net
        """
        output_directory_path = os.path.dirname(sr_fod_path)
        os.makedirs(output_directory_path, exist_ok=True)

        fodsr = torch.zeros_like(self.normalised_fodlr)
        self.normalised_fodlr = self.normalised_fodlr.permute(3, 0, 1, 2)

        size_3d_patch = 9
        margin = int(size_3d_patch / 2)

        '''fodgt_std = self.fodgt_std.squeeze(0).squeeze(0).squeeze(0)
        fodgt_mean = self.fodgt_mean.squeeze(0).squeeze(0).squeeze(0)'''

        print('Start FOD super resolution:')
        print(self.index_length)
        for i in trange(self.index_length):
            # get x, y, and z coordinates for each voxel we want to perform super resolution
            x = self.index_mask[0, i]
            y = self.index_mask[1, i]
            z = self.index_mask[2, i]

            x_start = x - margin
            x_end = x_start + size_3d_patch
            y_start = y - margin
            y_end = y_start + size_3d_patch
            z_start = z - margin
            z_end = z_start + size_3d_patch

            self.fodlr = self.normalised_fodlr[:, x_start:x_end, y_start:y_end, z_start:z_end]
            tensor_helper = self.fodlr
            self.fodlr = torch.stack(
                [self.fodlr.float(), tensor_helper.float()])
            self.forward()
            fodsr[x, y, z, :] = self.fodpred[0, :]  # * fodgt_std + fodgt_mean

        # Mask out zero regions
        fodsr *= self.brain_mask.unsqueeze(-1)
        fodsr = fodsr.detach().cpu().numpy()

        fodsr = fodsr[5:-5, 5:-5, 5:-5, :]
        print(fodsr.shape)

        # Save super resolved FOD image
        nii = nib.Nifti1Image(
            fodsr, affine=self.fod_affine, header=self.fod_header)
        nib.save(nii, sr_fod_path)


    @torch.no_grad()
    def our_fast_test(self, sr_fod_path):
        output_directory_path = os.path.dirname(sr_fod_path)
        os.makedirs(output_directory_path, exist_ok=True)

        size_3d_patch = 9
        margin = int(size_3d_patch / 2)

        print('Start FOD super resolution:')
        print(self.index_length)

        fodsr = torch.zeros_like(self.normalised_fodlr[:, :, :, :45])

        normalised_fodlr = self.normalised_fodlr.permute(3, 0, 1, 2)
        normalised_fodlr = normalised_fodlr.float()

        _, x_size, y_size, z_size = normalised_fodlr.shape
        print(f"x_size:{x_size}")
        my_index_mask = torch.from_numpy(self.index_mask)
        xs_all = my_index_mask[0, :]
        ys_all = my_index_mask[1, :]
        zs_all = my_index_mask[2, :]

        unique_yz = torch.unique(
            torch.stack([ys_all, zs_all], dim=1),
            dim=0
        )

        for yz in trange(unique_yz.shape[0]):
            y = unique_yz[yz, 0]
            z = unique_yz[yz, 1]

            yz_mask = (ys_all == y) & (zs_all == z)
            xs = xs_all[yz_mask]

            if xs.numel() == 0:
                continue

            y_start = y - margin
            y_end = y_start + size_3d_patch
            z_start = z - margin
            z_end = z_start + size_3d_patch

            patch_list = []
            valid_x_list = []
            xs = [x_value for x_value in range(4, x_size-6)]
            for x in xs:
                x_start = x - margin
                x_end = x_start + size_3d_patch

                patch = normalised_fodlr[
                    :,
                    x_start:x_end,
                    y_start:y_end,
                    z_start:z_end
                ]
                patch_list.append(patch)
                valid_x_list.append(x)

            if len(patch_list) == 0:
                continue

            self.fodlr = torch.stack(patch_list, dim=0)
            self.forward()

            # self.fodpred shape
            for idx, x in enumerate(valid_x_list):
                fodsr[x, y, z, :] = self.fodpred[idx, :]

        # Mask out zero regions
        fodsr *= self.brain_mask.unsqueeze(-1)

        fodsr = fodsr.detach().cpu().numpy()

        fodsr = fodsr[5:-5, 5:-5, 5:-5, :]

        # Save super resolved FOD image
        nii = nib.Nifti1Image(
            fodsr,
            affine=self.fod_affine,
            header=self.fod_header
        )
        nib.save(nii, sr_fod_path)