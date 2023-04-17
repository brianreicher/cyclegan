import os
import torch
import functools
import itertools

from collections import OrderedDict
from .cgan.cyclegan_model import CycleGAN_Model
from .cgan.cyclegan_loss import CycleGAN_Loss
from .cgan.cyclegan_optim import CycleGAN_Optimizer
from utils import *
from .networks.unet import *


# Normalization layer helper func
def get_normalization(n_dims: int):
    if n_dims == 3:  # 3D case
        norm_instance = torch.nn.InstanceNorm3d
    elif n_dims == 2:  # 2D case
        norm_instance = torch.nn.InstanceNorm2d
    # Initiate norm_layer  based on norm_instance
    norm_layer = functools.partial(norm_instance, affine=False, track_running_stats=False)
    return norm_layer


class CycleGAN():

    def __init___(self, gnet_type, gnet_kwargs, g_init_learning_rate, dnet_type, dnet_kwargs, 
                  d_init_learning_rate, loss_kwargs, adam_betas, ndims):
        # Initiate generator
        self.gnet_type = gnet_type
        self.gnet_kwargs = gnet_kwargs
        self.g_init_learning_rate = g_init_learning_rate
        # Initiate discriminator
        self.dnet_type = dnet_type
        self.dnet_kwargs = dnet_kwargs
        self.d_init_learning_rate = d_init_learning_rate
        # Initiate loss and optimizer
        # self.loss_style = loss_style
        self.loss_kwargs = loss_kwargs
        self.adam_betas = adam_betas
        self.ndims = ndims
        # Set CUDA device
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.visual_names = ["real_A", "fake_B", "cycled_A", "real_B", "fake_A", "cycled_B"]
        self.model_names = ['G1', 'G2', 'D1', 'D2']

    def set_downsample_factors(self):
        if 'downsample_factors' not in self.gnet_kwargs:
            down_factor = 2 if 'down_factor' not in self.gnet_kwargs else self.gnet_kwargs.pop('down_factor')
            num_downs = 3 if 'num_downs' not in self.gnet_kwargs else self.gnet_kwargs.pop('num_downs')
            self.gnet_kwargs.update({'downsample_factors': [(down_factor,)*self.ndims,] * (num_downs - 1)})

    def get_generator(self, gnet_kwargs=None):
        if gnet_kwargs is None:
            gnet_kwargs = self.gnet_kwargs

        # Initiate norm_layer  based on norm_instance
        norm_layer = get_normalization(self.ndims)
        self.gnet_kwargs.update({'norm_layer': norm_layer})

        if self.gnet_type == 'unet':
            generator = torch.nn.Sequential(UNet(**gnet_kwargs), torch.nn.Tanh())
                                            
        elif self.gnet_type == 'resnet':
            if self.ndims == 2:
                generator = ResnetGenerator(**gnet_kwargs)
            
            elif self.ndims == 3:
                generator = ResnetGenerator3D(**gnet_kwargs)

            else:
                raise f'Resnet generators only specified for 2D or 3D, not {self.ndims}D'

        else:
            raise f'Unknown generator type requested: {self.gnet_type}'

        activation = gnet_kwargs['activation'] if 'activation' in gnet_kwargs else torch.nn.ReLU

        if activation is not None:
            init_weights(generator, init_type='kaiming', nonlinearity=activation.__class__.__name__.lower())
        else:
            init_weights(generator, init_type='normal', init_gain=0.05) #TODO: MAY WANT TO ADD TO CONFIG FILE
        return generator

    # TODO: Add options for resnet, unet --> pass kwargs, throw out type (temporary)
    def get_discriminator(self, dnet_kwargs=None):
        # Initiate norm_layer  based on norm_instance
        norm_layer = get_normalization(self.ndims)
        self.dnet_kwargs.update({'norm_layer': norm_layer})

        if dnet_kwargs is None:
            dnet_kwargs = self.dnet_kwargs

        if self.dnet_type == 'unet': 
            # TODO
            discriminator = torch.nn.Sequential(UNet(**dnet_kwargs, ngf=64, fmap_inc_factor=None,
                                                norm_layer=norm_layer, activation=torch.nn.Tanh()))

        elif self.dnet_type == 'patchgan':
            if self.ndims == 2:
                discriminator = NLayerDiscriminator(**dnet_kwargs)
            
            elif self.ndims == 3:
                discriminator = NLayerDiscriminator3D(**dnet_kwargs)
        
        elif self.dnet_type == 'resnet':  # TODO
            # TODO
            raise f'Incomplete generator type requested: resnet'

        else:
            raise f'Unknown generator type requested: {self.gnet_type}'

        init_weights(discriminator, init_type='kaiming') # Initialize weights and set func
        return discriminator

    def setup_networks(self):
        self.netG1 = self.get_generator()
        self.netG2 = self.get_generator()
        
        self.netD1 = self.get_discriminator()
        self.netD2 = self.get_discriminator()

    def setup_model(self):
        if not hasattr(self, 'netG1'):
            self.setup_networks()
        
        self.model = CycleGAN_Model(self.netG1, self.netD1, self.netG2, self.netD2)
        self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(), lr=self.g_init_learning_rate, betas=self.adam_betas)
        self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=self.g_init_learning_rate, betas=self.adam_betas)
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD1.parameters(), self.netD2.parameters()), lr=self.d_init_learning_rate, betas=self.adam_betas)
        self.optimizer = CycleGAN_Optimizer(self.optimizer_G1, self.optimizer_G2, self.optimizer_D)
        self.loss = CycleGAN_Loss(self.netD1, self.netG1, self.netD2, self.netG2, self.optimizer_G1, self.optimizer_G2, self.optimizer_D, self.ndims, **self.loss_kwargs)


    def set_input(self, input): # pass self.real_# to self.model() (which is the CyleGAN_Split_Model)
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    def optimize_parameters(self):
        self.fake_B, self.cycled_B, self.fake_A, self.cycled_A = self.model(self.real_A, self.real_B)
        _ = self.loss(self.real_A, self.fake_A, self.cycled_A, self.real_B, self.fake_B, self.cycled_B)
   
    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        return self.loss.loss_dict
    
    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
