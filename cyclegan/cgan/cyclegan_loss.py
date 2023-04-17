import torch
from utils import *

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.Logger('CycleGAN_Loss', 'INFO')

class CycleGAN_Loss(torch.nn.Module):
    def __init__(self, 
                netD1, 
                netG1, 
                netD2, 
                netG2, 
                optimizer_D, 
                optimizer_G, 
                dims,
                l1_loss = torch.nn.SmoothL1Loss(), 
                l1_lambda=100, 
                identity_lambda=0,
                gan_mode='lsgan'
                 ):
        super(CycleGAN_Loss, self).__init__()
        self.l1_loss = l1_loss
        self.gan_loss = GANLoss(gan_mode=gan_mode)
        self.netD1 = netD1 # differentiates between fake and real Bs
        self.netG1 = netG1 # turns As into Bs
        self.netD2 = netD2 # differentiates between fake and real As
        self.netG2 = netG2 # turns Bs into As
        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G
        self.l1_lambda = l1_lambda
        self.identity_lambda = identity_lambda
        self.gan_mode = gan_mode
        self.dims = dims
        self.loss_dict = {
            'Loss/D1': float(),
            'Loss/D2': float(),
            'Loss/cycle': float(),
            'GAN_Loss/G1': float(),
            'GAN_Loss/G2': float(),
        }

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-self.dims] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def clamp_weights(self, net, min=-0.01, max=0.01):
        for module in net.model:
            if hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                temp = module.weight.data
                module.weight.data = temp.clamp(min, max)

    def backward_D(self, Dnet, real, fake):
        # Real
        pred_real = Dnet(real)
        loss_D_real = self.gan_loss(pred_real, True)
        
        # Fake; stop backprop to the generator by detaching fake
        pred_fake = Dnet(fake.detach())
        loss_D_fake = self.gan_loss(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_Ds(self, real_A, fake_A, real_B, fake_B, n_loop=5):
        # self.set_requires_grad([self.netG1, self.netG2], False)  # G does not require gradients when optimizing D
        self.set_requires_grad([self.netD1, self.netD2], True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero

        if self.gan_mode.lower() == 'wgangp': # Wasserstein Loss
            for _ in range(n_loop):
                self.loss_D1 = self.backward_D(self.netD1, real_B, fake_B)
                self.loss_D2 = self.backward_D(self.netD2, real_A, fake_A)
                self.optimizer_D.step()          # update D's weights
                self.clamp_weights(self.netD1)
                self.clamp_weights(self.netD2)
        else:
            self.loss_D1 = self.backward_D(self.netD1, real_B, fake_B)
            self.loss_D2 = self.backward_D(self.netD2, real_A, fake_A)
            self.optimizer_D.step()          # update D's weights            
        
        #return losses
        return self.loss_D1, self.loss_D2

    def backward_G(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B):
        self.set_requires_grad([self.netD1, self.netD2], False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero

        #get cycle loss for both directions (i.e. real == cycled, a.k.a. real_A == netG2(netG1(real_A)) for A and B)
        # crop if necessary
        if real_A.size()[-self.dims:] != cycled_A.size()[-self.dims:]:
            l1_loss_A = self.l1_loss(self.crop(real_A, cycled_A.size()[-self.dims:]), cycled_A)
            l1_loss_B = self.l1_loss(self.crop(real_B, cycled_B.size()[-self.dims:]), cycled_B)
        else:
            l1_loss_A = self.l1_loss(real_A, cycled_A)
            l1_loss_B = self.l1_loss(real_B, cycled_B)        
        self.loss_dict.update({
            'Cycle_Loss/A': float(l1_loss_A),                
            'Cycle_Loss/B': float(l1_loss_B),                
        })
        cycle_loss = self.l1_lambda * (l1_loss_A + l1_loss_B)

        #get identity loss (i.e. ||G_A(B) - B|| for G_A(A) --> B) if applicable
        if self.identity_lambda > 0:
            identity_B = self.netG1(real_B)
            identity_A = self.netG2(real_A)
            if real_A.size()[-self.dims:] != identity_A.size()[-self.dims:]:
                identity_loss_B = self.l1_loss(self.crop(real_B, identity_B.size()[-self.dims:]), identity_B)
                identity_loss_A = self.l1_loss(self.crop(real_A, identity_A.size()[-self.dims:]), identity_A)
            else:
                identity_loss_B = self.l1_loss(real_B, identity_B)#TODO: add ability to have unique loss function for identity
                identity_loss_A = self.l1_loss(real_A, identity_A)
            self.loss_dict.update({
                'Identity_Loss/A': float(identity_loss_A),                
                'Identity_Loss/B': float(identity_loss_B),                
            })
        else:
            identity_loss_B = 0
            identity_loss_A = 0
        identity_loss = self.identity_lambda * (identity_loss_A + identity_loss_B)

        #Then G1 discriminator loss first
        gan_loss_G1 = self.gan_loss(self.netD1(fake_B), True)

        #Then G2 discriminator loss
        gan_loss_G2 = self.gan_loss(self.netD2(fake_A), True)
        
        #Sum all losses
        self.loss_G = cycle_loss + identity_loss + gan_loss_G1 + gan_loss_G2

        #Calculate gradients
        self.loss_G.backward()

        #Step optimizer
        self.optimizer_G.step()             # udpate G's weights

        #return losses
        return cycle_loss, gan_loss_G1, gan_loss_G2

    def forward(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B):

        # crop if necessary
        if real_A.size()[-self.dims:] != fake_B.size()[-self.dims:]:
            real_A = self.crop(real_A, fake_A.size()[-self.dims:])
            real_B = self.crop(real_B, fake_B.size()[-self.dims:])

        # update Gs
        cycle_loss, gan_loss_G1, gan_loss_G2 = self.backward_G(real_A, fake_A, cycled_A, real_B, fake_B, cycled_B)
        
        # update Ds
        loss_D1, loss_D2 = self.backward_Ds(real_A, fake_A, real_B, fake_B)

        self.loss_dict.update({
            'Loss/D1': float(loss_D1),
            'Loss/D2': float(loss_D2),
            'Loss/cycle': float(cycle_loss),
            'GAN_Loss/G1': float(gan_loss_G1),
            'GAN_Loss/G2': float(gan_loss_G2),
        })

        total_loss = self.loss_G.detach()
        # define dummy backward pass to disable Gunpowder's Train node loss.backward() call
        total_loss.backward = lambda: None

        logger.info(self.loss_dict)
        return total_loss

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad