import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import imageio
import numpy as np
import torch
import sys
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

sys.path.append('/Users/brianreicher/Documents/GitHub/cyclegan')

# self written function imports
from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from data import DataClass


class CycleGAN():
    """
        Focused on this code -- network architecures very specifc to image channels, wanted to enforce a proper training pipeline for generators and discriminators.
    """
    def __init__(self, data_dr:str, input_nc=3, output_nc=3, lr=0.0002, batch_size=1) -> None:
        # general params
        self.input_nc: int = input_nc
        self.output_nc: int = output_nc
        self.lr: float = lr
        self.batch_size: int = batch_size
        self.data_directory: str = data_dr

        # initialize networks
        self.generator_A2B = None
        self.generator_B2A = None
        self.discriminator_A = None
        self.discriminator_B = None

        # initialize dataloder
        self.dataloader = None

        # initialize loss functions
        self.gan_loss: torch.nn.MSELoss = torch.nn.MSELoss()
        self.cycle_loss: torch.nn.L1Loss = torch.nn.L1Loss()
        self.id_loss: torch.nn.L1Loss = torch.nn.L1Loss()


        # data memory space constraints
        Tensor:torch.Tensor = torch.Tensor
        self.input_A:Tensor = Tensor(self.batch_size, self.input_nc, 256, 256)
        self.input_B:Tensor = Tensor(self.batch_size, self.output_nc, 256, 256)

        # tensor holders for output tensors
        self.output_Real: Variable = Variable(Tensor(self.batch_size).fill_(1.0), requires_grad=False)
        self.output_Generated: Variable = Variable(Tensor(self.batch_size).fill_(0.0), requires_grad=False)

        # initialize buffers for generated data
        self.generated_A_buffer: ReplayBuffer = ReplayBuffer()
        self.generated_B_buffer: ReplayBuffer = ReplayBuffer()

    def build_pipeline(self) -> None:
        # initialize actual networks
        self.generator_A2B: Generator = Generator(self.input_nc, self.output_nc)
        self.generator_B2A: Generator = Generator(self.output_nc, self.input_nc)
        self.discriminator_A: Discriminator = Discriminator(self.input_nc)
        self.discriminator_B: Discriminator = Discriminator(self.output_nc)

        def weights_init_normal(m) -> None:
            """
                Helper function to initalize weights depending on Convolutional or BatchNormilization class names
            """
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                torch.nn.init.normal(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant(m.bias.data, 0.0)

        # apply initial weights for first forward pass
        self.generator_A2B.apply(weights_init_normal)
        self.generator_B2A.apply(weights_init_normal)
        self.discriminator_A.apply(weights_init_normal)
        self.discriminator_B.apply(weights_init_normal)

        # apply optimizers to the networks, for backpropagation purposes, with given learning rate
        # used ADAM optimizer instead of GD or SGD, need to create for the total generators and each discriminator
        self.generator_optimizer = torch.optim.Adam(itertools.chain(self.generator_A2B.parameters(), self.generator_B2A.parameters()), lr=self.lr, betas=(0.5, 0.997))
        self.discriminator_A_optimizer = torch.optim.Adam(self.discriminator_A.parameters(), lr=self.lr, betas=(0.5, 0.997))
        self.discriminator_B_optimizer = torch.optim.Adam(self.discriminator_B.parameters(), lr=self.lr, betas=(0.5, 0.997))

    def load_data(self) -> None:
        """
            Load the image data directory to the dataloader object
        """

        # transform list to apply to images, using specific parameters tested by previous networks
        tform:list = [ transforms.Resize(int(256*1.5), Image.BICUBIC), 
                        transforms.RandomCrop(256), 
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    
        self.dataloader:DataLoader = DataLoader(DataClass(self.data_directory, tforms=tform), 
                                                batch_size=self.batch_size, num_workers=4)

    def train(self, n_epochs:int, display_loss=True, save_img=False) -> None:
        """
            Train the initialized networks
        """
        # cumulative loss lists for plotting
        loss_G_list:list = []
        loss_G_identity_list:list = []
        loss_G_GAN_list:list = []
        loss_G_cycle_list:list = []
        loss_D_list:list = []

        for _ in tqdm(range(n_epochs)):
            # iterate for a given number of epochs
            for i, batch in enumerate(self.dataloader):
                # iterate over each item and batch in the dataloader


                # establish iteration input
                real_A: Variable = Variable(self.input_A.copy_(batch['A']))
                real_B: Variable = Variable(self.input_B.copy_(batch['B']))

                # TRAIN BOTH GENERATORS, FROM STYLE A->B and B->A WITH CYCLES
                self.generator_optimizer.zero_grad()

                # Identity loss
                same_B = self.generator_A2B(real_B)


                loss_identity_B = self.id_loss(same_B, real_B)*7.0

                same_A = self.generator_B2A(real_A)
                loss_identity_A = self.id_loss(same_A, real_A)*7.0

                fake_B = self.generator_A2B(real_A)
                pred_fake = self.discriminator_B(fake_B)
                loss_GAN_A2B = self.gan_loss(pred_fake, self.output_Real)

                fake_A = self.generator_B2A(real_B)
                pred_fake = self.discriminator_A(fake_A)
                loss_GAN_B2A = self.gan_loss(pred_fake, self.output_Real)

                recon_A = self.generator_B2A(fake_B)
                loss_cycle_ABA = self.cycle_loss(recon_A, real_A)*9.0

                recon_B = self.generator_A2B(fake_A)
                loss_cycle_BAB = self.cycle_loss(recon_B, real_B)*9.0

                # total generator loss
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                loss_G.backward()
                
                self.generator_optimizer.step()
            

                # TRAIN A-STYLE DISCRIMINATOR
                self.discriminator_A_optimizer.zero_grad()

                # real
                pred_real = self.discriminator_A(real_A)
                loss_D_real = self.gan_loss(pred_real, self.output_Real)

                # generated
                fake_A = self.generated_A_buffer.push_and_pop(fake_A)
                pred_fake = self.discriminator_A(fake_A.detach())
                loss_D_fake = self.gan_loss(pred_fake, self.output_Generated)

                # compound loss
                loss_D_A = loss_D_real + loss_D_fake
                loss_D_A.backward()

                self.discriminator_A_optimizer.step()


                # TRAIN B-STYLE DISCRIMINATOR
                self.discriminator_B_optimizer.zero_grad()

                # real
                pred_real = self.discriminator_B(real_B)
                loss_D_real = self.gan_loss(pred_real, self.output_Real)
                
                # generated
                fake_B = self.generated_B_buffer.push_and_pop(fake_B)
                pred_fake = self.discriminator_B(fake_B.detach())
                loss_D_fake = self.gan_loss(pred_fake, self.output_Generated)

                # compound loss
                loss_D_B = loss_D_real + loss_D_fake
                loss_D_B.backward()

                self.discriminator_B_optimizer.step()


                if display_loss:
                    losses = {'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                                'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}
                    print(losses)
                    loss_G_list.append(losses['loss_G'].item())
                    loss_G_identity_list.append(losses['loss_G_identity'].item())
                    loss_G_GAN_list.append(losses['loss_G_GAN'].item())
                    loss_G_cycle_list.append(losses['loss_G_cycle'].item())
                    loss_D_list.append(losses['loss_D'].item())
                    plt.close('all')

                    # create a line plot for each loss
                    plt.plot(loss_G_list, label='loss_G')
                    plt.plot(loss_G_identity_list, label='loss_G_identity')
                    plt.plot(loss_G_GAN_list, label='loss_G_GAN')
                    plt.plot(loss_G_cycle_list, label='loss_G_cycle')
                    plt.plot(loss_D_list, label='loss_D')

                    # add title, labels, and legend to the plot
                    plt.title('CycleGAN Loss over Time')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.legend()

                    # show the plot
                    plt.show()


                if save_img:
                    iter_imgs: dict = {
                        'real_A': ((real_A+1)/2).squeeze().detach().cpu().numpy(),
                        'real_B': ((real_B+1)/2).squeeze().detach().cpu().numpy(),
                        'fake_A': ((fake_A+1)/2).squeeze().detach().cpu().numpy(),
                        'fake_B': ((fake_B+1)/2).squeeze().detach().cpu().numpy()
                    }

                    if not os.path.exists('./results'):
                        os.makedirs('./results')

                    # Save the rendered grid to disk as an image
                    for key, im in iter_imgs.items():
                        imageio.imwrite(f'./results/{key}_{i}.png', np.transpose(im, (1, 2, 0)))


if __name__ == '__main__':
    cgan:CycleGAN = CycleGAN(data_dr='./datasets/cezanne2photo')
    cgan.build_pipeline()
    cgan.load_data()

    cgan.train(n_epochs=200, display_loss=True, save_img=False)
