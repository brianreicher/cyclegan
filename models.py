import torch.nn as nn
import torch.nn.functional as F


# ADAPTED FROM https://github.com/aitorzip/PyTorch-CycleGAN neeed to ResidualBlock, could not develop specific network architecturces myself that created sufficient results

class ResidualBlock(nn.Module):
    def __init__(self, fmaps) -> None:
        """
            Create a new ResidualBlock for our generator. Adapted from PyTorch-CycleGAN.
        """
        super(ResidualBlock, self).__init__()

        block:list = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(fmaps, fmaps, 3),
                        nn.InstanceNorm2d(fmaps),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(fmaps, fmaps, 3),
                        nn.InstanceNorm2d(fmaps)  ]

        self.conv_block: nn.Sequential = nn.Sequential(*block) # pass the block list into the conv block sequential object 

    def forward(self, x):
        """ 
            Forward pass to execute model pipeline on data and add it to the existing tensor
        """
        return self.conv_block(x) + x

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9) -> None:
        """
            Constructor, builds model pipeline
        """
        super(Generator, self).__init__()

        # conv block for immediate feature extraction       
        model:list = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.SELU(inplace=True) ]

        # execute 2 layers of downsampling to extract 128 feature maps 
        fmaps:int = 64
        out_features:int = fmaps*2
        for _ in range(2):
            # additional conv block for more features
            model += [  nn.Conv2d(fmaps, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.SELU(inplace=True) ]
            fmaps = out_features
            out_features = fmaps*2

        # execute residual blocks on the initial number of fmaps
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(fmaps)]

        # upsample downsampled data to retrieve initial sizing
        out_features = fmaps//2 # divide and round down from fmaps in case of odd value
        for _ in range(2):
            model += [  nn.ConvTranspose2d(fmaps, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            fmaps = out_features
            out_features = fmaps//2

        # output with hyperbolic tangent activation function
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model: nn.Sequential = nn.Sequential(*model)

    def forward(self, x):
        """
            Forward pass to execute model pipeline on data
        """
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc=3) -> None:
        """
            Constructor, builds model pipeline.
        """
        super(Discriminator, self).__init__()

        # repeated convolutional passes to extract features from the initial input channels (3 RGB) through 512 channels for classification
        # execute normalations after convolution to ensure standardization
        model:list = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                      nn.LeakyReLU(0.2, inplace=True)]

        inc:list = [1, 2, 4]
        for i in inc:
            model += [nn.Conv2d(64*i, 128*i, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128*i), 
                    nn.LeakyReLU(0.2, inplace=True)]

        # conv without activation for classification
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        # pass convs to squential model
        self.model: nn.Sequential = nn.Sequential(*model)

    def forward(self, x):
        """
            Forward pass to execute model pipeline on data
        """
        x =  self.model(x)
        # execute average pooling and flattening of the tensor
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)