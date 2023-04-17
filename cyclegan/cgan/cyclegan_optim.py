import torch

class CycleGAN_Optimizer(torch.nn.Module):
    def __init__(self, optimizer_G, optimizer_D) -> None:
        super(CycleGAN_Optimizer, self).__init__()
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D

    def step(self) -> None:
        pass