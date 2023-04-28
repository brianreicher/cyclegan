import random
from torch.autograd import Variable
import torch


# ADAPTED FROM https://github.com/aitorzip/PyTorch-CycleGAN neeed to reference ReplayBuffer architecture to eliminate exploding/disappearning gradients
class ReplayBuffer():
    """
        CLASS TAKEN FROM https://github.com/aitorzip/PyTorch-CycleGAN repository

        Used for exploding gradient protection :) shouldn't reinvent the wheel
    """
    def __init__(self, max_size=50) -> None:
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size: int = max_size
        self.data:list = []

    def push_and_pop(self, data) -> Variable:
        to_return:list = []
        for element in data.data:
            element: torch.Tensor = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i: int = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))
