import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# ADAPTED FROM https://github.com/aitorzip/PyTorch-CycleGAN
class DataClass(Dataset):
    def __init__(self, root, tforms=None, mode='train') -> None:
        self.transform: transforms.Compose = transforms.Compose(tforms)
        self.files_A: list[str] = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B: list[str] = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index) -> dict:
        item_A: Image = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B:Image = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self) -> int:
        return max(len(self.files_A), len(self.files_B))