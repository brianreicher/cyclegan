import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# ADAPTED FROM https://github.com/aitorzip/PyTorch-CycleGAN
class DataClass(Dataset):
    def __init__(self, data_directory, tforms=None) -> None:
        self.transform: transforms.Compose = transforms.Compose(tforms)
        self.files_A: list[str] = glob.glob(os.path.join(data_directory, 'train/A') + '/*.*')
        self.files_B: list[str] = glob.glob(os.path.join(data_directory, 'train/B') + '/*.*')

    def __getitem__(self, idx) -> dict:
        iA: Image = self.transform(Image.open(self.files_A[idx % len(self.files_A)]))
        iB:Image = self.transform(Image.open(self.files_B[idx % len(self.files_B)]))

        return {'A': iA, 'B': iB}

    def __len__(self) -> int:
        return max(len(self.files_A), len(self.files_B))