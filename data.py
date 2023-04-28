import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# ADAPTED FROM https://github.com/aitorzip/PyTorch-CycleGAN
class DataClass(Dataset):
    def __init__(self, data_directory, tforms=None) -> None:
        # transform composition to later execute on each image
        self.transform: transforms.Compose = transforms.Compose(tforms)

        # use GLOB to load filenames into a list for the data directory for each datatype (A & B)
        self.files_A: list[str] = glob.glob(os.path.join(data_directory, 'train/A') + '/*.*')
        self.files_B: list[str] = glob.glob(os.path.join(data_directory, 'train/B') + '/*.*')

    def __getitem__(self, idx) -> dict:
        """
            Get an indexed item of both type A an B data and apply transforms and convert to a PIL image
        """
        iA: Image = self.transform(Image.open(self.files_A[idx % len(self.files_A)]))
        iB:Image = self.transform(Image.open(self.files_B[idx % len(self.files_B)]))

        return {'A': iA, 'B': iB}

    def __len__(self) -> int:
        """
            Returns the size of the dataset, whichver datatype has more in its path

        """
        return max(len(self.files_A), len(self.files_B))