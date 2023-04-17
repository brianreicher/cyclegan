import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# Define the Generator and Discriminator networks
class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, hidden_dim=64) -> None:
        super(Generator, self).__init__()
        self.input_channels: int = input_channels
        self.output_channels: int = output_channels
        self.hidden_dim: int = hidden_dim

        # Define the architecture of the generator network
        self.layers: nn.Sequential = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim * 4, self.output_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Define the forward pass of the generator network
        x = self.layers(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64) -> None:
        super(Discriminator, self).__init__()
        self.input_channels: int = input_channels
        self.hidden_dim: int = hidden_dim

        # Define the architecture of the discriminator network
        self.layers:nn.Sequential = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hidden_dim * 4, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Define the forward pass of the discriminator network
        x = self.layers(x)
        return x

# Define the CycleGAN model
class CycleGAN(nn.Module):
    def __init__(self, generator_A, generator_B, discriminator_A, discriminator_B) -> None:
        super(CycleGAN, self).__init__()
        self.generator_A = generator_A
        self.generator_B = generator_B
        self.discriminator_A = discriminator_A
        self.discriminator_B = discriminator_B

    def forward(self, real_A, real_B) -> tuple:
        if real_A is not None:
            fake_B = self.generator_A(real_A)
            cycled_A = None
        else:
            fake_B: None = None
            cycled_A: None = None

        if real_B is not None:
            fake_A = self.generator_B(real_B)
            cycled_B = None
            
        else:
            fake_A: None = None
            cycled_B: None = None

        return fake_B, cycled_B, fake_A, cycled_A

# Define the dataset and data loader
class ImageDataset(Dataset):
    def __init__(self, root, transform=None) -> None:
        self.dataset: ImageFolder = ImageFolder(root, transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> tuple:
        return self.dataset[index]


if __name__ == '__main__':
    # Define hyperparameters and other settings
    batch_size:int = 1
    learning_rate: float = 0.0002
    epochs:int = 200

    # Define data transforms for image preprocessing
    # You can customize these transforms based on your dataset
    transform: transforms.Compose = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create the data loader for dataset A
    dataset_A: ImageDataset = ImageDataset('./data/cezanne2photo', transform=transform)
    dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create the data loader for dataset B
    dataset_B: ImageDataset = ImageDataset('./data/cezanne2photo', transform=transform)
    dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the Generator and Discriminator networks
    generator_A: Generator = Generator()
    generator_B: Generator = Generator()
    discriminator_A: Discriminator = Discriminator()
    discriminator_B: Discriminator = Discriminator()

    # Initialize the CycleGAN model
    cyclegan: CycleGAN = CycleGAN(generator_A, generator_B, discriminator_A, discriminator_B)

    # Define the loss functions for the CycleGAN
    criterion_GAN: nn.BCELoss = nn.BCELoss()
    criterion_cycle: nn.L1Loss = nn.L1Loss()

    # Define the optimizers for the Generator and Discriminator networks
    optimizer_G: optim.Adam = optim.Adam(list(generator_A.parameters()) + list(generator_B.parameters()), lr=learning_rate)
    optimizer_D: optim.Adam = optim.Adam(list(discriminator_A.parameters()) + list(discriminator_B.parameters()), lr=learning_rate)

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cyclegan.to(device)
    criterion_GAN.to(device)
    criterion_cycle.to(device)
    generator_A.to(device)
    generator_B.to(device)
    discriminator_A.to(device)
    discriminator_B.to(device)
    dataloader_A: DataLoader = DataLoader(dataset_A, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_B: DataLoader = DataLoader(dataset_B, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Training loop
    for epoch in range(epochs):
        for batch_idx, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
            # Move real A and real B to device (e.g. GPU)
            real_A = real_A[0].to(device)
            real_B = real_B[0].to(device)

            # Update Discriminator networks
            optimizer_D.zero_grad()

            # Train Discriminator A
            fake_A, _, _, _ = cyclegan(real_A, real_B)
            pred_real_A = discriminator_A(real_A)
            pred_fake_A = discriminator_A(fake_A.detach())
            loss_D_real_A = criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))
            loss_D_fake_A = criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
            loss_D_A.backward()

            # Train Discriminator B
            fake_B, _, _, _ = cyclegan(real_A, real_B)
            pred_real_B = discriminator_B(real_B)
            pred_fake_B = discriminator_B(fake_B.detach())
            loss_D_real_B = criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))
            loss_D_fake_B = criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
            loss_D_B.backward()

            optimizer_D.step()

            # Update Generator networks
            optimizer_G.zero_grad()

            # Train Generator A
            fake_B, cycled_B, _, _ = cyclegan(real_A, real_B)
            pred_fake_B = discriminator_B(fake_B)
            loss_GAN_A = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))
            loss_cycle_A = criterion_cycle(cycled_B, real_A)
            loss_G_A = loss_GAN_A + (10 * loss_cycle_A)
            loss_G_A.backward()

            # Train Generator B
            fake_A, _, _, cycled_A = cyclegan(real_A, real_B)
            pred_fake_A = discriminator_A(fake_A)
            loss_GAN_B = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))
            loss_cycle_B = criterion_cycle(cycled_A, real_B)
            loss_G_B = loss_GAN_B + (10 * loss_cycle_B)
            loss_G_B.backward()

            optimizer_G.step()

            # Print losses
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(dataloader_A)}] Loss_D_A: {loss_D_A.item():.4f} "
                        f"Loss_D_B: {loss_D_B.item():.4f} Loss_G_A: {loss_G_A.item():.4f} Loss_G_B: {loss_G_B.item():.4f}")

