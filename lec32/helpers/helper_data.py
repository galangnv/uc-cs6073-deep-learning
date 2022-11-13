import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders_mnist(
    batch_size,
    num_workers=0,
    train_transforms=None,
    download=True
):

    # If custom transforms aren't specified, simply convert image to tensor
    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    # Get EuroSAT dataset
    train_dataset = datasets.MNIST(
        root='data',
        transform=train_transforms,
        download=download
    )

    # Create data loader for train dataset
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader