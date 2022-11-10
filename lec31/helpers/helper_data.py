from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders_eurosat(
    batch_size,
    num_workers=0,
    train_transforms=None,
    download=True
):

    # If custom transforms aren't specified, simply convert image to tensor
    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    # Get EuroSAT dataset
    train_dataset = datasets.EuroSAT(
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