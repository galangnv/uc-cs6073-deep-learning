import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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

def compute_average_class(
    class_idx,
    image_dim,
    data_loader,
    device=None,
    encoding_fn=None
):
    avg_img_with_class = torch.zeros(image_dim, dtype=torch.float32)
    avg_img_without_class = torch.zeros(image_dim, dtype=torch.float32)

    num_img_with_class = 0
    num_img_without_class = 0

    for images, labels in data_loader:
        idx_img_with_class = torch.Tensor([label == class_idx for label in labels]).to(torch.bool)
        
        if encoding_fn is None:
            embeddings = images
        else:
            with torch.no_grad():
                if device is not None:
                    images = images.to(device)
                embeddings = encoding_fn(images).to(torch.device('cpu'))
        
        avg_img_with_class += torch.sum(embeddings[idx_img_with_class], axis=0)
        avg_img_without_class += torch.sum(embeddings[~idx_img_with_class], axis=0)
        num_img_with_class += idx_img_with_class.sum(axis=0)
        num_img_without_class += (~idx_img_with_class).sum(axis=0)

    avg_img_with_class /= num_img_with_class
    avg_img_without_class /= num_img_without_class

    return avg_img_with_class, avg_img_without_class