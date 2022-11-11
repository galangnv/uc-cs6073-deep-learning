import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_generated_images(
    model,
    data_loader,
    device,
    num_images=15
):
    _, axes = plt.subplots(
        nrows=2, ncols=num_images,
        sharex=True, sharey=True,
        figsize=(20, 2.5)
    )

    for images, _ in data_loader:

        images = images.to(device)

        color_channels = images.shape[1]
        image_height = images.shape[2]
        image_width = images.shape[3]

        with torch.no_grad():
            _, _, _, decoded_images = model(images)[:num_images]
        
        original_images = images[:num_images]
        break

    for i in range(num_images):
        for ax, img in zip(axes, [original_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device('cpu'))
            
            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax[i].imshow(curr_img)
            else:
                ax[i].imshow(curr_img.view((image_height, image_width)), cmap='binary')

def plot_modified_classes(
    original,
    diff,
    diff_coefficients=(0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.),
    decoding_fn=None,
    device=None
):
    _, axes = plt.subplots(
        nrows=1, ncols=len(diff_coefficients),
        sharex=True, sharey=True,
        figsize=(11, 1.5)
    )

    for i, alpha in enumerate(diff_coefficients):
        transition = original + alpha * diff

        if decoding_fn is not None:

            with torch.no_grad():
                if device is not None:
                    transition = transition.to(device)
                
                transition = decoding_fn(transition).to(torch.device('cpu')).squeeze(0)
        
        if not alpha:
            s = 'original'
        else:
            s = f'$\\alpha=${alpha}'

        axes[i].set_title(s)
        axes[i].imshow(transition.permute(1, 2, 0))
        axes[i].axison = False