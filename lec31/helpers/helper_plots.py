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
