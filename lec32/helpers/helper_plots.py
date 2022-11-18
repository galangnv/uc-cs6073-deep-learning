import matplotlib.pyplot as plt
import numpy as np

def plot_generated_images(log_dict, num_epochs):
    for i in range(0, num_epochs, 5):
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.title(f'Generated images at epochs {i}')
        plt.imshow(np.transpose(log_dict['images_from_noise_per_epoch'][i], (1, 2, 0)))
        plt.show()
    
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title(f'Generated images after last epoch')
    plt.imshow(np.transpose(log_dict['images_from_noise_per_epoch'][-1], (1, 2, 0)))
    plt.show()