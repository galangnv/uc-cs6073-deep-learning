import time
import torch
import torch.nn.functional as F

def train_cvae(
    model,
    optimizer,
    num_epochs,
    train_loader,
    device,
    logging_interval=50,
    reconstruction_term_weight=1
):
    log_dict = {
        'train_combined_loss_per_batch': [],
        'train_combined_loss_per_epoch': [],
        'train_reconstruction_loss_per_batch': [],
        'train_kl_loss_per_batch': []
    }
    start_time = time.time()

    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (images, _) in enumerate(train_loader):

            images = images.to(device)

            # ********** Forward **********
            encoded, z_mean, z_log_var, decoded = model(images)

            # ********** Compute Loss **********
            kl_divergence = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), axis=1)
            batch_size = kl_divergence.size(0)
            kl_divergence = kl_divergence.mean()    # Average over entire batch

            pixelwise = F.mse_loss(decoded, images, reduction='none')
            pixelwise = pixelwise.view(batch_size, -1).sum(axis=1)
            pixelwise = pixelwise.mean()    # Average over entire batch

            loss = reconstruction_term_weight * pixelwise + kl_divergence

            # ********** Backpropagation **********
            loss.backward()

            # ********** Update Parameters **********
            optimizer.step()

            # ********** Logging **********
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            log_dict['train_kl_loss_per_batch'].append(kl_divergence.item())

            if not batch_idx % logging_interval:
                print(f'Epoch: {(epoch + 1):03d}/{num_epochs} | Batch: {batch_idx:04d}/{len(train_loader)} | Loss: {loss:.4f}')
        
        print(f'Time elapsed: {((time.time() - start_time) / 60):.2f} min')

    print(f'Total Training Time: {((time.time() - start_time) / 60):.2f} min')

    return log_dict