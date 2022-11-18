import time
import torch
import torch.nn.functional as F
import torchvision

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
            optimizer.zero_grad()
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

def train_gan(
    model,
    generator_optimizer,
    discriminator_optimizer,
    num_epochs,
    latent_dim,
    train_loader,
    device,
    logging_interval=100
):
    log_dict = {
        'train_generator_loss_per_batch': [],
        'train_discriminator_loss_per_batch': [],
        'train_discriminator_real_acc_per_batch': [],
        'train_discriminator_fake_acc_per_batch': [],
        'images_from_noise_per_epoch': []
    }
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    start_time = time.time()

    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (images, _) in enumerate(train_loader):

            batch_size = images.size(0)

            # Real images
            real_images = images.to(device)
            real_labels = torch.ones(batch_size, device=device)

            # Generated/Fake images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = model.generator_forward(noise)
            fake_labels = torch.zeros(batch_size, device=device)
            flipped_fake_labels = real_labels

            # ********** Train Discriminator **********

            discriminator_optimizer.zero_grad()

            # Discriminator loss on real images
            real_predictions = model.discriminator_forward(real_images).view(-1)
            real_loss = F.binary_cross_entropy_with_logits(real_predictions, real_labels)

            # Discriminator loss on fake images
            fake_predictions = model.discriminator_forward(fake_images).view(-1)
            fake_loss = F.binary_cross_entropy_with_logits(fake_predictions, fake_labels)

            # Combined loss
            discriminator_loss = 0.5 * (real_loss + fake_loss)
            discriminator_loss.backward(retain_graph=True)

            discriminator_optimizer.step()

            # ********** Train Generator **********

            generator_optimizer.zero_grad()

            # Discriminator loss on fake images with flipped labels
            fake_predictions = model.discriminator_forward(fake_images).view(-1)
            generator_loss = F.binary_cross_entropy_with_logits(fake_predictions, flipped_fake_labels)
            generator_loss.backward(retain_graph=True)

            generator_optimizer.step()

            # ********** Logging **********

            log_dict['train_generator_loss_per_batch'].append(generator_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discriminator_loss.item())

            predicted_labels_real = torch.where(real_predictions.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(fake_predictions.detach() > 0., 1., 0.)
            real_acc = (predicted_labels_real == real_labels).float().mean() * 100.
            fake_acc = (predicted_labels_fake == fake_labels).float().mean() * 100.
            log_dict['train_discriminator_real_acc_per_batch'].append(real_acc.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(fake_acc.item())

            if not batch_idx % logging_interval:
                print(f'Epoch: {(epoch + 1):03d}/{num_epochs} | Batch: {batch_idx:03d}/{len(train_loader)} | Gen/Dis Loss: {(generator_loss.item()):.4f}/{(discriminator_loss.item()):.4f}')
        
        with torch.no_grad():
            fake_images = model.generator_forward(fixed_noise).detach().cpu()
            log_dict['images_from_noise_per_epoch'].append(torchvision.utils.make_grid(fake_images, padding=2, normalize=True))
        
        print(f'Time elapsed: {((time.time() - start_time) / 60):.2f} min')
    
    print(f'Total Training Time: {((time.time() - start_time) / 60):.2f} min')

    return log_dict