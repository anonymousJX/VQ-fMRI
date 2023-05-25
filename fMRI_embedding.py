import logging
import numpy as np
import torch.utils.data
from torch import optim
from model.encoder import fMRI_Encoder
from torch.nn import functional as F
from utils import *

def fmri2image_train(epoch, img_vqvae, fmri_encoder, optimizer, train_loader, save_path):
    img_vqvae.eval()
    fmri_encoder.train()
    epoch_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.cuda()
        target = target.cuda()
        output = fmri_encoder(data)
        fmri_decoded, fmri_encoded, fmri_emb, fmri_argmin = img_vqvae(output)
        target_decoded, target_encoded, target_emb, target_argmin = img_vqvae(target)
        train_loss = vq_mse_loss(fmri_encoded, target_emb, fmri_argmin, target_argmin) + F.mse_loss(output, target)
        train_loss.backward()
        optimizer.step()

        epoch_losses.append(train_loss.item())
    logging.info('Train Epoch [%03d]  loss: %.4f' % 
                (epoch, np.mean(epoch_losses)))
    
    save_reconstructed_images(epoch, save_path, 'reconstruction_train', [target, fmri_decoded, target_decoded])
    if(epoch % 10 == 0):
        save_checkpoint(fmri_encoder, epoch, save_path, 'encoder')
        

@torch.no_grad()
def fmri2image_test(epoch, img_vqvae, fmri_encoder, test_loader, save_path):
    img_vqvae.eval()
    fmri_encoder.eval()
    epoch_losses = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.cuda()
        target = target.cuda()
        output = fmri_encoder(data)
        fmri_decoded, fmri_encoded, fmri_emb, fmri_argmin = img_vqvae(output)
        target_decoded, target_encoded, target_emb, target_argmin = img_vqvae(target)
        test_loss = vq_mse_loss(fmri_encoded, target_emb, fmri_argmin, target_argmin) + F.mse_loss(output, target)
        epoch_losses.append(test_loss.item())
    logging.info('Test  Epoch [%03d]  loss: %.4f'% (epoch, np.mean(epoch_losses)))
    save_reconstructed_images(epoch, save_path, 'reconstruction_test', [target, fmri_decoded])


def fmri2embedding_train(epoch, img_vqvae, fmri_vqvae, fmri_encoder, optimizer, train_loader, save_path):
    img_vqvae.eval()
    fmri_encoder.train()
    fmri_vqvae.train()
    epoch_mse_losses, epoch_vq_losses = [], []

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        output = fmri_encoder(data)
        target_decoded, target_encoded, target_emb, target_argmin = img_vqvae(target)

        fmri_emb, fmri_argmin = img_vqvae.emb(output)
        fmri_decoded = fmri_vqvae.decoder(fmri_emb)

        vq_loss = vq_mse_loss(output, target_emb, fmri_argmin, target_argmin)
        mse_loss = F.mse_loss(fmri_decoded, target_decoded)
        train_loss = vq_loss + mse_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        epoch_mse_losses.append(mse_loss.item())
        epoch_vq_losses.append(vq_loss.item())

    logging.info('Train Epoch [%03d]  mse loss: %.4f  vq loss: %.4f' % 
                (epoch, np.mean(epoch_mse_losses), np.mean(epoch_vq_losses)))

    if(epoch % 10 == 0):
        save_checkpoint(fmri_encoder, epoch, save_path, 'encoder')
        save_checkpoint(fmri_vqvae, epoch, save_path, 'vqvae')


@torch.no_grad()
def fmri2embedding_test(epoch, img_vqvae, fmri_vqvae, fmri_encoder, test_loader):
    img_vqvae.eval()
    fmri_encoder.eval()
    fmri_vqvae.eval()
    epoch_mse_losses, epoch_vq_losses = [], []
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.cuda()
        target = target.cuda()
        output = fmri_encoder(data)

        target_decoded, target_encoded, target_emb, target_argmin = img_vqvae(target)
        fmri_emb, fmri_argmin = img_vqvae.emb(output, weight_sg = True)
        fmri_decoded = fmri_vqvae.decoder(fmri_emb)

        vq_loss = vq_mse_loss(output, target_emb, fmri_argmin, target_argmin)
        mse_loss = F.mse_loss(fmri_decoded, target_decoded)

        epoch_mse_losses.append(mse_loss.item())
        epoch_vq_losses.append(vq_loss.item())

    logging.info('Test  Epoch [%03d]  mse loss: %.4f  vq loss: %.4f'% 
                        (epoch, np.mean(epoch_mse_losses), np.mean(epoch_vq_losses)))
    

def fMRI_decoding_train(fMRI_size, image_size, d, k, down_f, num_channels, train_loader, test_loader, epochs, 
                  decoding_target = 'image', save_dir = './results/fMRI2Image/'):
    
    save_path = setup_logging_from_args(save_dir, create_save_path(image_size, d, k))

    img_vqvae = create_vqvae(image_size, d, k, down_f, num_channels, 60)
    fmri_vqvae = create_vqvae(image_size, d, k, down_f, num_channels, 60)

    if(decoding_target == 'embedding'):
        fmri_encoder = fMRI_Encoder(input_size = fMRI_size, out_size = image_size // 8, out_channel = d).cuda()
        optimizer = optim.Adam(fmri_encoder.parameters(), lr = 2e-4)
        for epoch in range(1, epochs + 1):
            fmri2embedding_train(epoch, img_vqvae, fmri_vqvae, fmri_encoder, optimizer, train_loader, save_path)
            fmri2embedding_test(epoch, img_vqvae, fmri_vqvae, fmri_encoder, test_loader)

    elif(decoding_target == 'image'):
        fmri_encoder = fMRI_Encoder(input_size = fMRI_size, out_size = image_size, out_channel = num_channels).cuda()
        optimizer = optim.Adam(fmri_encoder.parameters(), lr = 1e-4)
        for epoch in range(1, epochs + 1):
            fmri2image_train(epoch, img_vqvae, fmri_encoder, optimizer, train_loader, save_path)
            fmri2image_test(epoch, img_vqvae, fmri_encoder, test_loader, save_path)
    else:
        raise NotImplementedError