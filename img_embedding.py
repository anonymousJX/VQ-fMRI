import time
import logging
from utils import *
from torch import optim
import torch.utils.data
from torch.nn import functional as F

def vqvae_train(epoch, model, train_loader, optimizer, save_path, 
                    max_train_samples = None, log_interval = 10, mask = False):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data, mask)
        loss = model.loss_function(data, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])

        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            logging.info('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                         ' {time:3.2f}   {loss}'
                         .format(epoch=epoch, batch=batch_idx * len(data), total_batch=len(train_loader) * len(data),
                                 percent=int(100. * batch_idx / len(train_loader)),
                                 time=time.time() - start_time,
                                 loss=loss_string))
            start_time = time.time()
            for key in latest_losses:
                losses[key + '_train'] = 0

        
        if max_train_samples != None and batch_idx * len(data) > max_train_samples:
            break
    for key in epoch_losses:
        epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
    loss_string = '\t'.join(['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    save_reconstructed_images(epoch, save_path, 'reconstruction_train', [data, outputs[0]])
    model.print_atom_hist(outputs[3])

def vqvae_test(epoch, model, test_loader, save_path, max_test_samples = None, mask = False):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.cuda()          
            outputs = model(data, mask)
            model.loss_function(data, *outputs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
            if max_test_samples != None and i * len(data) > max_test_samples:
                break
    for key in losses:
        losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
    loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
    logging.info('====> Test set losses: {}'.format(loss_string))
    save_reconstructed_images(epoch, save_path, 'reconstruction_test', [data, outputs[0]])
    save_checkpoint(model, epoch, save_path)

def image_embedding_train(image_size, d, k, down_f, num_channels, train_loader, test_loader, epoch_num,
                             save_dir = './results/VQVAE/', max_train_samples = None):

    save_path = setup_logging_from_args(save_dir, create_save_path(image_size, d, k))
    vqvae = VQ_VAE(d = d, k = k, down_f = down_f, num_channels = num_channels).cuda()
    optimizer = optim.Adam(vqvae.parameters(), lr = 2e-4)
    for epoch in range(1, epoch_num + 1):
        vqvae_train(epoch, vqvae, train_loader, optimizer, save_path, max_train_samples)
        vqvae_test(epoch, vqvae, test_loader, save_path)
