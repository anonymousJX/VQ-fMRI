import os
import shutil
import torch
from PIL import Image
import logging.config
from datetime import datetime
import torch.utils.data
from model.vqvae import VQ_VAE
from torch.nn import functional as F
from einops import rearrange
from torchvision.utils import save_image

def vq_mse_loss(fmri, img, fmri_argmin, img_argmin):
    index = rearrange(~(fmri_argmin == img_argmin), 'b h w -> (b h w)')
    fmri = rearrange(fmri, 'b d h w -> (b h w) d')[index]
    img = rearrange(img, 'b d h w -> (b h w) d')[index]
    return F.mse_loss(fmri, img)

def create_vqvae(image_size, d, k, down_f, num_channels, epoch, name = 'model'):
    vqvae_dir = get_model_path('VQVAE', image_size, d, k, epoch, name)
    vqvae = VQ_VAE(d = d, k = k, down_f = down_f, num_channels = num_channels).cuda()
    vqvae.load_state_dict(torch.load(vqvae_dir, map_location='cuda:0'))
    return vqvae

def create_save_path(image_size, d, k):
    return f's{image_size}-d{d}-k{k}'

def get_model_path(type, image_size, d, k, epoch, name = 'model'):
    save_path = create_save_path(image_size, d, k)
    return f'./results/{type}/{save_path}/checkpoints/{name}_{epoch}.pth'

def setup_logging_from_args(results_dir = './results', save_name = ''):
    """
    Calls setup_logging, exports args and creates a ResultsLog class.
    Can resume training/logging if args.resume is set
    """
    if save_name == '':
        save_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(results_dir, save_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok = True)
    log_file = os.path.join(save_path, 'log.txt')
    setup_logging(log_file)
    return save_path

def setup_logging(log_file='log.txt', resume = False):
    """
    Setup logging configuration
    """
    if os.path.isfile(log_file) and resume:
        file_mode = 'a'
    else:
        file_mode = 'w'

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=file_mode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def save_checkpoint(model, epoch, save_path, save_name = 'model'):
    os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(save_path, 'checkpoints', f'{save_name}_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)

def save_reconstructed_images(epoch, save_path, name, images, normalize = False, nrow = None):
    n = min(images[0].size(0), 20)
    comparison = torch.cat([images[i][:n] for i in range(len(images))])
    if(nrow is not None):
        n = nrow
    save_image(comparison.cpu(), os.path.join(save_path, name + '_' + str(epoch) + '.png'), 
                                                                     nrow = n, normalize = normalize)
    
def save_RGB_images(epoch, targets, outputs, save_path, name = 'RGB'):
    targets = targets.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    size = targets.shape[0]
    for i in range(size):
        target = Image.fromarray(targets[i].squeeze())
        target = target.convert('RGB')
        output = Image.fromarray(outputs[i].squeeze() * 255.0)
        output = output.convert('RGB')
        result = Image.new('RGB', (28 * 2, 28))
        result.paste(target, (0, 0, 28, 28))
        result.paste(output, (28, 0, 56, 28))
        result = result.resize((512, 256), Image.Resampling.LANCZOS)
        result.save(f'{save_path}/{name}_{epoch}_{i}.jpg')