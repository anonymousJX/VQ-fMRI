import numpy as np
from utils import *
from torch.nn import functional as F

def token_inpainting_train(epoch, train_loader, img_vqvae, token_inpaint, optimizer):
    epoch_mseloss, epoch_bceloss = [], []
    for batch_idx, (img, _) in enumerate(train_loader):
        img = img.cuda()
        img_decoded, img_encoded, img_emb, img_argmin = img_vqvae(img)
        img_argmin, true_index = img_vqvae.argmin_randreplace(img_argmin)

        dims = list(range(len(img_encoded.size())))
        shifted_shape = [img_encoded.shape[0], *
                        list(img_encoded.shape[2:]), img_encoded.shape[1]]
        result = img_vqvae.emb.weight.t().index_select(0, img_argmin.flatten()
                                    ).view(shifted_shape).permute(0, dims[-1], *dims[1:-1])
        
        decoded, mask = token_inpaint(result)
        z_q, argmin = img_vqvae.emb(decoded, weight_sg = True)
        result = img_vqvae.decode(z_q)
        mse_loss = 2*vq_mse_loss(decoded, img_emb, argmin, img_argmin)# + F.mse_loss(img_decoded, result)
        bce_loss = F.cross_entropy(mask.squeeze(), true_index.long().cuda())
        loss = mse_loss + bce_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_mseloss.append(mse_loss.item())
        epoch_bceloss.append(bce_loss.item())
    print(f"inpaint_epoch_{epoch} mse_loss:{np.mean(epoch_mseloss)} bce_loss:{np.mean(epoch_bceloss)}")
