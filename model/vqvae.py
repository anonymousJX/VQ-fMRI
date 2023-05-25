import numpy as np
import logging
import torch
from math import log
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torch.autograd import Function

class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)
                                      ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None

def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)

class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)

class NearestEmbedEMA(nn.Module):
    def __init__(self, n_emb, emb_dim, decay=0.99, eps=1e-5):
        super(NearestEmbedEMA, self).__init__()
        self.decay = decay
        self.eps = eps
        self.embeddings_dim = emb_dim
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        embed = torch.rand(emb_dim, n_emb)
        self.register_buffer('weight', embed)
        self.register_buffer('cluster_size', torch.zeros(n_emb))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, x):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """

        dims = list(range(len(x.size())))
        x_expanded = x.unsqueeze(-1)
        num_arbitrary_dims = len(dims) - 2
        if num_arbitrary_dims:
            emb_expanded = self.weight.view(
                self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
        else:
            emb_expanded = self.weight

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [x.shape[0], *list(x.shape[2:]), x.shape[1]]
        result = self.weight.t().index_select(
            0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1])

        if self.training:
            latent_indices = torch.arange(self.n_emb).type_as(argmin)
            emb_onehot = (argmin.view(-1, 1) ==
                          latent_indices.view(1, -1)).type_as(x.data)
            n_idx_choice = emb_onehot.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            flatten = x.permute(
                1, 0, *dims[-2:]).contiguous().view(x.shape[1], -1)

            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, n_idx_choice
            )
            embed_sum = flatten @ emb_onehot
            self.embed_avg.data.mul_(self.decay).add_(
                1 - self.decay, embed_sum)

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) /
                (n + self.n_emb * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.weight.data.copy_(embed_normalized)

        return result, argmin

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

class Encoder(nn.Module):
    def __init__(self, i_channels, d, down_f, bn = True):
        super(Encoder, self).__init__()
        layers = [nn.Conv2d(i_channels, d, kernel_size = 4, stride = 2, padding = 1),
                  nn.BatchNorm2d(d),
                  nn.ReLU(inplace = True)]
        for _ in range(int(log(down_f, 2) - 1)):
            layers.append(nn.Conv2d(d, d, kernel_size = 4, stride = 2, padding = 1))
            layers.append(nn.BatchNorm2d(d))
            layers.append(nn.ReLU(inplace = True))
        layers.append(ResBlock(d, d, bn=bn))
        layers.append(nn.BatchNorm2d(d))
        layers.append(ResBlock(d, d, bn = bn))
        layers.append(nn.BatchNorm2d(d))

        self.model = nn.Sequential(*layers)
        self.model[-1].weight.detach().fill_(1 / 40)

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, o_channels, d, up_f):
        super(Decoder, self).__init__()
        layers = [ResBlock(d, d),
                  nn.BatchNorm2d(d),
                  ResBlock(d, d),]
        for _ in range(int(log(up_f, 2) - 1)):
            layers.append(nn.ConvTranspose2d(d, d, kernel_size = 4, stride = 2, padding = 1))
            layers.append(nn.BatchNorm2d(d))
            layers.append(nn.ReLU(inplace = True))
        layers.append(nn.ConvTranspose2d(d, o_channels, kernel_size = 4, stride = 2, padding = 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VQ_VAE(nn.Module):
    def __init__(self, d, k, down_f, vq_coef=1, commit_coef=0.5, num_channels=3, emb_stop=False, **kwargs):
        super(VQ_VAE, self).__init__()
        self.d = d
        self.k = k
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0
        self.stop_emb = emb_stop

        self.encoder = Encoder(num_channels, d, down_f)
        self.decoder = Decoder(num_channels, d, down_f)
        self.emb = NearestEmbed(k, d)
        self.emb_t = NearestEmbed(k, d)

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)


        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder(x))

    def mask_replace(self, z_e, num = None):
        b, d, h, w = z_e.shape
        if(num is None):
            num = torch.randint(int(h * w * 0.5), int(h * w * 0.75), [1])
        z_e = rearrange(z_e, 'b d h w -> b d (h w)')
        random_index = torch.randperm(h * w)[:num]
        mask_labels = torch.ones((b, (h * w))
        mask_labels[:, random_index] = 0
        sample = torch.randn(b, self.d, num, requires_grad = False).cuda()
        z_e[:, :, random_index] = sample
        z_e = rearrange(z_e, 'b d (h w) -> b d h w', h = h, w = w)
        mask_labels = rearrange(mask_labels, 'b (h w) -> b h w', h = h, w = w)
        return z_e, mask_labels
    
    def mask_add(self, z_e, num = None):
        batch_size, d, h, w = z_e.shape
        if(num is None):
            num = torch.randint(int(h * w * 0.5), int(h * w * 0.75), [1])
        z_e = rearrange(z_e, 'b d h w -> b d (h w)')
        random_index = torch.randperm(h * w)[:num]
        sample = torch.randn(batch_size, self.d, num, requires_grad = False).cuda()
        zeros = torch.zeros_like(z_e)
        zeros[:, :, random_index] = sample
        z_e = zeros + z_e
        z_e = rearrange(z_e, 'b d (h w) -> b d h w', h = h, w = w)
        return z_e

    def argmin_randreplace(self, argmin):
        b, h, w = argmin.shape
        num = torch.randint(int(h * w * 0.5), int(h * w * 0.75), [1])
        argmin = rearrange(argmin, 'b h w -> b (h w)')
        random_index = torch.randperm(h * w)[:num]
        mask_labels = torch.ones((b, (h * w))
        mask_labels[:, random_index] = 0
        
        argmin[:, random_index] = torch.randint(0, self.k, [num]).cuda()
        argmin = rearrange(argmin, 'b (h w) -> b h w', h = h, w = w)
        mask_labels = rearrange(mask_labels, 'b (h w) -> b h w', h = h, w = w)
        return argmin, mask_labels

    def forward(self, x, mask = False, num = None):
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        if(mask):
            z_e = self.mask_add(z_e, num)
        z_q, argmin = self.emb(z_e, weight_sg = True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, emb, argmin

    def sample(self, size):
        sample = torch.randn(size, self.d, self.f, self.f, requires_grad = False).cuda()
        emb, argmin = self.emb(sample)
        return self.decode(emb.view(size, self.d, self.f, self.f)), emb, argmin

    def loss_function(self, x, recon_x, z_e, emb, argmin):
        self.mse = F.mse_loss(recon_x, x)
        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
        self.commit_loss = torch.mean(torch.norm((emb.detach() - z_e)**2, 2, 1))
        if(self.stop_emb):
            return self.mse
        return self.mse + self.vq_coef*self.vq_loss + self.commit_coef*self.commit_loss

    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss, 'commitment': self.commit_loss}

    def print_atom_hist(self, argmin):
        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)
