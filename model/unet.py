import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize = 'Instance', dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if(normalize == 'Instance'):
            layers.append(nn.InstanceNorm2d(out_size))
        elif(normalize == 'Batch'):
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, normalize = 'Instance', dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size) if normalize == 'Instance' else nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, nums, hidden_channels = 128, dropout = 0.0, normalize = 'Instance'):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.normalize = normalize

        self.down = self.down_block(nums, hidden_channels)
        self.up = self.up_block(nums, hidden_channels)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(hidden_channels * 2, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def down_block(self, nums, channels):
        block = [UNetDown(self.in_channels, channels, normalize = self.normalize, dropout = self.dropout)]
        for i in range(nums - 1):
            block.append(UNetDown(channels, channels, normalize = self.normalize, dropout = self.dropout))
        return nn.Sequential(*block)

    def up_block(self, nums, channels):
        block = [UNetUp(channels, channels, normalize = self.normalize, dropout = self.dropout)]
        for i in range(nums - 2):
            block.append(UNetUp(channels * 2, channels, normalize = self.normalize, dropout = self.dropout))
        return nn.Sequential(*block)

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        feats = []
        for layer in self.down:
            x = layer(x)
            feats.append(x)
        x = feats.pop()
        for layer in self.up:
            x = layer(x, feats.pop())
        return self.final(x)

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

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h = self.num_heads, qkv = 3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim = -1) / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

