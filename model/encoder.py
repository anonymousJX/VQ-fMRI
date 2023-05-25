from torch import nn
from torch.nn import functional as F
from model.unet import UNet

class fMRI_Encoder(nn.Module):
    def __init__(self, input_size, out_size, out_channel, unet_layer_nums = 2):
        super(fMRI_Encoder, self).__init__()
        self.size = out_size
        self.channel = out_channel
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, (self.size ** 2) * self.channel)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d((self.size ** 2) * self.channel)
        self.unet = UNet(in_channels = self.channel, out_channels = self.channel, 
                                                        nums = unet_layer_nums, dropout = 0.0)

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), 0.2, self.training)
        x = F.dropout(F.relu(self.fc2(x)), 0.2, self.training)
        x = F.dropout(F.tanh(self.fc3(x)), 0.2, self.training)
        x = x.view(-1, self.channel, self.size, self.size)
        x = self.unet(x)
        return x

    def loss_function(self, result, target):
        loss = F.mse_loss(result, target)
        return loss
