import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
from .sh import ls, sft, isft

def define_network(init_type='normal', init_gain=1., gpu_ids=[]):
    """Create the model

    Get the network architecture
    """
    net = None
    net = NFONet(c_in=1, n_out=1)

    return init_net(net, init_type, init_gain, gpu_ids)


class DiscreteParameter(nn.Module):
    def __init__(self, values=[4, 6, 8], initial_value=2.0):
        super(DiscreteParameter, self).__init__()
        self.values = values
        self.lmax_index = nn.Parameter(torch.tensor(initial_value, dtype=torch.float16))

    def forward(self):
        lmax_index = torch.clamp(self.lmax_index, min=0, max=2)
        lmax_index = torch.ceil(lmax_index)
        return lmax_index
    

class NFO_GE(torch.nn.Module):
    def __init__(self, c_in: int, c_out: int, l_max: int = 8):
        super().__init__()
        self.l_max = l_max
        self.expansion_mask = self._create_expansion_mask()

        # Add residual connection capability
        self.residual = (c_in == c_out)

        self.weights = torch.nn.Parameter(
            torch.empty(c_out, c_in, (l_max // 2) + 1)
        )

        torch.nn.init.kaiming_uniform_(self.weights, a=0.01)

    def _create_expansion_mask(self):
        mask = []
        for l in range(0, self.l_max + 1, 2):
            mask += [l // 2] * (2 * l + 1)
        return torch.tensor(mask)

    def forward(self, x):
        b,c,lm = x.shape
        expanded_weights = self.weights[:, :, self.expansion_mask]
        out = torch.einsum('bci,oci->boi', x, expanded_weights)
        return out + x if self.residual else out  # Residual connection


class NFONet(torch.nn.Module):
    def __init__(self, c_in: int, n_out: int):
        super().__init__()
        self.register_buffer("ls", ls)
        self.register_buffer("sft", sft)
        self.register_buffer("isft", isft)

        # encoder
        self.resi1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=729, out_channels=1, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.SiLU()
        )
        self.encoder_layer = NFO_GE(729, 512)

        # layer1
        self.resi2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512+1, out_channels=1, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.SiLU()
        )
        self.nfo1_layer = NFO_GE(512+1, 256)

        # layer2
        self.resi3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256+1, out_channels=1, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.SiLU()
        )
        self.nfo2_layer = NFO_GE(256+1, 128)

        # layer3
        self.resi4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128+1, out_channels=1, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.SiLU()
        )
        self.nfo3_layer = NFO_GE(128+1, 64)

        # layer4
        self.resi5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64+1, out_channels=1, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.SiLU()
        )
        self.nfo4_layer = NFO_GE(64+1, 32)

        # outlayer
        self.resi6 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32+2+1, out_channels=1, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.SiLU()
        )
        self.decoder_layer = NFO_GE(32+2+1, 8)
        self.decoder_last = NFO_GE(8+1, 1)


    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.resi1(x.unsqueeze(-1)) #s
        x = self.encoder_layer(x) #sh
        x = self.isft @ x.unsqueeze(-1)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.sft @ x
        x = torch.cat([x.squeeze(-1), residual.squeeze(-1)], dim=1)
        return x, residual

    def NFO1(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.resi2(x.unsqueeze(-1)) #s
        x = self.nfo1_layer(x) #sh
        x = self.isft @ x.unsqueeze(-1)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.sft @ x
        x = torch.cat([x.squeeze(-1), residual.squeeze(-1)], dim=1)
        return x

    def NFO2(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.resi3(x.unsqueeze(-1)) #s
        x = self.nfo2_layer(x) #sh
        x = self.isft @ x.unsqueeze(-1)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.sft @ x
        x = torch.cat([x.squeeze(-1), residual.squeeze(-1)], dim=1)
        return x

    def NFO3(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.resi4(x.unsqueeze(-1)) #s
        x = self.nfo3_layer(x) #sh
        x = self.isft @ x.unsqueeze(-1)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.sft @ x
        x = torch.cat([x.squeeze(-1), residual.squeeze(-1)], dim=1)
        return x

    def NFO4(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.resi5(x.unsqueeze(-1)) #s
        x = self.nfo4_layer(x) #sh
        x = self.isft @ x.unsqueeze(-1)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.sft @ x
        x = torch.cat([x.squeeze(-1), residual.squeeze(-1)], dim=1)
        return x
    
    def decoder0(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.resi6(x.unsqueeze(-1)) #s
        x = self.decoder_layer(x) #sh
        x = self.isft @ x.unsqueeze(-1)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.sft @ x
        x = torch.cat([x.squeeze(-1), residual.squeeze(-1)], dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batchsize, c, h, w, d = x.shape
        residual0 = x[:, :, 4, 4, 4]
        x = x.view(batchsize, h * w * d, c)

        # sfno1
        x, residual1 = self.encoder(x)

        x = self.NFO1(x)
        x = self.NFO2(x)
        x = self.NFO3(x)
        x = self.NFO4(x)

        x = torch.cat([x, residual1.squeeze(-1)], dim=1)
        x = torch.cat([x, residual0.unsqueeze(1)], dim=1)
        x = self.decoder0(x)
        odfs_sh = self.decoder_last(x).squeeze(1)

        return odfs_sh


def init_net(net, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        net = net.to(torch.device("cuda"))
    else:
        net = net.to(torch.device("cpu"))
    
    init_weights(net, init_type, init_gain=init_gain, activation='leaky_relu')
    return net


def init_weights(net, init_type='xavier', init_gain=1.0, activation='relu'):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain(activation))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity=activation)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


if __name__ == "__main__":
    from torchinfo import summary
    input_size = (256, 45, 9, 9, 9)
    model = NFONet(c_in=1, n_out=1)
    summary(model, input_size=input_size)
