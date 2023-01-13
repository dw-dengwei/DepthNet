import torch.nn as nn
import torch
import functools


class Discriminator(nn.Module):
    def __init__(self, in_channels, feature_size) -> None:
        super().__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, feature_map: torch.Tensor):
        pred = self.dis(feature_map)

        return pred

class PatchGANDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, in_channels, out_channels=8, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            in_channels (int)        -- the number of channels in input images
            out_channels (int)       -- the number of filters in the last conv layer
            n_layers (int)           -- the number of conv layers in the discriminator
            norm_layer               -- normalization layer
        """
        super(PatchGANDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(in_channels, out_channels, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(out_channels * nf_mult_prev, out_channels * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(out_channels * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(out_channels * nf_mult_prev, out_channels * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(out_channels * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(out_channels * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        sequence += [nn.Softmax()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        # print(input.shape)
        return self.model(input)