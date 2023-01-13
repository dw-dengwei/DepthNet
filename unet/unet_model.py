""" Full assembly of the parts to form the complete network """

from argparse import RawDescriptionHelpFormatter
from .unet_parts import *
from .discriminator import Discriminator, PatchGANDiscriminator


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.encoder = UNetEncoder(n_channels)

        # self._decoder_share = UNetDecoderShare(bilinear)

        self.reconstruct_decoder = UNetDecoder(bilinear)
        self.depth_mask_decoder = UNetDecoder(bilinear)
        # self.mask_decoder = self.depth_decoder

        self.reconstruct_out = OutConv(64, n_classes)
        self.depth_mask_out = OutConv(64, 1 + 6)
        # self.mask_out = OutConv(64, 6)

        # self.recon_dis = Discriminator(in_channels=3, feature_size=1 * 751 * 501)
        # self.depth_dis = Discriminator(in_channels=1, feature_size=1 * 751 * 501)
        self.recon_dis = PatchGANDiscriminator(in_channels=3)
        self.depth_dis = PatchGANDiscriminator(in_channels=1)

    def forward(self, x, true_recon, true_depth, mask_true):
        hidden_state = self.inc(x)
        hidden_state = self.encoder(hidden_state)

        reconstruct_feature = self.reconstruct_decoder(*hidden_state)
        depth_mask_feature = self.depth_mask_decoder(*hidden_state)
        # mask_feature = self.mask_decoder(*hidden_state)

        reconstruct_map = self.reconstruct_out(reconstruct_feature)
        depth_mask_map = self.depth_mask_out(depth_mask_feature)
        # mask_map = self.mask_out(mask_feature)

        depth_map = depth_mask_map[:, 0:1, :]
        mask_map = depth_mask_map[:, 1:, :]
        # print(depth_map.size(), mask_map.size(), depth_mask_map.size())

        pred_reconstruct_dis = self.recon_dis(reconstruct_map)
        pred_depth_dis = self.depth_dis(depth_map)

        true_reconstruct_dis = self.recon_dis(true_recon)
        true_depth_dis = self.depth_dis(true_depth)

        return reconstruct_map, \
            depth_map, \
            mask_map, \
            pred_reconstruct_dis, \
            pred_depth_dis, \
            true_reconstruct_dis, \
            true_depth_dis
