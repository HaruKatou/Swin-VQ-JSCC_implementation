import torch
import torch.nn as nn
from models.SwinJSCC.channel import Channel
from training.loss import Distortion
from random import choice
from .encoder import DJSCC_Encoder
from .decoder import DJSCC_Decoder
from random import choice
import torch.nn as nn
import torch
import torch.nn.functional as F

class DJSCC_CNN(nn.Module):
    def __init__(self, args, config):
        super(DJSCC_CNN, self).__init__()
        self.config = config
        self.channel = Channel(args, config)

        self.multiple_snr = [int(s) for s in args.multiple_snr.split(",")]
        self.channel_number = [int(c) for c in args.C.split(",")]
    
        self.encoder = DJSCC_Encoder(config, self.channel_number[0])
        self.decoder = DJSCC_Decoder(config, self.channel_number[0])

        self.distortion_loss = Distortion(args)
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, input_image, given_SNR=None, given_rate=None):
        B, _, H, W = input_image.shape

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        if given_rate is None:
            channel_number = self.channel_number[0]
        else:
            channel_number = given_rate

        feature = self.encoder(input_image)

        spatial_ratio = (input_image.size(-2) * input_image.size(-1)) / (feature.size(-2) * feature.size(-1))
        CBR = feature.size(1) / (2 * 3 * spatial_ratio)

        if self.config.pass_channel:
            noisy_feature = self.channel.forward(feature, chan_param)
        else:
            noisy_feature = feature
        
        recon_image = self.decoder(noisy_feature)

        # Compute metrics
        mse = self.mse_loss(input_image * 255., recon_image.clamp(0., 1.) * 255.).mean()
        loss = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.)).mean()
        
        return recon_image, CBR, chan_param, mse, loss