import torch
import torch.nn as nn
from models.SwinJSCC.channel import Channel
from training.loss import Distortion
from random import choice
from .encoder import DJSCC_Encoder
from .decoder import DJSCC_Decoder

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
            channel_number = choice(self.channel_number)
        else:
            channel_number = given_rate

        feature = self.encoder(input_image)

        CBR = channel_number * torch.prod(torch.tensor(feature.size()[-2:])) / torch.prod(torch.tensor(input_image.size()[1:]))

        # print("Compression Ratio: ", CBR.item())

        if self.config.pass_channel:
            noisy_feature = self.channel.forward(feature, chan_param)
        else:
            noisy_feature = feature
        
        recon_image = self.decoder(noisy_feature)
        
        mse = torch.mean((input_image - recon_image) ** 2)

        # Compute metrics
        mse = self.mse_loss(input_image * 255., recon_image.clamp(0., 1.) * 255.).mean()
        loss = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.)).mean()
        
        return recon_image, CBR, chan_param, mse, loss