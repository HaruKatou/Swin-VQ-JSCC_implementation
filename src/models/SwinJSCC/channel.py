import torch.nn as nn
import numpy as np
import os
import torch
import time

class Channel(nn.Module):
    """
    Currently the channel model is either error free, erasure channel,
    rayleigh channel or the AWGN channel.

    Simulated physical channel: y = hx + n
    where h is the channel gain, n is the additive noise.
    """

    def __init__(self, args, config):
        super(Channel, self).__init__()
        self.config = config
        self.channel_type = args.channel_type
        self.device = config.device
        self.h = torch.sqrt(torch.randn(1) ** 2
                            + torch.randn(1) ** 2) / 1.414
        
        if config.logger:
            config.logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                args.channel_type, args.multiple_snr))
            
    def AWGN(self, x, std):
        """ 
        Additive White Gaussian Noise Channel 
        y = x + n, n~N(0,std)
        """
        device = x.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(x), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(x), device=device)
        noise = noise_real + 1j * noise_imag
        return x + noise
    
    def Rayleigh(self, x, std):
        """ 
        Rayleigh Fading Channel 
        y = hx + n, h~Rayleigh(1), n~N(0,std)
        """
        device = x.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=x.shape)
        noise_imag = torch.normal(mean=0.0, std=std, size=x.shape)
        noise = noise_real + 1j * noise_imag

        h = torch.sqrt(torch.normal(mean=0.0, std=1, size=x.shape) ** 2
                       + torch.normal(mean=0.0, std=1, size=x.shape) ** 2) / np.sqrt(2)
        
        if self.config.CUDA:
            noise = noise.to(self.device)
            h = h.to(self.device)

        return h * x + noise
    
    def complex_normalize(self, x, target_power):
        if not torch.is_tensor(target_power):
            target_power = torch.tensor(target_power, device=self.device)

        power = torch.mean(x ** 2) * 2
        scale = torch.sqrt(target_power) / torch.sqrt(power)

        return x * scale, power
    
    def complex_forward(self, channel_in, snr_db):
        if self.channel_type == 'None':
            return channel_in
        
        snr_linear = 10 ** (snr_db / 10) 
        sigma = np.sqrt(1.0 / (2 * snr_linear))

        if self.channel_type == 'awgn':
            return self.AWGN(channel_in, sigma)
        
        elif self.channel_type == 'Rayleigh':
            return self.Rayleigh(channel_in, sigma)
        
        else:
            raise NotImplementedError('Channel type {} not implemented.'.format(self.channel_type))
        
    def forward(self, input, snr_db, avg_power=False):
        if avg_power:
            power = 1
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_power * 2)
        else:
            channel_tx, power = self.complex_normalize(input, target_power=1)
        
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j

        channel_output = self.complex_forward(channel_in, snr_db)
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_shape)

        if self.channel_type == 'awgn':
            noise = (channel_output - channel_tx).detach()
            noise.requires_grad = False
            channel_tx = channel_tx + noise
            if avg_power:
                return channel_tx * torch.sqrt(avg_power * 2)
            else:
                return channel_tx * torch.sqrt(power)
        elif self.channel_type == 'Rayleigh':
            if avg_power:
                return channel_output * torch.sqrt(avg_power * 2)
            else:
                return channel_output * torch.sqrt(power)
            
    def noiseless_forward(self, x):
        real, img = torch.chunk(x, 2, dim=1)
        x_complex = torch.complex(real, img)
        x_complex, _ = self.complex_normalize(x_complex)
        return torch.cat([x_complex.real, x_complex.imag], dim=1)
        
