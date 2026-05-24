from numpy import unique
import math
from .decoder import *
from .encoder import *
from .channel import Channel
from training.loss import Distortion
from random import choice
import torch.nn as nn
import torch
import torch.nn.functional as F

class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.beta           = beta

        # Codebook E — shape [J, D]
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        B, N, C = z.shape
        z_flat = z.reshape(-1, C)                                    # [B*N, C]

        d = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)              # [B*N, 1]
            + torch.sum(self.embedding.weight ** 2, dim=1)           # [J]
            - 2.0 * torch.matmul(z_flat, self.embedding.weight.t())  # [B*N, J]
        )

        indices  = torch.argmin(d, dim=1)    
                       
        z_q_flat = self.embedding(indices)                          

        unique_count = len(torch.unique(indices))
        print(f"Codebook usage: {unique_count}/{self.num_embeddings} entries used")

        z_q = z_flat + (z_q_flat - z_flat).detach()
        z_q = z_q.reshape(B, N, C)

        codebook_loss   = F.mse_loss(z_q_flat, z_flat.detach())
        commitment_loss = F.mse_loss(z_flat,   z_q_flat.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        indices = indices.reshape(B, N)                              
        return z_q, vq_loss, indices

    def get_normalized_codebook(self) -> torch.Tensor:
        """
        Returns L2-normalized codebook E_norm [J, D].

        Used to compute the orthogonality loss:
            L_s = ||E_norm @ E_norm.t()||_F^2 / J^2
        which pushes basis vectors to be mutually orthogonal.
        """
        w = self.embedding.weight                                   
        return w / (w.norm(dim=1, keepdim=True) + 1e-8)

class VectorQuantizer_EMA(nn.Module):
    """
    Discrete bottleneck layer (VQ-VAE style).
 
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25, ema_decay: float = 0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.beta           = beta
        self.ema_decay      = ema_decay

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
    
    def forward(self, z: torch.Tensor):
        B, N, C = z.shape
        z_flat = z.reshape(-1, C) # [B*N, C]
 
        d = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)             
            + torch.sum(self.embedding.weight ** 2, dim=1)         
            - 2.0 * torch.matmul(z_flat, self.embedding.weight.t())            
        )
        indices  = torch.argmin(d, dim=1)   

        z_q_flat = self.embedding(indices)

        if self.training:
            with torch.no_grad():
                one_hot = torch.zeros(B * N, self.num_embeddings, device=z.device)
                one_hot.scatter_(1, indices.reshape(-1, 1), 1)

                self.ema_cluster_size = (self.ema_decay * self.ema_cluster_size
                                        + (1 - self.ema_decay) * one_hot.sum(0))
                dw = one_hot.t() @ z_flat
                self.ema_w = (self.ema_decay * self.ema_w
                            + (1 - self.ema_decay) * dw)

                n = self.ema_cluster_size.sum()
                cluster_size = ((self.ema_cluster_size + 1e-5)
                                / (n + self.num_embeddings * 1e-5) * n)
                # e_j = m_j / cluster_size_j
                self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)

                # Dead entry reinitialization
                dead = (self.ema_cluster_size < 1.0).nonzero(as_tuple=True)[0]
                if len(dead) > 0:
                    print(f"Reinitializing {len(dead)} dead entries")
                    random_idx = torch.randint(0, B * N, (len(dead),), device=z.device)
                    self.embedding.weight.data[dead] = z_flat[random_idx].detach()
                    self.ema_w[dead] = z_flat[random_idx].detach()
                    self.ema_cluster_size[dead] = 1.0

        unique = torch.unique(indices)
        # print(f"Codebook usage: {len(unique)}/{self.num_embeddings} entries used")

        z_q = z_flat + (z_q_flat - z_flat).detach()
        z_q = z_q.reshape(B, N, C)
 
        # codebook_loss   = F.mse_loss(z_q_flat, z_flat.detach())
        # commitment_loss = F.mse_loss(z_flat,   z_q_flat.detach())
        # vq_loss = codebook_loss + self.beta * commitment_loss
        vq_loss = self.beta * F.mse_loss(z_flat, z_q_flat.detach())
 
        indices = indices.reshape(B, N)                             
        return z_q, vq_loss, indices
    
    def get_normalized_codebook(self) -> torch.Tensor:
        w = self.embedding.weight                                  
        return w / (w.norm(dim=1, keepdim=True) + 1e-8) # L2 normalized


class SwinJSCC(nn.Module):
    """
    SwinJSCC: Joint Source-Channel Coding framework using Swin Transformer.

    Attributes:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        channel (Channel): Channel simulator (AWGN, Rayleigh, etc.).
        distortion_loss (Distortion): Distortion metric.
    """
    
    def __init__(self, args, config):
        super(SwinJSCC, self).__init__()
        self.config = config
        self.model = args.model
        self.pass_channel = config.pass_channel
        self.downsample = config.downsample

        self.multiple_snr = [int(s) for s in args.multiple_snr.split(",")]
        self.channel_number = [int(c) for c in args.C.split(",")]

        self.encoder = create_encoder(**config.encoder_kwargs)
        self.decoder = create_decoder(**config.decoder_kwargs)

        self.channel = Channel(args, config)
        self.distortion_loss = Distortion(args)
        self.mse_loss = nn.MSELoss(reduction='none')

        self.H = self.W = 0

        if self.model == 'SwinJSCC_vq-vae':
            C = self.channel_number[0]
 
            vq_num_embeddings    = getattr(args, 'vq_num_embeddings', 512)
            vq_beta              = getattr(args, 'vq_beta',           0.25)
            self.vq_lambda_ortho = getattr(args, 'vq_lambda_ortho',   1e-4)
            self.vq_lambda = getattr(args, 'vq_lambda', 10)
 
            self.vq = VectorQuantizer_EMA(
                num_embeddings = vq_num_embeddings,
                embedding_dim  = C,
                beta           = vq_beta,
            )

            if config.logger:
                config.logger.info(
                    f'[VQ-VAE] Codebook: J={vq_num_embeddings}, '
                    f'dim={C}, beta={vq_beta}, '
                    f'lambda_ortho={self.vq_lambda_ortho}, lambda={self.vq_lambda}'
                )

        # Logging
        if config.logger:
            logger = config.logger
            logger.info("=== SwinJSCC Configuration ===")
            logger.info(f"Encoder: {config.encoder_kwargs}")
            logger.info(f"Decoder: {config.decoder_kwargs}")

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss
    
    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature
    
    def _update_resolution(self, H, W):
        """Update encoder/decoder internal resolution when input size changes."""
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H, self.W = H, W
    
    def forward(self, input_image, given_SNR=None, given_rate=None):
        B, _, H, W = input_image.shape

        self._update_resolution(H, W)

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        if given_rate is None:
            channel_number = choice(self.channel_number)
        else:
            channel_number = given_rate

        # Encode
        if self.model in ['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA']:
            feature, _ = self.encoder(input_image, chan_param, channel_number, self.model)
            CBR = channel_number / (2 * 3 * 2 ** (self.downsample * 2))

            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param)
            else:
                noisy_feature = feature

            # Decode
            recon_image = self.decoder(noisy_feature, chan_param, self.model)

        elif self.model in ['SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA']:
            feature, mask = self.encoder(input_image, chan_param, channel_number, self.model)
            CBR = channel_number / (2 * 3 * 2 ** (self.downsample * 2))
            avg_pwr = torch.sum(feature ** 2) / mask.sum()

            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param, avg_pwr)
            else:
                noisy_feature = feature
                
            noisy_feature = noisy_feature * mask

            # Decode
            recon_image = self.decoder(noisy_feature, chan_param, self.model)

        elif self.model == 'SwinJSCC_vq-vae':
            feature, _ = self.encoder(
                input_image, chan_param, channel_number, self.model
            )

            CBR = channel_number / (2 * 3 * 2 ** (self.downsample * 2))
            bit_per_index = int(math.log2(self.vq.num_embeddings))

            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param)
            else:
                noisy_feature = feature

            z_q, vq_loss, indices = self.vq(noisy_feature)           

            E_norm     = self.vq.get_normalized_codebook()
            ortho_loss = torch.norm(E_norm @ E_norm.t(), p='fro') ** 2
          
            # Decode
            recon_image = self.decoder(z_q, chan_param, 'SwinJSCC_w/o_SAandRA')

        else:
            raise ValueError(f"Unknown model variant: {self.model}")

        mse = self.mse_loss(input_image * 255., recon_image.clamp(0., 1.) * 255.).mean()
        loss = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.)).mean()

        if self.model == 'SwinJSCC_vq-vae':
            J = self.vq.num_embeddings
            ortho_loss_norm = ortho_loss / (J ** 2)

            # print(f"Recon Loss: {loss.item():.4f}, VQ Loss: { self.vq_lambda * vq_loss.item():.4f}, Ortho Loss: {self.vq_lambda_ortho * ortho_loss_norm.item():.6f} ")
            loss = loss + self.vq_lambda * vq_loss + self.vq_lambda_ortho * ortho_loss_norm

        return recon_image, CBR, chan_param, mse, loss