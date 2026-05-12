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

class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.beta           = beta

        # Codebook E — shape [J, D], trained alongside encoder/decoder
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
        # Kiểm tra nếu là 4D (CNN: B, C, H, W) thì chuyển về dạng chuẩn để tính toán
        is_4d = len(z.shape) == 4
        if is_4d:
            B, C, H, W = z.shape
            # Chuyển C ra cuối: [B, H, W, C]
            z = z.permute(0, 2, 3, 1).contiguous()
        else:
            B, N, C = z.shape # Giữ nguyên cho Transformer nếu cần

        z_flat = z.reshape(-1, self.embedding_dim) # [B*H*W, C]
 
        # Tính khoảng cách
        d = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)             
            + torch.sum(self.embedding.weight ** 2, dim=1)         
            - 2.0 * torch.matmul(z_flat, self.embedding.weight.t())             
        )
        indices = torch.argmin(d, dim=1)   

        z_q_flat = self.embedding(indices)

        # Cập nhật EMA (giữ nguyên logic của bạn)
        if self.training:
            with torch.no_grad():
                one_hot = torch.zeros(z_flat.size(0), self.num_embeddings, device=z.device)
                one_hot.scatter_(1, indices.reshape(-1, 1), 1)

                self.ema_cluster_size = (self.ema_decay * self.ema_cluster_size
                                        + (1 - self.ema_decay) * one_hot.sum(0))
                dw = one_hot.t() @ z_flat
                self.ema_w = (self.ema_decay * self.ema_w
                            + (1 - self.ema_decay) * dw)

                n = self.ema_cluster_size.sum()
                cluster_size = ((self.ema_cluster_size + 1e-5)
                                / (n + self.num_embeddings * 1e-5) * n)
                self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)

        # Lấy kết quả z_q
        z_q = z_flat + (z_q_flat - z_flat).detach()
        
        # Khôi phục lại shape ban đầu
        if is_4d:
            z_q = z_q.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            indices = indices.reshape(B, H, W)
        else:
            z_q = z_q.reshape(B, N, C)
            indices = indices.reshape(B, N)
 
        vq_loss = self.beta * F.mse_loss(z_flat, z_q_flat.detach())
 
        return z_q, vq_loss, indices
    
    def get_normalized_codebook(self) -> torch.Tensor:
        """
        Returns L2-normalized codebook E_norm [J, D].
 
        Used to compute the orthogonality loss:
            L_s = ||E_norm^T @ E_norm||_F^2
        which pushes basis vectors to be mutually orthogonal,
        maximising the margin between entries and improving robustness
        against feature-space perturbations (Section III-C, Hu et al.).
        """
        w = self.embedding.weight                                  
        return w / (w.norm(dim=1, keepdim=True) + 1e-8)

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

        vq_num_embeddings    = getattr(args, 'vq_num_embeddings', 1024)
        vq_beta              = getattr(args, 'vq_beta',           0.25)
        self.vq_lambda_ortho = getattr(args, 'vq_lambda_ortho',   0)
        self.vq_lambda = getattr(args, 'vq_lambda', 10)

        self.vq = VectorQuantizer_EMA(
            num_embeddings = vq_num_embeddings,
            embedding_dim  = self.channel_number[0] * 2,
            beta           = vq_beta,
        )

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

        z_q, vq_loss, indices = self.vq(feature)

        E_norm     = self.vq.get_normalized_codebook()
        ortho_loss = torch.norm(E_norm @ E_norm.t(), p='fro') ** 2

        CBR = channel_number * torch.prod(torch.tensor(feature.size()[-2:])) / torch.prod(torch.tensor(input_image.size()[1:]))

        if self.config.pass_channel:
            noisy_feature = self.channel.forward(z_q, chan_param)
        else:
            noisy_feature = z_q
        
        recon_image = self.decoder(noisy_feature)

        # Compute metrics
        mse = self.mse_loss(input_image * 255., recon_image.clamp(0., 1.) * 255.).mean()
        loss = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.)).mean()

        print(f"Recon Loss: {loss.item():.4f}, VQ Loss: { self.vq_lambda * vq_loss.item():.4f}")
        loss = loss + self.vq_lambda * vq_loss
        
        return recon_image, CBR, chan_param, mse, loss