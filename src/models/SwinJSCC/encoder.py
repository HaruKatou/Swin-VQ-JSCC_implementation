from .modules import *
import torch

class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                 downsample=None):

        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=out_dim,
                                 input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2),
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        
        # Patch Merging Layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    
    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x
    
    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
    
    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.downsample is not None:
            self.downsample.input_resolution = (H * 2, W * 2)

class SwinJSCC_Encoder(nn.Module):
    def __init__(self, model, img_size, patch_size, in_chans,
                 embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 bottleneck_dim=16):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patches_resolution = img_size
        self.H = img_size[0] // (2 ** self.num_layers)
        self.W = img_size[1] // (2 ** self.num_layers)
        self.hidden_dim = int(self.embed_dims[len(embed_dims)-1] * 1.5)
        self.layer_num = layer_num = 7

        self.patch_embed = PatchEmbed(img_size, self.patch_size, self.in_chans, embed_dims[0])

        # Build Encoder Layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else 3,
                out_dim=int(embed_dims[i_layer]),
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                    self.patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer != 0 else None
            )
            print("Encoder ", layer.extra_repr())
            self.layers.append(layer)

        self.norm = norm_layer(embed_dims[-1])
        if C != None:
            self.head_list = nn.Linear(embed_dims[-1], C)

        if model not in ['SwinJSCC_w/o_SAandRA', 'SwinJSCC_vq-vae']:
            self._build_modulators(model)

        self.apply(self._init_weights)

    def _build_modulators(self, model):
        # Channel ModNet: SNR Adaptation (SA)
        self.sa_sm_list = nn.ModuleList()
        self.sa_bm_list = nn.ModuleList()
        if "SA" in model:
            self.sa_sm_list.append(nn.Linear(self.embed_dims[-1], self.hidden_dim))
            for i in range(self.layer_num):
                outdim = self.embed_dims[-1] if i == self.layer_num - 1 else self.hidden_dim
                self.sa_bm_list.append(AdaptiveModulator(self.hidden_dim))
                self.sa_sm_list.append(nn.Linear(self.hidden_dim, outdim))
            self.sa_sigmoid = nn.Sigmoid()

        # Rate ModNet: Rate Adaptation (RA)
        self.ra_sm_list = nn.ModuleList()
        self.ra_bm_list = nn.ModuleList()
        if "RA" in model:
            self.ra_sm_list.append(nn.Linear(self.embed_dims[-1], self.hidden_dim))
            for i in range(self.layer_num):
                outdim = self.embed_dims[-1] if i == self.layer_num - 1 else self.hidden_dim
                self.ra_bm_list.append(AdaptiveModulator(self.hidden_dim))
                self.ra_sm_list.append(nn.Linear(self.hidden_dim, outdim))
            self.ra_sigmoid = nn.Sigmoid()

    def _apply_modulation(self, x, sm_list, bm_list, batch_input, H, W, sigmoid_fn):
        B = x.size(0)
        batch_input = torch.tensor(batch_input, dtype=torch.float).to(x.device)
        batch_input = batch_input.to(x.device).unsqueeze(0).expand(B, -1)

        temp = x.detach()
        for i in range(self.layer_num):
            temp = sm_list[i](temp)
            bm = bm_list[i](batch_input).unsqueeze(1)
            temp = temp * bm

        mod_val = sigmoid_fn(sm_list[-1](temp))
        x = x * mod_val
        return x, mod_val
    
    def forward(self, x, snr=None, rate=None, model=None):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        if model in ['SwinJSCC_w/o_SAandRA', 'SwinJSCC_vq-vae']:
            return self.head_list(x), None
        
        elif model == 'SwinJSCC_w/_SA':
            x, _ = self._apply_modulation(x, self.sa_sm_list, self.sa_bm_list, snr, self.H, self.W, self.sa_sigmoid)
            x = self.head_list(x)
            return x, None
        
        elif model == 'SwinJSCC_w/_RA':
            x, mod_val = self._apply_modulation(x=x, sm_list=self.ra_sm_list, bm_list=self.ra_bm_list, batch_input=rate, H=self.H, W=self.W, sigmoid_fn=self.ra_sigmoid)

        elif model == 'SwinJSCC_w/_SAandRA':
            x, _ = self._apply_modulation(x, self.sa_sm_list, self.sa_bm_list, snr, self.H, self.W, self.sa_sigmoid)
            x, mod_val = self._apply_modulation(x, self.ra_sm_list, self.ra_bm_list, rate, self.H, self.W, self.ra_sigmoid)

        mask = torch.sum(mod_val, dim=1)
        sorted_mask, indices = mask.sort(dim=1, descending=True)
        c_indices = indices[:, :rate]
        add = torch.Tensor(range(0, B * x.size()[2], x.size()[2])).unsqueeze(1).repeat(1, rate)
        c_indices = c_indices + add.int().cuda()
        mask = torch.zeros(mask.size()).reshape(-1).cuda()
        mask[c_indices.reshape(-1)] = 1
        mask = mask.reshape(B, x.size()[2])
        mask = mask.unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1)
        x = x * mask

        return (x, mask)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** (i_layer + 1)),
                                    W // (2 ** (i_layer + 1)))
            
def create_encoder(**kwargs):
    model = SwinJSCC_Encoder(**kwargs)
    return model


def build_model(config):
    input_image = torch.ones([1, 256, 256]).to(config.device)
    model = create_encoder(**config.encoder_kwargs)
    model(input_image)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))