import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat

from models.mobilemamba import *

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SNR_embedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.SNRembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, SNR):
        
        emb1 = self.SNRembedding(SNR)
        return emb1
    
class AdaptiveModulator(nn.Module):
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )

    def forward(self, snr):
        return self.fc(snr)
    
class Mamba_encoder(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        out_chans=36,
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        # =========================
        ssm_ratio=2.0,
        forward_type="v2",
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN",
        sample_version: str = "v1", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False, 
        channel_adaptive="CA",
        img_resolution=256,
        channel_input='conv',
        # MobileMamba specific parameters
        global_ratio=0.8,
        local_ratio=0.2,
        kernels=5,
        **kwargs,
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.num_features = dims[-1]
        self.dims = dims
        self.channel_adaptive=channel_adaptive
        self.global_ratio = global_ratio
        self.local_ratio = local_ratio
        self.kernels = kernels
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        if norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer)

        _make_downsample = dict(
            v1=PatchMerging, #mobilemamba
        ).get(sample_version, None)

        if channel_adaptive == "CA":
            self.SNR_embedding = SNR_embedding(25, dims[0], dims[0])
            self.proj_list=nn.ModuleList()
        elif channel_adaptive == 'attn':
            self.hidden_dim = int(self.dims[-1] * 1.5)
            self.layer_num = layer_num = 7
            self.bm_list = nn.ModuleList()
            self.sm_list = nn.ModuleList()
            self.sm_list.append(nn.Linear(self.dims[-1], self.hidden_dim))
            for i in range(layer_num):
                if i == layer_num - 1:
                    outdim = self.dims[-1]
                else:
                    outdim = self.hidden_dim
                self.bm_list.append(AdaptiveModulator(self.hidden_dim))
                self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
            self.sigmoid = nn.Sigmoid()
        elif channel_adaptive=='no':
            pass
        else:
            raise ValueError("channel adaptive error")

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()
        
            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                ssm_ratio=ssm_ratio,
                forward_type=forward_type,
                SNR_dim=dims[0],
                resolution=img_resolution//(2**(i_layer+1)),
            ))
        self.channel_input=channel_input
        if channel_input=='conv':
            self.head=nn.Conv2d(dims[-1], out_chans, kernel_size=1, padding=0, stride=1)
        elif channel_input=='fc':
            self.head=nn.Linear(dims[-1], out_chans)

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=2, patch_norm=True, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()), 
        )

    def _make_layer(self,
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        ssm_ratio=2.0,
        forward_type="v2",
        SNR_dim=96,
        resolution=128,
    ):
        depth = len(drop_path)
        blocks = []
        if self.channel_adaptive == "CA":
            self.proj_list.append(nn.Linear(SNR_dim, dim))
        for d in range(depth):
            blocks.append(MobileMambaBlock(
                type='s',
                ed=dim,
                global_ratio=self.global_ratio,
                local_ratio=self.local_ratio,
                kernels=self.kernels,
                drop_path=drop_path[d],
                ssm_ratio=ssm_ratio,
                forward_type=forward_type
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self,x: torch.Tensor, SNR):
        B, C, H, W =x.shape
        x = self.patch_embed(x)
        if self.channel_adaptive=='ssm':
            for layer in self.layers:
                x=layer((x,SNR))
            x=x[0]
        elif self.channel_adaptive == 'attn':
            for layer in self.layers:
                x = layer((x,SNR))
            
            x=x[0]
            x=x.flatten(1,2)
            snr_cuda = torch.tensor(SNR, dtype=torch.float).cuda()
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)

            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)

                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, H*W//(4**self.num_layers) , -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            b,l,c=x.shape
            x=x.view(B, int(math.sqrt(l)), int(math.sqrt(l)), -1)
        else:
            for layer in self.layers:
                x = layer((x,SNR))
        x=x[0]
        x=x.permute(0,3,1,2)
        x = self.head(x)
        return x




def create_encoder(config):
    encoder_kwargs=dict(        
        patch_size=config.MODEL.VSSM.PATCH_SIZE, 
        in_chans=config.MODEL.VSSM.IN_CHANS, 
        out_chans=config.MODEL.VSSM.OUT_CHANS,
        depths=config.MODEL.VSSM.DEPTHS, 
        dims=config.MODEL.VSSM.EMBED_DIM, 
        # ===================
        ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
        forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
        # ===================
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=config.MODEL.VSSM.PATCH_NORM,
        norm_layer=config.MODEL.VSSM.NORM_LAYER,
        sample_version=config.MODEL.VSSM.DOWNSAMPLE,
        patchembed_version=config.MODEL.VSSM.PATCHEMBED,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        channel_adaptive=config.CHANNEL.ADAPTIVE,
        img_resolution=config.DATA.IMG_SIZE,
        channel_input=config.MODEL.VSSM.channel_input,
        # MobileMamba specific parameters
        global_ratio=config.MODEL.VSSM.GLOBAL_RATIO,
        local_ratio=config.MODEL.VSSM.LOCAL_RATIO,
        kernels=config.MODEL.VSSM.KERNELS,
    )
    
    model = Mamba_encoder(**encoder_kwargs)
    return model

