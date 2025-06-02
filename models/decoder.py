import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
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

class Mamba_decoder(nn.Module):
    """
    MobileMamba解码器
    
    使用MobileMambaBlock构建解码器网络，支持不同的通道自适应策略。
    """
    
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        out_chans=3,
        depths=[2, 2, 6, 2],
        dims=[96, 192, 384, 768],
        type='s',
        ed=96,
        global_ratio=0.5,
        local_ratio=0.5,
        kernels=[3, 3, 3, 3],
        drop_path_rate=0.1,
        use_checkpoint=False,
        norm_layer=nn.LayerNorm,
        channel_adaptive="CA",
        channel_input="CA",
        img_resolution=224,
    ):
        """
        初始化MobileMamba解码器
        
        Args:
            patch_size (int): 图像块大小
            in_chans (int): 输入通道数
            out_chans (int): 输出通道数
            depths (list): 每层的深度
            dims (list): 每层的维度
            type (str): MobileMamba类型
            ed (int): 嵌入维度
            global_ratio (float): 全局注意力比例
            local_ratio (float): 局部注意力比例
            kernels (list): 每层的卷积核大小
            drop_path_rate (float): Drop path比率
            use_checkpoint (bool): 是否使用checkpoint
            norm_layer (nn.Module): 归一化层
            channel_adaptive (str): 通道自适应策略
            channel_input (str): 输入通道策略
            img_resolution (int): 输入图像分辨率
        """
        super().__init__()
        
        # 参数验证
        assert len(depths) == len(dims), "depths和dims长度必须相同"
        assert channel_adaptive in ["CA", "attn", None], "channel_adaptive必须是'CA'、'attn'或None"
        assert channel_input in ["CA", "attn", None], "channel_input必须是'CA'、'attn'或None"
        assert 0 <= global_ratio <= 1, "global_ratio必须在0到1之间"
        assert 0 <= local_ratio <= 1, "local_ratio必须在0到1之间"
        assert global_ratio + local_ratio <= 1, "global_ratio和local_ratio之和不能超过1"
        
        # 反转depths和dims以匹配解码器结构
        depths = depths[::-1]
        dims = dims[::-1]
        
        # 设置基本参数
        self.num_layers = len(depths)
        self.num_features = dims[-1]
        self.drop_path_rate = drop_path_rate
        self.use_checkpoint = use_checkpoint
        self.channel_adaptive = channel_adaptive
        self.channel_input = channel_input
        self.type = type
        self.ed = ed
        self.global_ratio = global_ratio
        self.local_ratio = local_ratio
        self.kernels = kernels
        
        # 初始化层
        self.layers = nn.ModuleList()
        self.proj_list = nn.ModuleList()
        self.norm = norm_layer(dims[-1])
        
        # 创建解码器层
        for i_layer in range(self.num_layers):
            self.layers.append(self._make_layer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                SNR_dim=dims[-1],
                resolution=img_resolution//(2**(self.num_layers-i_layer))
            ))
            
        # 创建输出层
        self.head = nn.Linear(out_chans, dims[0])
        
        # 初始化权重
        self.apply(self._init_weights)

    def _make_layer(self, dim, depth, SNR_dim, resolution):
        """
        创建解码器层
        
        Args:
            dim (int): 输入维度
            depth (int): 层深度
            SNR_dim (int): SNR嵌入维度
            resolution (tuple): 输入分辨率 (H, W)
            
        Returns:
            nn.Sequential: 包含MobileMambaBlock和上采样层的序列
        """
        # 创建SNR投影层（如果需要）
        if self.channel_adaptive == "CA":
            self.proj_list.append(nn.Linear(SNR_dim, dim))
            
        # 创建MobileMambaBlock序列
        blocks = []
        for i in range(depth):
            blocks.append(
                MobileMambaBlock(
                    dim=dim,
                    type='s',
                    ed=dim,
                    global_ratio=self.global_ratio,
                    local_ratio=self.local_ratio,
                    kernels=self.kernels,
                    drop_path=self.drop_path_rate * (i / (depth - 1)) if depth > 1 else 0,
                    use_checkpoint=self.use_checkpoint,
                    channel_adaptive=self.channel_adaptive,
                    channel_input=self.channel_input,
                    resolution=resolution
                )
            )
            
        # 创建上采样层
        upsample = PatchReverseMerging2D(
            dim=dim,
            out_dim=dim // 2,
            norm_layer=self.norm_layer
        )
        
        return nn.Sequential(*blocks, upsample)

    def forward(self, x: torch.Tensor, SNR):
        """
        前向传播函数
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, H, W]
            SNR (int): 信噪比值
            
        Returns:
            torch.Tensor: 输出张量，形状为 [B, 3, H, W]
        """
        B, C, H, W = x.shape
        
        # 处理输入通道
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # 根据不同的通道自适应策略处理
        if self.channel_adaptive == "CA":
            # 创建SNR嵌入
            SNR_embed = x.new_ones([B, ], dtype=torch.long) * SNR
            SNR_embedding = self.SNR_embedding(SNR_embed)
            
            # 逐层处理
            for layer, proj in zip(self.layers, self.proj_list):
                emb = proj(SNR_embedding)[:, None, None, :]  # [B, 1, 1, C]
                x = x + emb
                x = layer((x, SNR))
            x = x[0]
            
        elif self.channel_adaptive == 'attn':
            # 逐层处理
            for layer in self.layers:
                x = layer((x, SNR))
            x = x[0]
            
            # 注意力处理
            x = x.flatten(1, 2)  # [B, H*W, C]
            snr_cuda = torch.tensor(SNR, dtype=torch.float).cuda()
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            
            # 应用注意力机制
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)
                    
                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, H*W//(4**self.num_layers), -1)
                temp = temp * bm
                
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            
            # 重塑输出
            b, l, c = x.shape
            x = x.view(B, int(math.sqrt(l)), int(math.sqrt(l)), -1)
            
        else:
            # 直接逐层处理
            for layer in self.layers:
                x = layer((x, SNR))
            x = x[0]
        
        # 转换回原始维度顺序
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        return x

    def flops(self, shape=(3, 224, 224)):
        """
        计算模型的FLOPs和参数量
        
        Args:
            shape (tuple): 输入张量的形状，默认为(3, 224, 224)
            
        Returns:
            str: 包含参数量和FLOPs的字符串
        """
        # 定义支持的算子
        supported_ops = {
            "aten::silu": None,
            "aten::neg": None,
            "aten::exp": None,
            "aten::flip": None,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
        }

        # 创建模型副本并移动到GPU
        model = copy.deepcopy(self)
        model.cuda().eval()

        # 生成随机输入
        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        
        # 计算参数量和FLOPs
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(
            model=model, 
            inputs=(input,), 
            supported_ops=supported_ops
        )

        # 清理内存
        del model, input
        torch.cuda.empty_cache()

        # 返回结果
        return f"参数量: {params:,} FLOPs: {sum(Gflops.values()):.2f}G"

    def _init_weights(self, m):
        """
        初始化模型权重
        
        Args:
            m: 需要初始化的模块
        """
        if isinstance(m, nn.Linear):
            # 使用截断正态分布初始化线性层权重
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # 初始化LayerNorm层
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # 使用kaiming初始化卷积层
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # 初始化BatchNorm层
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

def create_decoder(config):
    """
    创建MobileMamba解码器
    
    Args:
        config: 配置对象，包含模型参数
        
    Returns:
        Mamba_decoder: 解码器模型实例
    """
    decoder_kwargs = dict(        
        patch_size=config.MODEL.MOBILEMAMBA.PATCH_SIZE, 
        in_chans=config.MODEL.MOBILEMAMBA.OUT_CHANS, 
        out_chans=config.MODEL.MOBILEMAMBA.IN_CHANS,
        depths=config.MODEL.MOBILEMAMBA.DEPTHS, 
        dims=config.MODEL.MOBILEMAMBA.EMBED_DIM, 
        type=config.MODEL.MOBILEMAMBA.TYPE,
        ed=config.MODEL.MOBILEMAMBA.ED,
        global_ratio=config.MODEL.MOBILEMAMBA.GLOBAL_RATIO,
        local_ratio=config.MODEL.MOBILEMAMBA.LOCAL_RATIO,
        kernels=config.MODEL.MOBILEMAMBA.KERNELS,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        norm_layer=nn.LayerNorm,
        channel_adaptive=config.CHANNEL.ADAPTIVE,
        channel_input=config.CHANNEL.INPUT,
        img_resolution=config.DATA.IMG_SIZE,
    )
    
    model = Mamba_decoder(**decoder_kwargs)
    return model

