'''
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
'''

import torch
import torch.nn.functional as F
import numpy as np
from taming.modules.losses.vqperceptual import * 
from taming.modules.losses.lpips import LPIPS as lpips
import logging

@torch.jit.script
def create_window(window_size: int, sigma: float, channel: int):
    """创建1-D高斯核
    Args:
        window_size: 高斯核大小
        sigma: 正态分布的标准差
        channel: 输入通道数
    Returns:
        1D高斯核
    """
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    g = g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)
    return g

@torch.jit.script
def _gaussian_filter(x, window_1d, use_padding: bool):
    """使用1-D核进行高斯模糊
    Args:
        x: 输入张量批次
        window_1d: 1-D高斯核
        use_padding: 是否在卷积前进行填充
    Returns:
        模糊后的张量
    """
    C = x.shape[1]
    padding = 0
    if use_padding:
        window_size = window_1d.shape[3]
        padding = window_size // 2
    out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, window_1d.transpose(2, 3), stride=1, padding=(padding, 0), groups=C)
    return out

@torch.jit.script
def ssim(X, Y, window, data_range: float, use_padding: bool = False):
    """计算SSIM指标
    Args:
        X: 输入图像
        Y: 目标图像
        window: 1-D高斯核
        data_range: 输入图像的值范围（通常为1.0或255）
        use_padding: 是否在卷积前进行填充
    Returns:
        SSIM值和对比度结构值
    """
    K1 = 0.01
    K2 = 0.03
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(X, window, use_padding)
    mu2 = _gaussian_filter(Y, window, use_padding)
    sigma1_sq = _gaussian_filter(X * X, window, use_padding)
    sigma2_sq = _gaussian_filter(Y * Y, window, use_padding)
    sigma12 = _gaussian_filter(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    cs_map = F.relu(cs_map)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    cs = cs_map.mean(dim=(1, 2, 3))

    return ssim_val, cs

@torch.jit.script
def ms_ssim(X, Y, window, data_range: float, weights, use_padding: bool = False, eps: float = 1e-8):
    """计算多尺度SSIM
    Args:
        X: 输入图像批次 (N,C,H,W)
        Y: 目标图像批次 (N,C,H,W)
        window: 1-D高斯核
        data_range: 输入图像的值范围
        weights: 不同尺度的权重
        use_padding: 是否在卷积前进行填充
        eps: 用于避免梯度为nan的小值
    Returns:
        多尺度SSIM值
    """
    weights = weights[:, None]
    levels = weights.shape[0]
    vals = []
    
    for i in range(levels):
        ss, cs = ssim(X, Y, window=window, data_range=data_range, use_padding=use_padding)

        if i < levels - 1:
            vals.append(cs)
            X = F.avg_pool2d(X, kernel_size=2, stride=2, ceil_mode=True)
            Y = F.avg_pool2d(Y, kernel_size=2, stride=2, ceil_mode=True)
        else:
            vals.append(ss)

    vals = torch.stack(vals, dim=0)
    vals = vals.clamp_min(eps)
    ms_ssim_val = torch.prod(vals[:-1] ** weights[:-1] * vals[-1:] ** weights[-1:], dim=0)
    return ms_ssim_val

class SSIM(torch.jit.ScriptModule):
    """SSIM计算模块"""
    __constants__ = ['data_range', 'use_padding']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False):
        """初始化SSIM模块
        Args:
            window_size: 高斯核大小
            window_sigma: 正态分布的标准差
            data_range: 输入图像的值范围
            channel: 输入通道数
            use_padding: 是否在卷积前进行填充
        """
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)
        self.data_range = data_range
        self.use_padding = use_padding

    @torch.jit.script_method
    def forward(self, X, Y):
        r = ssim(X, Y, window=self.window, data_range=self.data_range, use_padding=self.use_padding)
        return r[0]

class MS_SSIM(torch.jit.ScriptModule):
    """多尺度SSIM计算模块"""
    __constants__ = ['data_range', 'use_padding', 'eps']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=1.0, channel=3, use_padding=False, weights=None,
                 levels=None, eps=1e-8):
        """初始化多尺度SSIM模块
        Args:
            window_size: 高斯核大小
            window_sigma: 正态分布的标准差
            data_range: 输入图像的值范围
            channel: 输入通道数
            use_padding: 是否在卷积前进行填充
            weights: 不同尺度的权重
            levels: 下采样层数
            eps: 用于避免梯度为nan的小值
        """
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        self.data_range = data_range
        self.use_padding = use_padding
        self.eps = eps

        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = torch.tensor(weights, dtype=torch.float)

        if levels is not None:
            weights = weights[:levels]
            weights = weights / weights.sum()

        self.register_buffer('weights', weights)

    @torch.jit.script_method
    def forward(self, X, Y):
        return 1 - ms_ssim(X, Y, window=self.window, data_range=self.data_range, weights=self.weights,
                       use_padding=self.use_padding, eps=self.eps)

class loss_matrix(torch.nn.Module):
    """损失函数矩阵"""
    def __init__(self, config):
        """初始化损失函数矩阵
        Args:
            config: 配置对象
        """
        super(loss_matrix, self).__init__()
        self.config = config
        self.Cal_lpips = lpips().eval().cuda()
        
        if self.config.TRAIN.GAN_LOSS:
            self.discriminator = NLayerDiscriminator(
                input_nc=config.MODEL.MOBILEMAMBA.OUT_CHANS,
                n_layers=config.MODEL.disc_num_layers,
                use_actnorm=config.MODEL.use_actnorm
            ).cuda().apply(weights_init)
            self.discriminator_weight = config.TRAIN.DIS_WEIGHT
            self.disc_loss = hinge_d_loss
            
        _loss_dict = dict(
            PSNR=self.MSE_loss,
            MSSSIM=self.MSSSIM_loss,
            LPIPS=self.LPIPS_loss
        )
        self.loss = _loss_dict.get(config.TRAIN.LOSS, None)
        logging.info(f"使用损失函数: {config.TRAIN.LOSS}")

    def MSSSIM_loss(self, x, y):
        """计算多尺度SSIM损失
        Args:
            x: 重建图像
            y: 目标图像
        Returns:
            多尺度SSIM损失值
        """
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()
        rec_loss = CalcuSSIM(x, y).mean() * x.numel() / x.shape[0]
        return rec_loss
    
    def MSE_loss(self, x, y):
        """计算MSE损失
        Args:
            x: 重建图像
            y: 目标图像
        Returns:
            MSE损失值
        """
        rec_loss = F.mse_loss(x, y, reduction='sum') / x.shape[0]
        return rec_loss
    
    def LPIPS_loss(self, x, y):
        """计算LPIPS损失
        Args:
            x: 重建图像
            y: 目标图像
        Returns:
            LPIPS损失值
        """
        rec_loss = self.Cal_lpips.forward(x, y).mean() * x.numel() / x.shape[0]
        return rec_loss
    
    def forward(self, recon, input, feature, last_layer=None, opt_idx=0, global_step=0):
        """前向传播
        Args:
            recon: 重建图像
            input: 输入图像
            feature: 特征图
            last_layer: 最后一层
            opt_idx: 优化器索引
            global_step: 全局步数
        Returns:
            损失值
        """
        if self.config.TRAIN.GAN_LOSS:
            if opt_idx == 0:  # 更新自编码器
                recon_loss = self.loss(recon, input)
                logits_fake = self.discriminator(feature)
                g_loss = -torch.mean(logits_fake)
                d_weight = self.calculate_adaptive_weight(recon_loss, g_loss, last_layer=last_layer) if last_layer is not None else 0.5
                g_factor = 1 if global_step + 1 > self.config.TRAIN.START_EPOCH else 0
                loss = recon_loss + d_weight * g_factor * g_loss
                return loss, recon_loss, g_loss
            else:  # 更新判别器
                logits_real = self.discriminator(input)
                logits_fake = self.discriminator(feature.detach())
                d_loss = self.disc_loss(logits_real, logits_fake)
                return d_loss, None, None
        else:
            return self.loss(recon, input), None, None

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """计算自适应权重
        Args:
            nll_loss: 重建损失
            g_loss: 生成器损失
            last_layer: 最后一层
        Returns:
            自适应权重
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
            d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
            d_weight = d_weight * self.discriminator_weight
        else:
            d_weight = self.discriminator_weight
        return d_weight

class eval_matrix(torch.nn.Module):
    """评估指标矩阵"""
    def __init__(self, config):
        """初始化评估指标矩阵
        Args:
            config: 配置对象
        """
        super(eval_matrix, self).__init__()
        self.config = config
        self.Cal_lpips = lpips().eval().cuda()
        logging.info("初始化评估指标矩阵")

    def psnr(self, x, y):
        """计算PSNR
        Args:
            x: 重建图像
            y: 目标图像
        Returns:
            PSNR值
        """
        mse = F.mse_loss(x, y, reduction='mean')
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr

    def msssim(self, x, y):
        """计算多尺度SSIM
        Args:
            x: 重建图像
            y: 目标图像
        Returns:
            多尺度SSIM值
        """
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()
        return 1 - CalcuSSIM(x, y).mean()

    def LPIPS_loss(self, x, y):
        """计算LPIPS
        Args:
            x: 重建图像
            y: 目标图像
        Returns:
            LPIPS值
        """
        return self.Cal_lpips.forward(x, y).mean()

    def forward(self, x, y):
        """前向传播
        Args:
            x: 重建图像
            y: 目标图像
        Returns:
            评估指标字典
        """
        metrics = {
            'PSNR': self.psnr(x, y).item(),
            'MSSSIM': self.msssim(x, y).item(),
            'LPIPS': self.LPIPS_loss(x, y).item()
        }
        return metrics
        