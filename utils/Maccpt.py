'''
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
'''

from thop import profile
from thop import clever_format
import torch.nn as nn
import torch
import logging
from models.network import Mamba_encoder, Mamba_decoder

class MobileMambaNet(nn.Module):
    """MobileMamba网络模型"""
    def __init__(self, config):
        """初始化MobileMamba网络
        Args:
            config: 配置对象
        """
        super().__init__()
        self.encoder = Mamba_encoder(config)
        self.decoder = Mamba_decoder(config)
        self.config = config

    def forward(self, input):
        """前向传播
        Args:
            input: 输入张量
        Returns:
            输出张量
        """
        SNR = self.config.CHANNEL.SNR
        x = self.encoder(input, SNR)
        y = self.decoder(x, SNR)
        return y

def test_mem_and_comp(config):
    """测试模型的内存使用和计算量
    Args:
        config: 配置对象
    Returns:
        MACs和参数量
    """
    # 创建模型
    network = MobileMambaNet(config).cuda()
    
    # 创建输入张量
    input = torch.randn(1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE).cuda()
    
    # 计算MACs和参数量
    macs, params = profile(network, inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    del network
    torch.cuda.empty_cache()
    
    # 记录结果
    logging.info(f"模型计算量 (MACs): {macs}")
    logging.info(f"模型参数量: {params}")
    
    return macs, params

def analyze_model_complexity(config):
    """分析模型复杂度
    Args:
        config: 配置对象
    Returns:
        模型复杂度分析结果
    """
    # 测试内存和计算量
    macs, params = test_mem_and_comp(config)
    
    # 计算理论吞吐量
    batch_size = config.DATA.BATCH_SIZE
    img_size = config.DATA.IMG_SIZE
    fps = 1e12 / float(macs.split()[0]) * batch_size  # 假设1T FLOPS
    
    # 记录分析结果
    logging.info("=" * 50)
    logging.info("MobileMamba 模型复杂度分析")
    logging.info("=" * 50)
    logging.info(f"输入图像大小: {img_size}x{img_size}")
    logging.info(f"批次大小: {batch_size}")
    logging.info(f"计算量 (MACs): {macs}")
    logging.info(f"参数量: {params}")
    logging.info(f"理论吞吐量: {fps:.2f} FPS")
    logging.info("=" * 50)
    
    return {
        'macs': macs,
        'params': params,
        'fps': fps,
        'img_size': img_size,
        'batch_size': batch_size
    }