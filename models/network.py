'''
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
'''
from models.encoder import create_encoder
from models.decoder import create_decoder
import torch.nn as nn


class Mamba_encoder(nn.Module):
    """
    MobileMamba编码器包装类
    """
    def __init__(self, config):
        super(Mamba_encoder, self).__init__()
        self.config = config
        self.encoder = create_encoder(config)

    def forward(self, input_image, SNR):
        """
        前向传播
        
        Args:
            input_image (torch.Tensor): 输入图像
            SNR (int): 信噪比值
            
        Returns:
            torch.Tensor: 编码特征
        """
        feature = self.encoder(input_image, SNR)
        return feature


class Mamba_decoder(nn.Module):
    """
    MobileMamba解码器包装类
    """
    def __init__(self, config):
        super(Mamba_decoder, self).__init__()
        self.config = config
        self.decoder = create_decoder(config)

    def forward(self, feature, SNR):
        """
        前向传播
        
        Args:
            feature (torch.Tensor): 输入特征
            SNR (int): 信噪比值
            
        Returns:
            torch.Tensor: 重建图像
        """
        recon_image = self.decoder(feature, SNR)
        return recon_image


class Mamba_classify(nn.Module):
    """
    MobileMamba分类器包装类
    """
    def __init__(self, config):
        super(Mamba_classify, self).__init__()
        self.config = config
        self.decoder = create_decoder(config)

    def forward(self, feature, SNR):
        """
        前向传播
        
        Args:
            feature (torch.Tensor): 输入特征
            SNR (int): 信噪比值
            
        Returns:
            torch.Tensor: 分类结果
        """
        recon_image = self.decoder(feature, SNR)
        return recon_image