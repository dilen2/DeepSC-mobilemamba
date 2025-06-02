import numpy as np
import torch
import torch.nn as nn


class Channel(nn.Module):
    """
    信道模型类
    
    支持以下信道类型：
    - 无噪声信道
    - AWGN信道
    - Rayleigh信道
    """

    def __init__(self, config):
        """
        初始化信道模型
        
        Args:
            config: 配置对象，包含信道参数
        """
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = config.CHANNEL.TYPE
        #self.config.CUDA=True
        #self.device = config.device
        #self.h = torch.sqrt(torch.randn(1) ** 2
        #                   + torch.randn(1) ** 2) / 1.414


    def gaussian_noise_layer(self, input_layer, std, name=None):
        """
        添加高斯噪声
        
        Args:
            input_layer (torch.Tensor): 输入张量
            std (float): 噪声标准差
            name (str, optional): 层名称
            
        Returns:
            torch.Tensor: 添加噪声后的张量
        """
        device = input_layer.get_device()

        # print(np.shape(input_layer))
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std, name=None):
        """
        添加Rayleigh噪声
        
        Args:
            input_layer (torch.Tensor): 输入张量
            std (float): 噪声标准差
            name (str, optional): 层名称
            
        Returns:
            tuple: (添加噪声后的张量, 信道系数)
        """
        device = input_layer.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        
        h = (torch.normal(mean=0.0, std=1, size=np.shape(input_layer), device=device)
             + 1j * torch.normal(mean=0.0, std=1, size=np.shape(input_layer), device=device)) / np.sqrt(2)
        
        #if self.config.CUDA:
        noise = noise.to(input_layer.get_device())
        h = h.to(input_layer.get_device())
        return input_layer * h + noise, h

    def complex_normalize(self, x, power=1):
        """
        复数归一化
        
        Args:
            x (torch.Tensor): 输入张量
            power (float): 目标功率
            
        Returns:
            tuple: (归一化后的张量, 原始功率)
        """
        # print(x.shape)
        pwr = torch.mean(x ** 2) * 2  # 复数功率是实数功率2倍
        out = np.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr

    def reyleigh_layer(self, x):

        L = x.shape[2]
        channel_in = x[:, :, :L // 2, :] + x[:, :, L // 2:, :] * 1j
        h = torch.sqrt(torch.normal(mean=0.0, std=1, size=np.shape(channel_in)) ** 2
                       + torch.normal(mean=0.0, std=1, size=np.shape(channel_in)) ** 2) / np.sqrt(2)
        h = h.cuda()
        channel_output = channel_in * h
        channel_output = torch.cat((torch.real(channel_output), torch.imag(channel_output)), dim=2)
        # channel_output = channel_output.reshape(x.shape)
        # h = torch.cat((torch.real(h), torch.imag(h)), dim=2)
        # h = h.reshape(x.shape)

        return channel_output, h

    def forward(self, input, chan_param, avg_pwr=False):
        """
        前向传播
        
        Args:
            input (torch.Tensor): 输入张量
            chan_param (float): 信道参数（如SNR）
            avg_pwr (bool): 是否使用平均功率
            
        Returns:
            tuple: (信道输出, 功率, 信道系数)
        """
        # 功率归一化
        if avg_pwr:
            power = 1
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)
        # print(input.shape)
        input_shape = channel_tx.shape
        # channel_in = channel_tx.reshape(-1)
        channel_in = channel_tx
        L = channel_in.shape[2]
        channel_in = channel_in[:, :, :L // 2, :] + channel_in[:, :, L // 2:, :] * 1j
        channel_output, h = self.complex_forward(channel_in, chan_param)
        #channel_output = torch.cat((torch.real(channel_output), torch.imag(channel_output)), dim=2)
        # h = torch.cat((torch.real(h), torch.imag(h)), dim=2)
        # channel_output = channel_output.reshape(input_shape)
        if self.chan_type in [1, 'awgn']:
            #noise = (channel_output - channel_tx).detach()
            #noise.requires_grad = False
            #channel_tx = channel_tx + noise
            return channel_output, pwr, torch.ones(channel_output.shape, device=channel_output.device)
            # if avg_pwr:
            #     return channel_tx * torch.sqrt(avg_pwr * 2)
            # else:
            #     return channel_tx * torch.sqrt(pwr)

        elif self.chan_type in [2, 'rayleigh']:
            
            return channel_output, pwr, h
        else:
            raise ValueError(f"不支持的信道类型: {self.chan_type}")

    def complex_forward(self, channel_in, chan_param):
        """
        复数前向传播
        
        Args:
            channel_in (torch.Tensor): 输入张量
            chan_param (float): 信道参数
            
        Returns:
            tuple: (信道输出, 信道系数)
        """
        if self.chan_type in [0, 'none']:
            return channel_in, 1

        elif self.chan_type in [1, 'awgn']:
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))  # 实部虚部分别加，所以除2
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="awgn_chan_noise")
            return chan_output,1

        elif self.chan_type in [2, 'rayleigh']:
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output, h = self.rayleigh_noise_layer(channel_tx,
                                                       std=sigma,
                                                       name="rayleigh_chan_noise")
            return chan_output, h
        else:
            raise ValueError(f"不支持的信道类型: {self.chan_type}")

    def noiseless_forward(self, channel_in):
        channel_tx = self.normalize(channel_in, power=1)
        return channel_tx
