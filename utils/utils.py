# --------------------------------------------------------
# Modified By Mzero
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
from math import inf
import torch
import torch.distributed as dist
from timm.utils import ModelEma as ModelEma
import numpy as np
import random
from torch.optim.lr_scheduler import _LRScheduler
import logging
from datetime import datetime

class GradualWarmupScheduler(_LRScheduler):
    """渐进式预热学习率调度器
    在'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'中提出
    Args:
        optimizer: 优化器
        multiplier: 目标学习率 = 基础学习率 * multiplier
        warm_epoch: 目标学习率在warm_epoch时达到，逐渐增加
        after_scheduler: 在target_epoch之后使用的调度器(例如ReduceLROnPlateau)
    """
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        """获取当前学习率"""
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        """更新学习率"""
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

def save_model(model, save_path, epoch=None, metrics=None):
    """保存模型到指定路径
    Args:
        model: 模型
        save_path: 保存路径
        epoch: 当前epoch
        metrics: 评估指标
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 准备保存的数据
    save_dict = {
        'model': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    
    # 保存模型
    torch.save(save_dict, save_path)
    logging.info(f"模型已保存到 {save_path}")
    if epoch is not None:
        logging.info(f"Epoch: {epoch}")
    if metrics is not None:
        logging.info(f"评估指标: {metrics}")

def load_model(model, load_path):
    """从指定路径加载模型
    Args:
        model: 模型
        load_path: 加载路径
    Returns:
        加载的epoch和评估指标
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"找不到模型文件: {load_path}")
    
    # 加载模型
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    
    epoch = checkpoint.get('epoch')
    metrics = checkpoint.get('metrics')
    
    logging.info(f"模型已从 {load_path} 加载")
    if epoch is not None:
        logging.info(f"加载的Epoch: {epoch}")
    if metrics is not None:
        logging.info(f"加载的评估指标: {metrics}")
    
    return epoch, metrics

def check_gpus():
    """检查GPU可用性
    Returns:
        bool: GPU是否可用
    """
    if not torch.cuda.is_available():
        logging.warning('此脚本只能用于管理NVIDIA GPU，但在您的设备中未找到GPU')
        return False
    elif not 'NVIDIA System Management' in os.popen('nvidia-smi -h').read():
        logging.warning("未找到'nvidia-smi'工具")
        return False
    return True

def parse(line, qargs):
    """解析nvidia-smi返回的GPU信息行
    Args:
        line: GPU信息行
        qargs: 查询参数
    Returns:
        dict: 解析后的GPU信息
    """
    numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']
    power_manage_enable = lambda v: (not 'Not Support' in v)
    to_numberic = lambda v: float(v.upper().strip().replace('MIB','').replace('W',''))
    process = lambda k,v: ((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
    return {k:process(k,v) for k,v in zip(qargs,line.strip().split(','))}

def query_gpu(qargs=[]):
    """查询GPU信息
    Args:
        qargs: 查询参数
    Returns:
        list: GPU信息列表
    """
    qargs = ['index','gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit'] + qargs
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    return [parse(line,qargs) for line in results]

def by_power(d):
    """按功率排序GPU的辅助函数
    Args:
        d: GPU信息字典
    Returns:
        float: 功率比率
    """
    power_infos = (d['power.draw'],d['power.limit'])
    if any(v==1 for v in power_infos):
        logging.warning('GPU {} 的电源管理不可用'.format(d['index']))
        return 1
    return float(d['power.draw'])/d['power.limit']

class GPUManager():
    """GPU设备管理器，用于自动选择最空闲的GPU"""
    def __init__(self, qargs=[]):
        """初始化GPU管理器
        Args:
            qargs: 查询参数
        """
        self.qargs = qargs
        self.gpus = query_gpu(qargs)
        for gpu in self.gpus:
            gpu['specified'] = False
        self.gpu_num = len(self.gpus)

    def _sort_by_memory(self, gpus, by_size=False):
        """按内存排序GPU
        Args:
            gpus: GPU列表
            by_size: 是否按大小排序
        Returns:
            list: 排序后的GPU列表
        """
        if by_size:
            logging.info('按可用内存大小排序')
            return sorted(gpus, key=lambda d:d['memory.free'], reverse=True)
        else:
            logging.info('按可用内存比例排序')
            return sorted(gpus, key=lambda d:float(d['memory.free'])/ d['memory.total'], reverse=True)

    def _sort_by_power(self, gpus):
        """按功率排序GPU
        Args:
            gpus: GPU列表
        Returns:
            list: 排序后的GPU列表
        """
        return sorted(gpus, key=by_power)
    
    def _sort_by_custom(self, gpus, key, reverse=False, qargs=[]):
        """按自定义规则排序GPU
        Args:
            gpus: GPU列表
            key: 排序键
            reverse: 是否反向排序
            qargs: 查询参数
        Returns:
            list: 排序后的GPU列表
        """
        if isinstance(key,str) and (key in qargs):
            return sorted(gpus, key=lambda d:d[key], reverse=reverse)
        if isinstance(key,type(lambda a:a)):
            return sorted(gpus, key=key, reverse=reverse)
        raise ValueError("参数'key'必须是函数或查询参数中的键")

    def auto_choice(self, mode=3):
        """自动选择最空闲的GPU
        Args:
            mode: 选择模式
                0: 按内存大小
                1: 按功率
                2: 按功率（备用）
                3: 按内存比例
        Returns:
            int: 选择的GPU索引
        """
        for old_infos,new_infos in zip(self.gpus,query_gpu(self.qargs)):
            old_infos.update(new_infos)
        unspecified_gpus = [gpu for gpu in self.gpus if not gpu['specified']] or self.gpus
        
        if mode == 0:
            chosen_gpu = self._sort_by_memory(unspecified_gpus, True)
        elif mode == 1:
            chosen_gpu = self._sort_by_power(unspecified_gpus)
        elif mode == 2:
            chosen_gpu = self._sort_by_power(unspecified_gpus)
        else:
            chosen_gpu = self._sort_by_memory(unspecified_gpus)
            
        if int(chosen_gpu[0]['index']) == 3:
            chosen_gpu = chosen_gpu[1]
        else:
            chosen_gpu = chosen_gpu[0]
            
        chosen_gpu['specified'] = True
        index = chosen_gpu['index']
        logging.info(f'使用GPU {index}: {chosen_gpu}')
        return int(index)

def setup_logger(config):
    """设置日志记录器
    Args:
        config: 配置对象
    Returns:
        logging.Logger: 日志记录器实例
    """
    log_dir = os.path.join(config.OUTPUT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()

def seed_torch(seed=1024):
    """设置随机种子以确保可重复性
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logging.info(f"设置随机种子: {seed}")

def get_model_size(model):
    """获取模型大小
    Args:
        model: 模型
    Returns:
        float: 模型大小（MB）
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb