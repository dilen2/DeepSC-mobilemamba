# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import sys
import logging
import functools
from datetime import datetime
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    """创建日志记录器
    Args:
        output_dir: 输出目录
        dist_rank: 分布式训练的进程排名
        name: 日志记录器名称
    Returns:
        日志记录器实例
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'log_rank{dist_rank}_{timestamp}.txt')
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

def setup_logger(config, dist_rank=0):
    """设置日志记录器
    Args:
        config: 配置对象
        dist_rank: 分布式训练的进程排名
    Returns:
        日志记录器实例
    """
    # 创建输出目录
    output_dir = os.path.join(config.OUTPUT, config.TAG)
    os.makedirs(output_dir, exist_ok=True)

    # 创建日志记录器
    logger = create_logger(output_dir, dist_rank, name='MobileMamba')
    
    # 记录配置信息
    logger.info("=" * 50)
    logger.info("MobileMamba 训练配置")
    logger.info("=" * 50)
    logger.info(f"模型配置: {config.MODEL}")
    logger.info(f"训练配置: {config.TRAIN}")
    logger.info(f"数据配置: {config.DATA}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 50)

    return logger

class Logger:
    """日志记录器类"""
    def __init__(self, config, dist_rank=0):
        """初始化日志记录器
        Args:
            config: 配置对象
            dist_rank: 分布式训练的进程排名
        """
        self.logger = setup_logger(config, dist_rank)
        self.config = config
        self.dist_rank = dist_rank

    def info(self, msg):
        """记录信息级别的日志
        Args:
            msg: 日志消息
        """
        self.logger.info(msg)

    def warning(self, msg):
        """记录警告级别的日志
        Args:
            msg: 日志消息
        """
        self.logger.warning(msg)

    def error(self, msg):
        """记录错误级别的日志
        Args:
            msg: 日志消息
        """
        self.logger.error(msg)

    def debug(self, msg):
        """记录调试级别的日志
        Args:
            msg: 日志消息
        """
        self.logger.debug(msg)

    def log_metrics(self, metrics, epoch=None, step=None):
        """记录指标
        Args:
            metrics: 指标字典
            epoch: 当前epoch
            step: 当前步数
        """
        msg = []
        if epoch is not None:
            msg.append(f"Epoch: {epoch}")
        if step is not None:
            msg.append(f"Step: {step}")
        for k, v in metrics.items():
            msg.append(f"{k}: {v:.4f}")
        self.info(" | ".join(msg))
