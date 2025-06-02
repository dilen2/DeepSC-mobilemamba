# --------------------------------------------------------
# Modified by Mzero
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import bisect
import torch
import logging
import math
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):
    """构建学习率调度器
    Args:
        config: 配置对象
        optimizer: 优化器
        n_iter_per_epoch: 每个epoch的迭代次数
    Returns:
        学习率调度器实例
    """
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)
    multi_steps = [i * n_iter_per_epoch for i in config.TRAIN.LR_SCHEDULER.MULTISTEPS]

    logging.info(f"构建学习率调度器: {config.TRAIN.LR_SCHEDULER.NAME}")
    logging.info(f"总步数: {num_steps}, 预热步数: {warmup_steps}, 衰减步数: {decay_steps}")

    lr_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps) if config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX else num_steps,
            t_mul=1.,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=config.TRAIN.LR_SCHEDULER.MIN_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            milestones=multi_steps,
            gamma=config.TRAIN.LR_SCHEDULER.GAMMA,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'cyclic':
        lr_scheduler = CyclicLRScheduler(
            optimizer,
            base_lr=config.TRAIN.BASE_LR,
            max_lr=config.TRAIN.LR_SCHEDULER.MAX_LR,
            step_size=config.TRAIN.LR_SCHEDULER.STEP_SIZE * n_iter_per_epoch,
            mode=config.TRAIN.LR_SCHEDULER.MODE,
            gamma=config.TRAIN.LR_SCHEDULER.GAMMA,
            warmup_t=warmup_steps,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            t_in_epochs=False,
        )
    else:
        raise NotImplementedError(f"不支持的学习率调度器类型: {config.TRAIN.LR_SCHEDULER.NAME}")

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    """线性学习率调度器"""
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        """初始化线性学习率调度器
        Args:
            optimizer: 优化器
            t_initial: 总步数
            lr_min_rate: 最小学习率比例
            warmup_t: 预热步数
            warmup_lr_init: 预热初始学习率
            t_in_epochs: 是否以epoch为单位
            noise_range_t: 噪声范围
            noise_pct: 噪声百分比
            noise_std: 噪声标准差
            noise_seed: 噪声种子
            initialize: 是否初始化
        """
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        """获取当前学习率
        Args:
            t: 当前步数
        Returns:
            当前学习率列表
        """
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        """获取指定epoch的学习率
        Args:
            epoch: epoch数
        Returns:
            学习率列表
        """
        if self.t_in_epochs:
            return self._get_lr(epoch)
        return None

    def get_update_values(self, num_updates: int):
        """获取指定更新次数的学习率
        Args:
            num_updates: 更新次数
        Returns:
            学习率列表
        """
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        return None


class MultiStepLRScheduler(Scheduler):
    """多步长学习率调度器"""
    def __init__(self, 
                 optimizer: torch.optim.Optimizer, 
                 milestones, 
                 gamma=0.1, 
                 warmup_t=0, 
                 warmup_lr_init=0, 
                 t_in_epochs=True) -> None:
        """初始化多步长学习率调度器
        Args:
            optimizer: 优化器
            milestones: 学习率调整的步数列表
            gamma: 学习率衰减率
            warmup_t: 预热步数
            warmup_lr_init: 预热初始学习率
            t_in_epochs: 是否以epoch为单位
        """
        super().__init__(optimizer, param_group_field="lr")
        
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        
        assert self.warmup_t <= min(self.milestones), "预热步数必须小于第一个milestone"
    
    def _get_lr(self, t):
        """获取当前学习率
        Args:
            t: 当前步数
        Returns:
            当前学习率列表
        """
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            lrs = [v * (self.gamma ** bisect.bisect_right(self.milestones, t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        """获取指定epoch的学习率
        Args:
            epoch: epoch数
        Returns:
            学习率列表
        """
        if self.t_in_epochs:
            return self._get_lr(epoch)
        return None

    def get_update_values(self, num_updates: int):
        """获取指定更新次数的学习率
        Args:
            num_updates: 更新次数
        Returns:
            学习率列表
        """
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        return None


class CyclicLRScheduler(Scheduler):
    """循环学习率调度器"""
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 base_lr: float,
                 max_lr: float,
                 step_size: int,
                 mode='triangular',
                 gamma=1.,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True) -> None:
        """初始化循环学习率调度器
        Args:
            optimizer: 优化器
            base_lr: 基础学习率
            max_lr: 最大学习率
            step_size: 步长
            mode: 循环模式 ('triangular', 'triangular2', 'exp_range')
            gamma: 衰减率
            warmup_t: 预热步数
            warmup_lr_init: 预热初始学习率
            t_in_epochs: 是否以epoch为单位
        """
        super().__init__(optimizer, param_group_field="lr")
        
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
    
    def _get_lr(self, t):
        """获取当前学习率
        Args:
            t: 当前步数
        Returns:
            当前学习率列表
        """
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            cycle = math.floor(1 + t / (2 * self.step_size))
            x = abs(t / self.step_size - 2 * cycle + 1)
            
            if self.mode == 'triangular':
                lrs = [self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) for _ in self.base_values]
            elif self.mode == 'triangular2':
                lrs = [self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) / (2 ** (cycle - 1)) for _ in self.base_values]
            elif self.mode == 'exp_range':
                lrs = [self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * (self.gamma ** t) for _ in self.base_values]
            else:
                raise ValueError(f"不支持的循环模式: {self.mode}")
        return lrs

    def get_epoch_values(self, epoch: int):
        """获取指定epoch的学习率
        Args:
            epoch: epoch数
        Returns:
            学习率列表
        """
        if self.t_in_epochs:
            return self._get_lr(epoch)
        return None

    def get_update_values(self, num_updates: int):
        """获取指定更新次数的学习率
        Args:
            num_updates: 更新次数
        Returns:
            学习率列表
        """
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        return None
