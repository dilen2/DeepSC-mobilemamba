# --------------------------------------------------------
# Modified by Mzero
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from functools import partial
from torch import optim as optim
import logging


def build_optimizer(config, model, logger, **kwargs):
    """构建优化器
    Args:
        config: 配置对象
        model: 模型
        logger: 日志记录器
        **kwargs: 额外参数
    Returns:
        优化器实例
    """
    logger.info(f"==============> 构建优化器 {config.TRAIN.OPTIMIZER.NAME}....................")
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters, no_decay_names = set_weight_decay(model, skip, skip_keywords)
    logger.info(f"无权重衰减列表: {no_decay_names}")

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(
            parameters,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            nesterov=True,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(
            parameters,
            eps=config.TRAIN.OPTIMIZER.EPS,
            betas=config.TRAIN.OPTIMIZER.BETAS,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
    else:
        raise NotImplementedError(f"不支持的优化器类型: {config.TRAIN.OPTIMIZER.NAME}")

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    """设置权重衰减
    Args:
        model: 模型
        skip_list: 跳过权重衰减的参数列表
        skip_keywords: 跳过权重衰减的关键字列表
    Returns:
        参数组列表和无权重衰减的参数名称列表
    """
    has_decay = []
    no_decay = []
    no_decay_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 冻结的权重
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_names.append(name)
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}], no_decay_names


def check_keywords_in_name(name, keywords=()):
    """检查名称中是否包含关键字
    Args:
        name: 参数名称
        keywords: 关键字列表
    Returns:
        是否包含关键字
    """
    return any(keyword in name for keyword in keywords)


# ==========================
# for mim, currently not used, and may have bugs...

def build_optimizer_swimmim(config, model, logger, simmim=True, is_pretrain=False):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    if is_pretrain:
        parameters = get_pretrain_param_groups(model, skip, skip_keywords)
    else:
        depths = config.MODEL.SWIN.DEPTHS if config.MODEL.TYPE == 'swin' else config.MODEL.SWINV2.DEPTHS
        num_layers = sum(depths)
        get_layer_func = partial(get_swin_layer, num_layers=num_layers + 2, depths=depths)
        scales = list(config.TRAIN.LAYER_DECAY ** i for i in reversed(range(num_layers + 2)))
        parameters = get_finetune_param_groups(model, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY, get_layer_func, scales, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    else:
        raise NotImplementedError

    return optimizer


def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def get_swin_layer(name, num_layers, depths):
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split('.')[1])
        block_id = name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())


def build_optimizer_mobilemamba(config, model, logger):
    """构建MobileMamba优化器
    Args:
        config: 配置对象
        model: 模型
        logger: 日志记录器
    Returns:
        优化器实例
    """
    logger.info(f"==============> 构建MobileMamba优化器 {config.TRAIN.OPTIMIZER.NAME}....................")
    
    # 获取参数组
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    
    parameters = get_mobilemamba_param_groups(model, config, skip, skip_keywords)
    
    # 构建优化器
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(
            parameters,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            nesterov=True,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(
            parameters,
            eps=config.TRAIN.OPTIMIZER.EPS,
            betas=config.TRAIN.OPTIMIZER.BETAS,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
    else:
        raise NotImplementedError(f"不支持的优化器类型: {config.TRAIN.OPTIMIZER.NAME}")
    
    logger.info(f"优化器参数组数量: {len(parameters)}")
    for i, param_group in enumerate(parameters):
        logger.info(f"参数组 {i}:")
        logger.info(f"  学习率: {param_group['lr']}")
        logger.info(f"  权重衰减: {param_group.get('weight_decay', 0.0)}")
        logger.info(f"  参数数量: {len(param_group['params'])}")
    
    return optimizer


def get_layer_params(model, layer_id, skip_list=(), skip_keywords=()):
    """获取指定层的参数
    Args:
        model: 模型
        layer_id: 层ID
        skip_list: 跳过权重衰减的参数列表
        skip_keywords: 跳过权重衰减的关键字列表
    Returns:
        参数列表
    """
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if f"layer_{layer_id}" in name:
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                    check_keywords_in_name(name, skip_keywords):
                params.append(param)
    return params


def get_mobilemamba_param_groups(model, config, skip_list=(), skip_keywords=()):
    """获取MobileMamba参数组
    Args:
        model: 模型
        config: 配置对象
        skip_list: 跳过权重衰减的参数列表
        skip_keywords: 跳过权重衰减的关键字列表
    Returns:
        参数组列表
    """
    parameter_groups = []
    
    # 编码器参数
    encoder_params = []
    encoder_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                    check_keywords_in_name(name, skip_keywords):
                encoder_no_decay.append(param)
            else:
                encoder_params.append(param)
    
    if encoder_params:
        parameter_groups.append({
            'params': encoder_params,
            'lr': config.TRAIN.BASE_LR,
            'weight_decay': config.TRAIN.WEIGHT_DECAY
        })
    if encoder_no_decay:
        parameter_groups.append({
            'params': encoder_no_decay,
            'lr': config.TRAIN.BASE_LR,
            'weight_decay': 0.0
        })
    
    # 解码器参数
    decoder_params = []
    decoder_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "decoder" in name:
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                    check_keywords_in_name(name, skip_keywords):
                decoder_no_decay.append(param)
            else:
                decoder_params.append(param)
    
    if decoder_params:
        parameter_groups.append({
            'params': decoder_params,
            'lr': config.TRAIN.BASE_LR * config.TRAIN.DECODER_LR_SCALE,
            'weight_decay': config.TRAIN.WEIGHT_DECAY
        })
    if decoder_no_decay:
        parameter_groups.append({
            'params': decoder_no_decay,
            'lr': config.TRAIN.BASE_LR * config.TRAIN.DECODER_LR_SCALE,
            'weight_decay': 0.0
        })
    
    # 其他参数
    other_params = []
    other_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" not in name and "decoder" not in name:
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                    check_keywords_in_name(name, skip_keywords):
                other_no_decay.append(param)
            else:
                other_params.append(param)
    
    if other_params:
        parameter_groups.append({
            'params': other_params,
            'lr': config.TRAIN.BASE_LR,
            'weight_decay': config.TRAIN.WEIGHT_DECAY
        })
    if other_no_decay:
        parameter_groups.append({
            'params': other_no_decay,
            'lr': config.TRAIN.BASE_LR,
            'weight_decay': 0.0
        })
    
    return parameter_groups

