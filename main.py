'''
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
@modified: Mzero
@date: 2024-01-23
'''

import os
import time
import json
import random
import argparse
import datetime
import numpy as np
import logging

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import accuracy, AverageMeter
from configs.config import get_config
from utils.utils import seed_torch
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count
from timm.utils import ModelEma as ModelEma
from run.train import train_MambaJSCC
from run.eval import eval_MambaJSCC
from utils.utils import GPUManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_gpu():
    """设置GPU环境
    Returns:
        int: 选择的GPU设备索引
    """
    try:
        gm = GPUManager()
        device_idx = gm.auto_choice(mode=3)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
        logger.info(f"使用GPU设备: {device_idx}")
        return device_idx
    except Exception as e:
        logger.error(f"GPU设置失败: {str(e)}")
        raise

def parse_args():
    """解析命令行参数
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser('MobileMambaJSCC训练和评估脚本', add_help=False)
    
    # 基本配置
    parser.add_argument('--cfg', type=str, required=True, help='配置文件路径')
    parser.add_argument('--opts', help="使用命令行键值对修改配置选项", default=[], nargs='+')
    parser.add_argument('--batch-size', type=int, help="单GPU的批次大小")
    parser.add_argument('--data-path', type=str, help='数据集路径')
    
    # 数据集配置
    parser.add_argument('--zip', action='store_true', help='使用压缩数据集而不是文件夹数据集')
    parser.add_argument('--cache-mode', type=str, default='part', 
                        choices=['no', 'full', 'part'],
                        help='缓存模式: no=不缓存, full=缓存所有数据, part=将数据集分片并只缓存一片')
    
    # 训练配置
    parser.add_argument('--resume', help='从检查点恢复')
    parser.add_argument('--accumulation-steps', type=int, help="梯度累积步数")
    parser.add_argument('--use-checkpoint', action='store_true', 
                        help="是否使用梯度检查点以节省内存")
    parser.add_argument('--amp-opt-level', type=str, default='O1', 
                        choices=['O0', 'O1', 'O2'],
                        help='混合精度优化级别: O0=不使用amp, O1=使用amp')
    
    # 输出配置
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='输出文件夹根目录, 完整路径为 <output>/<model_name>/<tag>')
    parser.add_argument('--tag', help='实验标签')
    
    # 运行模式
    parser.add_argument('--eval', action='store_true', help='仅执行评估')
    parser.add_argument('--throughput', action='store_true', help='仅测试吞吐量')
    parser.add_argument('--local_rank', type=int, default=0, 
                        help='DistributedDataParallel的本地rank')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'eval'], help='训练或评估模式')
    
    args = parser.parse_args()
    return args

def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_args()
        logger.info("开始解析配置...")
        config = get_config(args)
        
        # 设置GPU
        device_idx = setup_gpu()
        
        # 根据模式执行相应操作
        if args.mode == 'train':
            logger.info("开始训练模式...")
            seed_torch()
            train_MambaJSCC(config)
            logger.info("训练完成，开始评估...")
            seed_torch()
            eval_MambaJSCC(config)
        elif args.mode == 'eval':
            logger.info("开始评估模式...")
            seed_torch()
            eval_MambaJSCC(config)
        else:
            raise ValueError(f"不支持的模式: {args.mode}")
            
        logger.info("程序执行完成")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main()
