# --------------------------------------------------------
# Modified by Mzero
# --------------------------------------------------------
# MobileMambaJSCC
# Copyright (c) 2024
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.TRAIN_BATCH = 4
_C.DATA.TEST_BATCH = 1
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'DIV2K'
# Input image size
_C.DATA.IMG_SIZE = 256
# path
_C.DATA.train_data_dir = r"/mnt/wutong/datasets/DIV2K/DIV2K_train_HR"
_C.DATA.test_data_dir = r"/mnt/wutong/datasets/DIV2K/DIV2K_valid_HR"
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
_C.DATA.ZIP_MODE = False
# Cache Data in Memory
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# [SimMIM] Mask settings
_C.DATA.MASK_PATCH_SIZE = 32
_C.DATA.MASK_RATIO = 0.6
_C.DATA.REQUIRE_DISTRIBUTION = True
_C.DATA.RANGE = 5
_C.DATA.SCALE = 10000

# -----------------------------------------------------------------------------
# Channel settings
# -----------------------------------------------------------------------------
_C.CHANNEL = CN()
_C.CHANNEL.TYPE = 'awgn'
_C.CHANNEL.SNR = [20]
_C.CHANNEL.ADAPTIVE = 'CA'
_C.CHANNEL.INPUT = 'conv'

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'mobilemamba'
# Model name
_C.MODEL.NAME = 'mobilemamba_tiny'
# Pretrained weight from checkpoint
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume
_C.MODEL.RESUME = ''
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# MobileMamba parameters
_C.MODEL.MOBILEMAMBA = CN()

_C.MODEL.MOBILEMAMBA.PATCH_SIZE = 4
_C.MODEL.MOBILEMAMBA.IN_CHANS = 3
_C.MODEL.MOBILEMAMBA.OUT_CHANS = 36
_C.MODEL.MOBILEMAMBA.DEPTHS = [2, 2, 6, 2]
_C.MODEL.MOBILEMAMBA.EMBED_DIM = [96, 192, 384, 768]
_C.MODEL.MOBILEMAMBA.TYPE = 's'
_C.MODEL.MOBILEMAMBA.ED = 96
_C.MODEL.MOBILEMAMBA.GLOBAL_RATIO = 0.5
_C.MODEL.MOBILEMAMBA.LOCAL_RATIO = 0.5
_C.MODEL.MOBILEMAMBA.KERNELS = [3, 3, 3, 3]

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 1
_C.TRAIN.SAVE_FRE = 1 
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 1e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# Loss function
_C.TRAIN.LOSS = 'MSE'
_C.TRAIN.DATA_PARALLEL = False
_C.TRAIN.EVAL_MATRIX = 'PSNR'
_C.TRAIN.GAN_LOSS = False
_C.TRAIN.DIS_WEIGHT = 0.5
_C.TRAIN.START_EPOCH = 10
_C.TRAIN.ENCODER_PATH = '/mnt/wutong/MambaJSCCcheckpoints/Journal/encoder'
_C.TRAIN.DECODER_PATH = '/mnt/wutong/MambaJSCCcheckpoints/Journal/decoder'

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.COLOR_JITTER = 0.4
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
_C.AUG.REPROB = 0.25
_C.AUG.REMODE = 'pixel'
_C.AUG.RECOUNT = 1
_C.AUG.MIXUP = 0.8
_C.AUG.CUTMIX = 1.0
_C.AUG.CUTMIX_MINMAX = None
_C.AUG.MIXUP_PROB = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.CROP = True
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.ENABLE_AMP = False
_C.AMP_ENABLE = True
_C.AMP_OPT_LEVEL = ''
_C.OUTPUT = ''
_C.TAG = 'default'
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 10
_C.SEED = 0
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False
_C.FUSED_LAYERNORM = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.TRAIN_BATCH = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output dir
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
