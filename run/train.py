'''
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
'''
from models.network import Mamba_encoder, Mamba_decoder
from models.channel import Channel
from data.datasets import get_loader

import torch.optim as optim
from tqdm import tqdm
import torch
from utils.utils import *
from utils.distortion import *
from torchvision.utils import save_image
from utils.utils import seed_torch
import os

def train_MambaJSCC(config):
    train_loader, _ = get_loader(config)
    encoder = Mamba_encoder(config).cuda()
    decoder = Mamba_decoder(config).cuda()
    channel = Channel(config)

    optimizer_encoder = optim.AdamW(encoder.parameters(), lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    optimizer_decoder = optim.AdamW(decoder.parameters(), lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    cosineScheduler_encoder = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_encoder, T_max=config.TRAIN.EPOCHS, eta_min=config.TRAIN.MIN_LR, last_epoch=-1)
    warmUpScheduler_encoder = GradualWarmupScheduler(
        optimizer=optimizer_encoder, multiplier=1., warm_epoch=config.TRAIN.WARMUP_EPOCHS,
        after_scheduler=cosineScheduler_encoder)
    
    cosineScheduler_decoder = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_decoder, T_max=config.TRAIN.EPOCHS, eta_min=config.TRAIN.MIN_LR, last_epoch=-1)
    warmUpScheduler_decoder = GradualWarmupScheduler(
        optimizer=optimizer_decoder, multiplier=1., warm_epoch=config.TRAIN.WARMUP_EPOCHS,
        after_scheduler=cosineScheduler_decoder)
    
    criterion = loss_matrix(config)
    matrix = eval_matrix(config)

    encoder.train()
    decoder.train()
    
    print(f"Model configuration:")
    print(f"Encoder: {config.MODEL.MOBILEMAMBA.EMBED_DIM}, {config.MODEL.MOBILEMAMBA.DEPTHS}")
    print(f"Training parameters:")
    print(f"OUT_CHANS: {config.MODEL.MOBILEMAMBA.OUT_CHANS}")
    print(f"LOSS: {config.TRAIN.LOSS}")
    print(f"SNR: {config.CHANNEL.SNR}")
    print(f"ADAPTIVE: {config.CHANNEL.ADAPTIVE}")
    print(f"TYPE: {config.CHANNEL.TYPE}")
    print(f"DEPTH: {len(config.MODEL.MOBILEMAMBA.EMBED_DIM)}")
    print(f"EMBED_DIM: {config.MODEL.MOBILEMAMBA.EMBED_DIM}")
    print(f"DEPTHS: {config.MODEL.MOBILEMAMBA.DEPTHS}")
    print(f"RESOLUTION: {config.DATA.IMG_SIZE}")

    seed_torch()
    for e in range(config.TRAIN.EPOCHS):
        loss_ave = 0
        
        with tqdm(train_loader, dynamic_ncols=False) as tqdmTrainData:
            for i, (input_image, target) in enumerate(tqdmTrainData):
                SNR_list = config.CHANNEL.SNR
                SNR_index = torch.randint(0, len(SNR_list), (1,)).item()
                SNR = SNR_list[SNR_index]

                # Encoder
                input_image = input_image.cuda()
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()     

                feature = encoder(input_image, SNR)
                CBR = feature.numel() / input_image.numel() / 2
                
                # Channel
                received, pwr, h = channel.forward(feature, SNR)
                if config.CHANNEL.TYPE == 'rayleigh':
                    sigma_square = 1.0 / (10 ** (SNR / 10))
                    received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
                elif config.CHANNEL.TYPE == 'awgn':
                    pass
                else:
                    raise ValueError("channel type error")

                # Decoder
                received = torch.cat((torch.real(received), torch.imag(received)), dim=2) * torch.sqrt(pwr)
                recon_image = decoder(received, SNR)

                loss = criterion(recon_image, input_image, feature, opt_idx=0, global_step=e)
                loss.backward()

                performance = matrix(recon_image, input_image)
                loss_ave = (loss_ave + loss.item())

                torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.TRAIN.CLIP_GRAD)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.TRAIN.CLIP_GRAD)
                
                optimizer_encoder.step()
                optimizer_decoder.step()

                tqdmTrainData.set_postfix({
                    'e': e,
                    'loss': (loss.item(), loss_ave/(i+1)),
                    'matrix': performance,
                    'CBR': CBR,
                    'SNR': SNR,
                    "LR": (optimizer_encoder.state_dict()['param_groups'][0]["lr"], 
                          optimizer_decoder.state_dict()['param_groups'][0]["lr"])
                })

        warmUpScheduler_encoder.step()
        warmUpScheduler_decoder.step()
        loss_ave = loss_ave/(i+1)

        if (e + 1) % config.TRAIN.SAVE_FRE == 0:
            model_name = f"mobilemamba_tiny_OUTCHANS{config.MODEL.MOBILEMAMBA.OUT_CHANS}_loss{config.TRAIN.LOSS}_SNR{config.CHANNEL.SNR}_adp{config.CHANNEL.ADAPTIVE}_type{config.CHANNEL.TYPE}_depth{len(config.MODEL.MOBILEMAMBA.EMBED_DIM)}_embed{config.MODEL.MOBILEMAMBA.EMBED_DIM}_nums{config.MODEL.MOBILEMAMBA.DEPTHS}_rsl{config.DATA.IMG_SIZE}"
            save_model(encoder, save_path=os.path.join(config.TRAIN.ENCODER_PATH, model_name + '.pt'))
            save_model(decoder, save_path=os.path.join(config.TRAIN.DECODER_PATH, model_name + '.pt'))


