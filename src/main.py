import torch.optim as optim
from models.SwinJSCC.network import *
from data.datasets import get_loader
from utils.helpers import *
from utils.logger import *
from config import Config
from pathlib import Path
from datetime import datetime
from training.loss import *
from training.trainer import Trainer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import argparse
import time
import torchvision

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SwinJSCC")
    p.add_argument("--training", action="store_true", help="training mode")
    p.add_argument("--trainset", type=str, default="div2k",
                choices=["CIFAR10", "div2k"])
    p.add_argument("--testset", type=str, default="kodak",
                choices=["kodak", "clic21", "ffhq"])
    p.add_argument("--distortion-metric", type=str, default="MSE",
                choices=["MSE", "MS-SSIM"])
    p.add_argument("--model", type=str,
                default="SwinJSCC_w/_SAandRA",
                choices=["SwinJSCC_w/o_SAandRA", "SwinJSCC_w/_SA",
                            "SwinJSCC_w/_RA", "SwinJSCC_w/_SAandRA", "SwinJSCC_vq-vae"])
    p.add_argument("--channel-type", type=str, default="awgn",
                choices=["awgn", "rayleigh"])
    p.add_argument("--C", type=str, default="96", help="bottleneck dimension(s), comma separated")
    p.add_argument("--multiple-snr", type=str, default="10",
                help="SNR values for test, comma separated")
    p.add_argument("--model_size", type=str, default="base",
                choices=["small", "base", "large"])
    p.add_argument("--freeze-encoder-decoder", action="store_true", help="Freeze encoder and decoder parameters")
    return p

def load_weights(model_path, net: nn.Module):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = Config(args)

    cfg.trainset = args.trainset
    cfg.testset = args.testset
    cfg.multiple_snr = args.multiple_snr
    cfg.C = args.C
    cfg.model = args.model
    cfg.model_size = args.model_size

    print("Batch size:", cfg.batch_size)

    seed_torch(cfg.seed)
    logger = configure_logger(cfg)
    logger.info(f"Configuration: {cfg.__dict__}")

    train_loader, test_loader = get_loader(args, cfg)

    net = SwinJSCC(args, cfg).to(cfg.device)

    pretrained_path = r"/kaggle/input/models/harukatou/swin-vq-vae-3/pytorch/default/1/EP1.model"
    if Path(pretrained_path).exists():
        print("Loading author's pretrained model from:", pretrained_path)
        author_checkpoint = torch.load(pretrained_path)

        new_state_dict = {}

    #     print("Translating state_dict keys...")
    #     for old_key, value in author_checkpoint.items():
    #         new_key = old_key

    #        
    #         if new_key.startswith('encoder.'):
                
    #             
    #             if cfg.model == 'SwinJSCC_w/_SA':
    #                 if 'sm_list' in new_key and 'sm_list1' not in new_key:
    #                     new_key = new_key.replace('sm_list', 'sa_sm_list')
    #                 elif 'bm_list' in new_key and 'bm_list1' not in new_key:
    #                     new_key = new_key.replace('bm_list', 'sa_bm_list')
                
    #             
    #             elif cfg.model == 'SwinJSCC_w/o_SAandRA':
    #                 if 'sm_list' in new_key or 'bm_list' in new_key:
    #                     continue

    #             
    #             else:
    #                 if 'sm_list1' in new_key:
    #                     new_key = new_key.replace('sm_list1', 'sa_sm_list')
    #                 elif 'bm_list1' in new_key:
    #                     new_key = new_key.replace('bm_list1', 'sa_bm_list')
    #                 elif 'sm_list' in new_key:
    #                     new_key = new_key.replace('sm_list', 'ra_sm_list')
    #                 elif 'bm_list' in new_key:
    #                     new_key = new_key.replace('bm_list', 'ra_bm_list')

    #         
    #         if 'bm_list' in new_key:
    #             if '.fc.0.' in new_key:
    #                 new_key = new_key.replace('.fc.0.', '.fc1.')
    #             elif '.fc.2.' in new_key:
    #                 new_key = new_key.replace('.fc.2.', '.fc2.')
    #             elif '.fc.4.' in new_key:
    #                 new_key = new_key.replace('.fc.4.', '.fc3.')
            
            # new_state_dict[new_key] = value
        new_state_dict = torch.load(pretrained_path)

        try:
            net.load_state_dict(new_state_dict, strict=True)
            print("Successfully loaded translated state_dict!")
        except RuntimeError as e:
            print("Error loading translated state_dict. There may be more mismatches:")
            print(e)

    trainer = Trainer(cfg, net, train_loader, test_loader, logger)

    if args.training:
        start_epoch = 0
        for epoch in range(start_epoch, cfg.total_epochs):
            print("Epoch [{}/{}]".format(epoch + 1, cfg.total_epochs))
            trainer.train_one_epoch(epoch)
            if (epoch + 1) % cfg.save_model_freq == 0:
                trainer.save_checkpoint(epoch + 1)
                trainer.evaluate()
    else:
        trainer.evaluate()

if __name__ == "__main__":
    main()



