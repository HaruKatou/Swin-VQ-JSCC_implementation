import sys
import os
from pathlib import Path

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.append(str(project_root))

from models.deepJSCC.network import DJSCC_CNN
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
    p = argparse.ArgumentParser(description="DJSCC")
    p.add_argument("--training", action="store_true", help="training mode")
    p.add_argument("--trainset", type=str, default="CIFAR10",
                choices=["CIFAR10", "div2k"])
    p.add_argument("--testset", type=str, default="kodak",
                choices=["kodak", "clic21", "ffhq"])
    p.add_argument("--distortion-metric", type=str, default="MSE",
                choices=["MSE", "MS-SSIM"])
    p.add_argument("--model", type=str,
                default="DJSCC",
                choices=["DJSCC"])
    p.add_argument("--channel-type", type=str, default="awgn",
                choices=["awgn", "rayleigh"])
    p.add_argument("--C", type=str, default="8", help="bottleneck dimension(s), comma separated")
    p.add_argument("--multiple-snr", type=str, default="10",
                help="SNR values for test, comma separated")
    p.add_argument("--model_size", type=str, default="base",
                choices=["small", "base", "large"])
    return p

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

    net = DJSCC_CNN(args, cfg).to(cfg.device)

    # project_root = current_file_path.parent.parent.parent
    # pretrained_path = project_root / "checkpoints" / "DJSCC_CIFAR10_awgn_10_2_0506121901" / "models" / "EP5.model"
    # if Path(pretrained_path).exists():
    #     try:
    #         checkpoint = torch.load(pretrained_path, map_location=cfg.device)
    #         net.load_state_dict(checkpoint, strict=True)
    #         print("Successfully loaded translated state_dict!")
    #     except RuntimeError as e:
    #         print("Error loading translated state_dict. There may be more mismatches:")
    #         print(e)
    # else:
    #     print("Checkpoint not found:", pretrained_path)

    trainer = Trainer(cfg, net, train_loader, test_loader, logger)

    if args.training:
        start_epoch = 0
        for epoch in range(start_epoch, cfg.total_epochs):
            trainer.train_one_epoch(epoch)
            if (epoch + 1) % cfg.save_model_freq == 0:
                trainer.save_checkpoint(epoch + 1)
                trainer.evaluate()
    else:
        trainer.evaluate()

if __name__ == "__main__":
    main()

