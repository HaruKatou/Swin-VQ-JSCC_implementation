import sys
import os
from pathlib import Path

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.append(str(project_root))

import argparse
import torch
import numpy as np
from pathlib import Path
from data.datasets import Datasets
from utils.helpers import AverageMeter
from training.loss import MS_SSIM
from utils.bpgcodec import BPGCodec

def main():
    current_file = Path(__file__).resolve()
    workspace_root = current_file.parents[3]
    bpg_dir = workspace_root / "BPG"
    prj_root = workspace_root / "SwinJSCC_implementation"

    codec = BPGCodec(
        encoder_path=str(bpg_dir / "bpgenc.exe"), 
        decoder_path=str(bpg_dir / "bpgdec.exe")
    )

    test_dir = [str(prj_root / "dataset" / "raw" / "kodak")]
    dataset = Datasets(test_dir)
    ssim_module = MS_SSIM(data_range=1.0, levels=4, channel=3)

    q_values = [25, 30, 35, 40, 45] 
    
    print(f"Testing BPG on {len(dataset)} images...")

    for q in q_values:
        print(f"Evaluating QP={q}...")
        meters = {k: AverageMeter() for k in ["psnr", "msssim", "bpp"]}
        
        for i in range(len(dataset)):
            img, name = dataset[i]
            img = img.unsqueeze(0) # (1, 3, H, W)

            # Run Codec
            recon, bpp = codec(img, quality=q)

            # Calculate Metrics
            mse = torch.mean((img - recon)**2)
            psnr = 10 * torch.log10(1.0 / mse).item()
            msssim = ssim_module(img, recon.clamp(0, 1)).mean().item()

            meters["psnr"].update(psnr)
            meters["msssim"].update(msssim)
            meters["bpp"].update(bpp)

        print(f"QP: {q} | BPP: {meters['bpp'].avg:.4f} | PSNR: {meters['psnr'].avg:.2f} | MS-SSIM: {meters['msssim'].avg:.4f}")

if __name__ == "__main__":
    main()