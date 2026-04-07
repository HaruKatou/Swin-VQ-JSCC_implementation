import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from config import Config
from utils.helpers import *
from utils.logger import *
from training.loss import MS_SSIM
from models.SwinJSCC.network import SwinJSCC
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "checkpoints"

class Trainer:
    def __init__(self, cfg: Config, net: nn.Module, train_loader, test_loader, logger: logging.Logger):
        self.cfg = cfg
        self.net = net.to(cfg.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.logger = logger

        model_params = [{'params': filter(lambda p: p.requires_grad, self.net.parameters()), 'lr': 0.0001}]
        self.optimizer = optim.Adam(model_params, lr=cfg.learning_rate)
        self.ssim = MS_SSIM(data_range=1.0, levels=4, channel=3).to(cfg.device)
        if cfg.trainset == "CIFAR10":
            self.ssim = MS_SSIM(window_size=3, data_range=1.0, levels=4, channel=3).to(cfg.device)

        self.global_step = 0

    @torch.no_grad()
    def _calc_metrics(self, inp: torch.Tensor, recon: torch.Tensor, mse: torch.Tensor):
        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10)).item() if mse.item() > 0 else 0.0
        mssim = 1 - self.ssim(inp, recon.clamp(0, 1)).mean().item()
        return psnr, mssim
    
    def train_one_epoch(self, epoch: int) -> None:
        self.net.train()
        meters = {k: AverageMeter() for k in ["time", "loss", "cbr", "snr", "psnr", "msssim"]}

        for batch in self.train_loader:
            self.global_step += 1
            start = time.time()

            if self.cfg.trainset == "CIFAR10":
                input, _ = batch
            else:
                input = batch[0] if isinstance(batch, (list, tuple)) else batch

            input = input.to(self.cfg.device)

            recon, CBR, SNR, mse, loss = self.net(input)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            meters["time"].update(time.time() - start)
            meters["loss"].update(loss.item())
            meters["cbr"].update(CBR.item() if torch.is_tensor(CBR) else CBR)
            meters["snr"].update(SNR.item() if torch.is_tensor(SNR) else SNR)

            if mse.item() > 0:
                psnr, msssim = self._calc_metrics(input, recon, mse)
                meters["psnr"].update(psnr)
                meters["msssim"].update(msssim)

            if self.global_step % self.cfg.print_step == 0:
                self._log_step(epoch, meters)

        # end of epoch
        self.logger.info(f"Epoch {epoch} finished – "
                         f"Loss {meters['loss'].avg:.4f} "
                         f"PSNR {meters['psnr'].avg:.2f} "
                         f"MS-SSIM {meters['msssim'].avg:.4f}")

    def _log_step(self, epoch: int, meters: dict) -> None:
        prog = (self.global_step % len(self.train_loader)) / len(self.train_loader) * 100
        log = " | ".join([
            f"Epoch {epoch}",
            f"Step [{self.global_step % len(self.train_loader)}/{len(self.train_loader)}={prog:.1f}%]",
            f"Time {meters['time'].val:.3f}",
            f"Loss {meters['loss'].val:.4f} ({meters['loss'].avg:.4f})",
            f"CBR {meters['cbr'].val:.4f} ({meters['cbr'].avg:.4f})",
            f"SNR {meters['snr'].val:.1f} ({meters['snr'].avg:.1f})",
            f"PSNR {meters['psnr'].val:.2f} ({meters['psnr'].avg:.2f})",
            f"MS-SSIM {meters['msssim'].val:.4f} ({meters['msssim'].avg:.4f})",
            f"LR {self.cfg.learning_rate}",
        ])
        self.logger.info(log)

    @torch.no_grad()
    def evaluate(self) -> None:
        self.net.eval()
        snrs = [int(v) for v in self.cfg.multiple_snr.split(",")]
        rates = [int(v) for v in self.cfg.C.split(",")]

        results = {
            "snr": np.zeros((len(snrs), len(rates))),
            "cbr": np.zeros((len(snrs), len(rates))),
            "psnr": np.zeros((len(snrs), len(rates))),
            "msssim": np.zeros((len(snrs), len(rates))),
        }

        for i, snr in enumerate(snrs):
            for j, rate in enumerate(rates):
                meters = {k: AverageMeter() for k in ["time", "cbr", "snr", "psnr", "msssim"]}

                for batch in self.test_loader:
                    start = time.time()
                    if self.cfg.trainset == "CIFAR10":
                        inp, _ = batch
                    else:
                        inp = batch[0] if isinstance(batch, (list, tuple)) else batch
                    inp = inp.to(self.cfg.device)

                    recon, CBR, SNR, mse, _ = self.net(inp, snr, rate)

                    meters["time"].update(time.time() - start)
                    meters["cbr"].update(CBR.item() if torch.is_tensor(CBR) else CBR)
                    meters["snr"].update(SNR.item() if torch.is_tensor(SNR) else SNR)

                    if mse.item() > 0:
                        psnr, msssim = self._calc_metrics(inp, recon, mse)
                        meters["psnr"].update(psnr)
                        meters["msssim"].update(msssim)

                    log = ' | '.join([
                        f'Time {meters["time"].val:.3f}',
                        f'CBR {meters["cbr"].val:.4f} ({meters["cbr"].avg:.4f})',
                        f'SNR {meters["snr"].val:.1f}',
                        f'PSNR {meters["psnr"].val:.3f} ({meters["psnr"].avg:.3f})',
                        f'MSSSIM {meters["msssim"].val:.3f} ({meters["msssim"].avg:.3f})',
                    ])
                    self.logger.info(log)

                # store averages
                results["snr"][i, j] = meters["snr"].avg
                results["cbr"][i, j] = meters["cbr"].avg
                results["psnr"][i, j] = meters["psnr"].avg
                results["msssim"][i, j] = meters["msssim"].avg

                self.logger.info(
                    f"Test SNR={snr} Rate={rate} → "
                    f"CBR {meters['cbr'].avg:.4f} "
                    f"PSNR {meters['psnr'].avg:.3f} "
                    f"MS-SSIM {meters['msssim'].avg:.3f}"
                )

                for t in meters.values():
                    t.reset()

        self._print_results(results)

    def _print_results(self, results: dict) -> None:
        print("SNR: {}".format(results["snr"].tolist()))
        print("CBR: {}".format(results["cbr"].tolist()))
        print("PSNR: {}".format(results["psnr"].tolist()))
        print("MS-SSIM: {}".format(results["msssim"].tolist()))
        print("Finish Test!")

    def save_checkpoint(self, epoch: int) -> None:
        filename = f"EP{epoch}.model"
        path = self.cfg.models / filename

        # Create directory if missing
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.net.state_dict(), path)
        self.logger.info(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.cfg.device)
        self.net.load_state_dict(state, strict=True)
        self.logger.info(f"Pre-trained weights loaded from {path}")