#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = os.path.dirname(__file__)
CONFIG_DEFAULT_PATH = os.path.join(ROOT, "/home/mukh1485/media_project_git/config_machines.yaml")
CONFIG_ENV_VAR = "MACHINE_CONFIG_PATH"

# Make sure we can import the repository's src/ modules and dataset
import sys
sys.path.insert(0, os.path.join(ROOT, "src"))

from dataset_machines import get_dataloaders
from src.latent_encoder.autoencoder import AutoencoderKL


def load_global_config(path: Optional[str] = None) -> Dict[str, Any]:

    if path is None:
        path = os.environ.get(CONFIG_ENV_VAR, CONFIG_DEFAULT_PATH)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Config file not found at '{}'.\n"
            "Either create it there or set {} to point to your YAML.".format(
                path, CONFIG_ENV_VAR
            )
        )

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if "autoencoder" not in cfg:
        raise KeyError("Config is missing top-level 'autoencoder' section.")
    if "paths" not in cfg:
        raise KeyError("Config is missing top-level 'paths' section (for data_root).")

    return cfg


class VAETrainer(object):
    """Trainer for Variational Autoencoder."""

    def _align_hw(self, x: torch.Tensor, y: torch.Tensor):
        """Align spatial dimensions of two tensors (crop to min H,W)."""
        H = min(x.size(-2), y.size(-2))
        W = min(x.size(-1), y.size(-1))
        if x.size(-2) != H or x.size(-1) != W:
            x = x[..., :H, :W]
            y = y[..., :H, :W]
        return x, y

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ddconfig = {
            "double_z": True,
            "z_channels": config["z_channels"],
            "resolution": config["resolution"],
            "in_channels": 1,
            "out_ch": 1,
            "ch": config["ch"],
            "ch_mult": config["ch_mult"],
            "num_res_blocks": config["num_res_blocks"],
            "attn_resolutions": config["attn_resolutions"],
            "dropout": config["dropout"],
        }

        lossconfig = {
            "target": "src.latent_diffusion.modules.losses.contperceptual.LPIPSWithDiscriminator",
            "params": {
                "disc_start": 999999999,  # Never start discriminator (disabled)
                "kl_weight": config["kl_weight"],
                "disc_weight": 0.0,
                "perceptual_weight": 0.0,
            },
        }

        print("\n" + "=" * 80)
        print("Creating AutoencoderKL (VAE)")
        print("=" * 80)
        print("  embed_dim:   {}".format(config["embed_dim"]))
        print("  mel_bands:   {}".format(config["mel_bands"]))
        print("  resolution:  {}".format(config["resolution"]))
        print("  kl_weight:   {}  (lower = more latent variance)".format(config["kl_weight"]))
        print("  ddconfig:    {}".format(ddconfig))
        print("  lossconfig:  {}".format(lossconfig["target"]))
        print("=" * 80 + "\n")

        self.model = AutoencoderKL(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=config["embed_dim"],
            mel_band=config["mel_bands"],
            image_key="fbank",
            base_learning_rate=config["learning_rate"],
        ).to(self.device)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=config["learning_rate"],
            betas=(0.9, 0.999),
        )

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config["epochs"])

        self.kl_weight = config["kl_weight"]

        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        num_params = sum(p.numel() for p in self.model.parameters())
        print("Model parameters: {:,}".format(num_params))
        print("Device: {}\n".format(self.device))

    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        pbar = tqdm(train_loader, desc="Epoch {}".format(epoch))

        for batch in pbar:
            mel = batch["mel"].to(self.device)

            reconstructed, posterior = self.model(mel)
            reconstructed, mel = self._align_hw(reconstructed, mel)

            recon_l1 = F.l1_loss(reconstructed, mel)
            recon_l2 = F.mse_loss(reconstructed, mel)
            recon_loss = recon_l1 + 0.5 * recon_l2

            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            loss = recon_loss + self.kl_weight * kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

            pbar.set_postfix(
                {
                    "loss": "{:.4f}".format(loss.item()),
                    "recon": "{:.4f}".format(recon_loss.item()),
                    "kl": "{:.4f}".format(kl_loss.item()),
                }
            )

        n = len(train_loader)
        return {
            "loss": total_loss / n,
            "recon_loss": total_recon_loss / n,
            "kl_loss": total_kl_loss / n,
        }

    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                mel = batch["mel"].to(self.device)

                reconstructed, posterior = self.model(mel)
                reconstructed, mel = self._align_hw(reconstructed, mel)

                recon_l1 = F.l1_loss(reconstructed, mel)
                recon_l2 = F.mse_loss(reconstructed, mel)
                recon_loss = recon_l1 + 0.5 * recon_l2

                kl_loss = posterior.kl()
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                loss = recon_loss + self.kl_weight * kl_loss

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

        n = len(val_loader)
        return {
            "val_loss": total_loss / n,
            "val_recon_loss": total_recon_loss / n,
            "val_kl_loss": total_kl_loss / n,
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        torch.save(checkpoint, self.output_dir / "autoencoder_latest.ckpt")

        if not hasattr(self, "best_val_loss") or metrics["val_loss"] < self.best_val_loss:
            self.best_val_loss = metrics["val_loss"]
            torch.save(checkpoint, self.output_dir / "autoencoder_best.ckpt")
            print(" Saved best model (val_loss: {:.4f})".format(metrics["val_loss"]))

        if (epoch + 1) % 10 == 0:
            torch.save(
                checkpoint,
                self.output_dir / "autoencoder_epoch_{}.ckpt".format(epoch + 1),
            )

    def train(self, train_loader, val_loader, epochs: int) -> None:
        """Full training loop."""
        print("\n" + "=" * 80)
        print("Starting Autoencoder Training")
        print("=" * 80)
        print("Epochs:         {}".format(epochs))
        print("Train batches:  {}".format(len(train_loader)))
        print("Val batches:    {}".format(len(val_loader)))
        print("=" * 80 + "\n")

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            self.scheduler.step()

            metrics = dict(train_metrics)
            metrics.update(val_metrics)

            print("\nEpoch {} Summary:".format(epoch))
            print("  Train Loss:  {:.4f}".format(metrics["loss"]))
            print("  Val Loss:    {:.4f}".format(metrics["val_loss"]))
            print("  Recon Loss:  {:.4f}".format(metrics["recon_loss"]))
            print("  KL Loss:     {:.4f}".format(metrics["kl_loss"]))
            print("  LR:          {:.6f}".format(self.scheduler.get_last_lr()[0]))

            self.save_checkpoint(epoch, metrics)
            print("=" * 80)


def main(config_path: Optional[str] = None) -> None:
    """Entry point."""
    global_cfg = load_global_config(config_path)
    ae_cfg = global_cfg["autoencoder"]
    paths_cfg = global_cfg["paths"]

    required_keys = [
        "embed_dim",
        "ch",
        "ch_mult",
        "num_res_blocks",
        "attn_resolutions",
        "dropout",
        "mel_bands",
        "resolution",
        "z_channels",
        "learning_rate",
        "kl_weight",
        "epochs",
        "batch_size",
        "output_dir",
    ]
    missing = [k for k in required_keys if k not in ae_cfg]
    if missing:
        raise KeyError(
            "Missing keys in config['autoencoder']: {}. Please add them to config_machines.yaml".format(
                missing
            )
        )

    data_root = paths_cfg.get("data_root", "data/processed_machines")
    num_workers = ae_cfg.get("num_workers", 4)

    config = dict(ae_cfg)
    config["data_root"] = data_root

    print("\n" + "=" * 80)
    print("VAE TRAINING (DCASE machines) - SHARED CONFIG VERSION")
    print("=" * 80)
    print("Config file:   {}".format(os.environ.get(CONFIG_ENV_VAR, CONFIG_DEFAULT_PATH)))
    print("Data root:     {}".format(data_root))
    print("Output dir:    {}".format(config["output_dir"]))
    print("Batch size:    {}".format(config["batch_size"]))
    print("Epochs:        {}".format(config["epochs"]))
    print("=" * 80 + "\n")

    train_loader, val_loader = get_dataloaders(
        batch_size=config["batch_size"],
        num_workers=num_workers,
        data_root=data_root,
    )

    # print("Train samples: {}".format(len(train_loader.dataset)))
    # print("Val samples:   {}".format(len(val_loader.dataset)))
    # print("")

    trainer = VAETrainer(config)
    trainer.train(train_loader, val_loader, config["epochs"])

    print("\n" + "=" * 80)
    print("Autoencoder training complete!")
    print("Checkpoints saved to {}".format(config["output_dir"]))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
