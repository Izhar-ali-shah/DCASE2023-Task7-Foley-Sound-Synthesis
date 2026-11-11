#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = os.path.dirname(__file__)
CONFIG_DEFAULT_PATH = os.path.join(ROOT, "config_machines.yaml")
CONFIG_ENV_VAR = "MACHINE_CONFIG_PATH"

# repo imports
sys.path.insert(0, os.path.join(ROOT, "src"))

from dataset_machines import get_dataloaders, MachineDataset
from src.latent_diffusion.models.ddpm import LatentDiffusion
from src.latent_diffusion.util import default
from laion_clap import CLAP_Module
from src.latent_encoder.autoencoder import AutoencoderKL


def load_global_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the shared YAML config.
    """
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

    if "diffusion" not in cfg:
        raise KeyError("Config is missing top-level 'diffusion' section.")
    if "autoencoder" not in cfg:
        raise KeyError("Config is missing top-level 'autoencoder' section.")
    if "paths" not in cfg:
        raise KeyError("Config is missing top-level 'paths' section (for data_root, etc.).")

    return cfg


class ConditionedLatentDiffusion(LatentDiffusion):
    """Extended LatentDiffusion with proper FiLM-based conditioning."""

    def get_loss(self, pred, target, mean=True):
        loss_type = getattr(self, "loss_type", "l2")
        if loss_type == "l2":
            loss = F.mse_loss(pred, target, reduction="none")
        elif loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction="none")
        else:
            raise NotImplementedError("Unknown loss_type: {}".format(loss_type))
        if mean:
            return loss.mean(dim=list(range(1, loss.ndim)))
        return loss

    def p_losses(self, x_start, cond, t, noise=None):
        """Override p_losses to handle FiLM conditioning."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if cond.dim() == 2:  # [B, 512] from CLAP
            cond = cond.unsqueeze(1)  # -> [B, 1, 512]
        cond_dict = {"c_film": [cond]}

        model_out = self.apply_model(x_noisy, t, cond_dict)

        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(
                "Parameterization {} not supported".format(self.parameterization)
            )

        per_sample = self.get_loss(model_out, target, mean=False).mean([1, 2, 3])

        log_prefix = "train" if self.training else "val"
        loss_dict = {"{}/loss_simple".format(log_prefix): per_sample.mean()}

        loss_simple = per_sample.mean() * self.l_simple_weight
        loss_vlb = (self.lvlb_weights[t] * per_sample).mean()
        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update(
            {"{}/loss_vlb".format(log_prefix): loss_vlb, "{}/loss".format(log_prefix): loss}
        )
        return loss, loss_dict


class DummyConditioningModel(nn.Module):
    """Pass-through conditioning model."""

    def __init__(self):
        super(DummyConditioningModel, self).__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x

    def encode(self, x):
        return x


class FiLMDiffusionTrainer(object):
    """Trainer for LatentDiffusion with FiLM conditioning."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("\n" + "=" * 80)
        print("Creating Diffusion Model with FiLM Conditioning")
        print("=" * 80)

        
        # Load VAE checkpoint and infer latent shape
        
        vae_checkpoint = torch.load(config["vae_path"], map_location="cpu")
        vae_config = vae_checkpoint["config"]

        ddconfig = {
            "double_z": True,
            "z_channels": vae_config["z_channels"],
            "resolution": vae_config["resolution"],
            "in_channels": 1,
            "out_ch": 1,
            "ch": vae_config["ch"],
            "ch_mult": vae_config["ch_mult"],
            "num_res_blocks": vae_config["num_res_blocks"],
            "attn_resolutions": vae_config["attn_resolutions"],
            "dropout": vae_config["dropout"],
        }

        lossconfig = {
            "target": "src.latent_diffusion.modules.losses.contperceptual.LPIPSWithDiscriminator",
            "params": {
                "disc_start": 999999999,
                "kl_weight": vae_config["kl_weight"],
                "disc_weight": 0.0,
                "perceptual_weight": 0.0,
            },
        }

        temp_vae = AutoencoderKL(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=vae_config["embed_dim"],
            mel_band=vae_config["mel_bands"],
            image_key="fbank",
        ).to(self.device)

        with torch.no_grad():
            dummy_mel = torch.randn(1, 1, 80, 344).to(self.device)
            dummy_latent = temp_vae.encode(dummy_mel).sample()
            latent_h, latent_w = dummy_latent.shape[2], dummy_latent.shape[3]

        del temp_vae
        torch.cuda.empty_cache()

        self.pad_h = (4 - latent_h % 4) % 4
        self.pad_w = (4 - latent_w % 4) % 4
        latent_h_padded = latent_h + self.pad_h
        latent_w_padded = latent_w + self.pad_w

        print(
            "   Latent shape: [B, {}, {}, {}]".format(
                vae_config["embed_dim"], latent_h, latent_w
            )
        )
        if self.pad_h > 0 or self.pad_w > 0:
            print(
                "   Padded shape: [B, {}, {}, {}]".format(
                    vae_config["embed_dim"], latent_h_padded, latent_w_padded
                )
            )

        
        # Load CLAP (standard laion_clap behaviour)
        
        print("\nLoading CLAP (HTSAT-tiny)...")
        self.clap = CLAP_Module(enable_fusion=False, device=self.device, amodel="HTSAT-tiny")
        self.clap.load_ckpt()
        self.clap.model.eval()
        print(" CLAP loaded (512-dim embeddings)")

        
        # Build LatentDiffusion config with FiLM UNet
        
        first_stage_config = {
            "target": "src.latent_encoder.autoencoder.AutoencoderKL",
            "params": {
                "ddconfig": ddconfig,
                "lossconfig": lossconfig,
                "embed_dim": vae_config["embed_dim"],
                "mel_band": vae_config["mel_bands"],
                "image_key": "fbank",
            },
        }

        cond_stage_config = {
            "target": "__main__.DummyConditioningModel",
            "params": {},
        }

        unet_config = {
            "target": "src.latent_diffusion.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {
                "image_size": latent_h_padded,
                "in_channels": vae_config["embed_dim"],
                "out_channels": vae_config["embed_dim"],
                "model_channels": 128,
                "attention_resolutions": [4, 2],
                "num_res_blocks": config["num_res_blocks"],
                "channel_mult": [1, 2, 4],
                "num_heads": 8,
                "dropout": config["dropout"],
                # FiLM conditioning (stronger than cross-attention)
                "extra_film_condition_dim": 512,
                "extra_film_use_concat": False,
                "use_spatial_transformer": False,
            },
        }

        print("\nCreating model with FiLM conditioning...")
        print("  FiLM condition dim: 512 (CLAP)")
        print("  FiLM mode: Addition (not concatenation)")
        print("  CFG uncond_prob: {}".format(config["cfg_uncond_prob"]))

        self.model = ConditionedLatentDiffusion(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            unet_config=unet_config,
            timesteps=config["num_timesteps"],
            channels=vae_config["embed_dim"],
            conditioning_key="film",
            cond_stage_trainable=False,
            latent_t_size=latent_h_padded,
            latent_f_size=latent_w_padded,
            base_learning_rate=config["lr"],
        ).to(self.device)

        self.model.first_stage_model.load_state_dict(vae_checkpoint["model_state_dict"])
        self.model.first_stage_model.eval()
        for p in self.model.first_stage_model.parameters():
            p.requires_grad = False

        for p in self.model.cond_stage_model.parameters():
            p.requires_grad = False

        print(" Diffusion model created")

        
        # Compute reference latent statistics (mean/std)
        
        print("\nComputing reference latent statistics...")
        self.ref_mean, self.ref_std = self._compute_reference_stats(
            config["data_root"], sample_size=config.get("stats_sample_size", 500)
        )
        # print(" Reference stats: mean={:.4f}, std={:.4f}".format(self.ref_mean, self.ref_std))

        
        # Optimizer & scheduler
        
        diffusion_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = Adam(diffusion_params, lr=config["lr"], betas=(0.9, 0.999))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config["epochs"])

        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n Trainer initialized with FiLM")
        print("   Unconditional probability: {}".format(config["cfg_uncond_prob"]))
        print("   Trainable parameters: {:,}".format(sum(p.numel() for p in diffusion_params)))
        print("=" * 80 + "\n")

    def _compute_reference_stats(self, data_root: str, sample_size: int = 500):
        dataset = MachineDataset(
            metadata_path=os.path.join(data_root, "train_metadata.json"),
            data_root=data_root,
        )

        sample_size = min(sample_size, len(dataset))
        indices = np.random.choice(len(dataset), sample_size, replace=False)

        latents = []
        with torch.no_grad():
            for idx in tqdm(indices, desc="Computing stats"):
                mel = dataset[idx]["mel"].unsqueeze(0).to(self.device)
                posterior = self.model.encode_first_stage(mel)
                z = self.model.get_first_stage_encoding(posterior)
                latents.append(z.cpu())

        latents = torch.cat(latents, dim=0)
        return latents.mean().item(), latents.std().item()

    def _get_text_embedding(self, text_prompts):
        """Get CLAP embeddings (flattened to [B, 512])."""
        with torch.no_grad():
            text_embeds = self.clap.get_text_embedding(text_prompts, use_tensor=True)
            if isinstance(text_embeds, np.ndarray):
                text_embeds = torch.from_numpy(text_embeds).to(self.device)
            if text_embeds.dim() == 3:
                text_embeds = text_embeds.squeeze(1)
            return text_embeds

    def _step(self, batch):
        mel = batch["mel"].to(self.device)
        text_prompts = batch["text_prompt"]
        batch_size = mel.shape[0]

        with torch.no_grad():
            posterior = self.model.encode_first_stage(mel)
            z = self.model.get_first_stage_encoding(posterior)

        z_std = (z - self.ref_mean) / (self.ref_std + 1e-6)

        _, _, h, w = z_std.shape
        if self.pad_h > 0 or self.pad_w > 0:
            z_std = F.pad(
                z_std,
                (0, self.pad_w, 0, self.pad_h),
                mode="constant",
                value=0.0,
            )

        c = self._get_text_embedding(text_prompts)

        cfg_uncond_prob = self.config["cfg_uncond_prob"]
        cfg_mask = torch.rand(batch_size, device=self.device) < cfg_uncond_prob
        if cfg_mask.any():
            c_uncond = torch.zeros_like(c)
            c = torch.where(cfg_mask.unsqueeze(1), c_uncond, c)

        result = self.model(z_std, c)

        if isinstance(result, tuple):
            loss, loss_dict = result
            main_loss = loss_dict.get("train/loss", loss_dict.get("train/loss_simple", loss.item()))
        else:
            loss = result
            main_loss = loss.item()

        return loss, main_loss

    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.model.first_stage_model.eval()
        self.model.cond_stage_model.eval()

        total_loss = 0.0
        total_main_loss = 0.0

        pbar = tqdm(train_loader, desc="Epoch {}".format(epoch))

        for batch in pbar:
            loss, main_loss = self._step(batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_main_loss += main_loss

            pbar.set_postfix({"loss": "{:.4f}".format(loss.item()), "main": "{:.4f}".format(main_loss)})

        n = len(train_loader)
        return {
            "loss": total_loss / n,
            "main_loss": total_main_loss / n,
        }

    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the diffusion model."""
        self.model.eval()

        total_loss = 0.0
        total_main_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                loss, main_loss = self._step(batch)
                total_loss += loss.item()
                total_main_loss += main_loss

        n = len(val_loader)
        return {
            "val_loss": total_loss / n,
            "val_main_loss": total_main_loss / n,
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save diffusion checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "ref_mean": self.ref_mean,
            "ref_std": self.ref_std,
            "metrics": metrics,
            "config": self.config,
        }

        torch.save(checkpoint, self.output_dir / "diffusion_film_latest.ckpt")

        if not hasattr(self, "best_val_loss") or metrics["val_loss"] < self.best_val_loss:
            self.best_val_loss = metrics["val_loss"]
            torch.save(checkpoint, self.output_dir / "diffusion_film_best.ckpt")
            print(" Saved best model (val_loss: {:.4f})".format(metrics["val_loss"]))

        if (epoch + 1) % 10 == 0:
            torch.save(
                checkpoint,
                self.output_dir / "diffusion_film_epoch_{}.ckpt".format(epoch + 1),
            )

    def train(self, train_loader, val_loader, epochs: int) -> None:
        """Full training loop."""
        print("\n" + "=" * 80)
        print("Starting FiLM Training")
        print("=" * 80)
        print("Epochs:              {}".format(epochs))
        # print("CFG uncond prob:     {}".format(self.config["cfg_uncond_prob"]))
        print("Conditioning:        FiLM ")
        print("=" * 80 + "\n")

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            self.scheduler.step()

            metrics = dict(train_metrics)
            metrics.update(val_metrics)

            print("\nEpoch {} Summary:".format(epoch))
            print("  Train Loss:   {:.4f}".format(metrics["loss"]))
            print("  Val Loss:     {:.4f}".format(metrics["val_loss"]))
            print("  LR:           {:.6f}".format(self.scheduler.get_last_lr()[0]))

            self.save_checkpoint(epoch, metrics)
            print("=" * 80)


def main(config_path: Optional[str] = None) -> None:
    """Script entry point."""
    global_cfg = load_global_config(config_path)
    diff_cfg = global_cfg["diffusion"]
    ae_cfg = global_cfg["autoencoder"]
    paths_cfg = global_cfg["paths"]

    required_keys = [
        "num_res_blocks",
        "dropout",
        "num_timesteps",
        "lr",
        "epochs",
        "batch_size",
        "output_dir",
    ]
    missing = [k for k in required_keys if k not in diff_cfg]
    if missing:
        raise KeyError(
            "Missing keys in config['diffusion']: {}. Please add them to config_machines.yaml".format(
                missing
            )
        )

    data_root = paths_cfg.get("data_root", "data/processed_machines")

    vae_path = diff_cfg.get("vae_path")
    if vae_path is None:
        # Default: best checkpoint from autoencoder output dir
        ae_out = ae_cfg.get(
            "output_dir", "training_workspace/checkpoints/autoencoder_machines"
        )
        vae_path = os.path.join(ae_out, "autoencoder_best.ckpt")

    num_workers = diff_cfg.get("num_workers", 4)
    cfg_uncond_prob = diff_cfg.get("cfg_uncond_prob", 0.1)
    stats_sample_size = diff_cfg.get("stats_sample_size", 500)

    config = dict(diff_cfg)
    config["vae_path"] = vae_path
    config["data_root"] = data_root
    config["cfg_uncond_prob"] = cfg_uncond_prob
    config["stats_sample_size"] = stats_sample_size

    print("\n" + "=" * 80)
    print("DIFFUSION TRAINING WITH FiLM CONDITIONING - SHARED CONFIG VERSION")
    print("=" * 80)
    print("Config file:   {}".format(os.environ.get(CONFIG_ENV_VAR, CONFIG_DEFAULT_PATH)))
    print("Data root:     {}".format(data_root))
    print("VAE ckpt:      {}".format(config["vae_path"]))
    print("Output dir:    {}".format(config["output_dir"]))
    print("Batch size:    {}".format(config["batch_size"]))
    print("Epochs:        {}".format(config["epochs"]))
    print("Timesteps:     {}".format(config["num_timesteps"]))
    print("CFG p(uncond): {}".format(config["cfg_uncond_prob"]))
    print("=" * 80 + "\n")

    train_loader, val_loader = get_dataloaders(
        batch_size=config["batch_size"],
        num_workers=num_workers,
        data_root=data_root,
    )

    print(" Train samples: {}".format(len(train_loader.dataset)))
    print(" Val samples:   {}".format(len(val_loader.dataset)))
    print("")

    trainer = FiLMDiffusionTrainer(config)
    trainer.train(train_loader, val_loader, config["epochs"])

    print("\n" + "=" * 80)
    print(" FiLM diffusion training complete!")
    print(" Checkpoints saved to {}".format(config["output_dir"]))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
