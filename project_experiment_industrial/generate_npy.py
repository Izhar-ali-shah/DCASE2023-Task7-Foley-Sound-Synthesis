#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np

import torch

ROOT = os.path.dirname(__file__)
CONFIG_DEFAULT_PATH = os.path.join(ROOT, "config_machines.yaml")
CONFIG_ENV_VAR = "MACHINE_CONFIG_PATH"

sys.path.insert(0, os.path.join(ROOT, "src"))

from src.latent_encoder.autoencoder import AutoencoderKL
from laion_clap import CLAP_Module

# Import FiLM diffusion classes from training script
from train_diffusion_with_film import ConditionedLatentDiffusion, DummyConditioningModel


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

    if "generation" not in cfg:
        raise KeyError("Config is missing top-level 'generation' section.")
    if "diffusion" not in cfg:
        raise KeyError("Config is missing top-level 'diffusion' section.")
    if "autoencoder" not in cfg:
        raise KeyError("Config is missing top-level 'autoencoder' section.")
    if "paths" not in cfg:
        cfg["paths"] = {}

    return cfg


class FiLMAudioGenerator(object):
    """Generate decoder outputs (mel tensors)"""

    def __init__(
        self,
        diffusion_ckpt_path: str,
        device: str = "cuda",
        clap_ckpt_path: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print("Loading Generator (decoder -> .npy)")

        # Load diffusion checkpoint (includes config + latent statistics)
        checkpoint = torch.load(diffusion_ckpt_path, map_location="cpu")
        self.config = checkpoint["config"]
        self.ref_mean = checkpoint["ref_mean"]
        self.ref_std = checkpoint["ref_std"]
        print(" Checkpoint loaded (epoch {})".format(checkpoint["epoch"]))
        print("   Conditioning: FiLM")

        # Load VAE (AutoencoderKL for mel-spectrograms)
        vae_checkpoint = torch.load(self.config["vae_path"], map_location="cpu")
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

        self.vae = AutoencoderKL(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=vae_config["embed_dim"],
            mel_band=vae_config["mel_bands"],
            image_key="fbank",
        ).to(self.device)
        self.vae.load_state_dict(vae_checkpoint["model_state_dict"])
        self.vae.eval()
        print(" VAE loaded")

         
        # Infer latent dimensions from a dummy mel
         
        with torch.no_grad():
            dummy_mel = torch.randn(1, 1, 80, 344).to(self.device)
            dummy_latent = self.vae.encode(dummy_mel).sample()
            self.latent_channels = dummy_latent.shape[1]
            self.latent_h, self.latent_w = dummy_latent.shape[2], dummy_latent.shape[3]

        self.pad_h = (4 - self.latent_h % 4) % 4
        self.pad_w = (4 - self.latent_w % 4) % 4
        self.latent_h_padded = self.latent_h + self.pad_h
        self.latent_w_padded = self.latent_w + self.pad_w

        print(
            "   Latent shape: [{}, {}, {}]".format(
                self.latent_channels, self.latent_h, self.latent_w
            )
        )

        # CLAP for text conditioning (OFFLINE)
         
        # def _offline_load_ckpt(self_clap, ckpt_path: Optional[str] = None):
        #     """
        #     Replacement for CLAP_Module.load_ckpt that never uses the internet
        #     and loads weights from a local .pt file.
        #     """
        #     if ckpt_path is None:
        #         if clap_ckpt_path is not None:
        #             ckpt_path_local = clap_ckpt_path
        #         else:
        #             ckpt_path_local = os.path.join(
        #                 ROOT, "clap_ckpt", "630k-audioset-best.pt"
        #             )
        #     else:
        #         ckpt_path_local = ckpt_path

        #     if not os.path.isfile(ckpt_path_local):
        #         raise FileNotFoundError(
        #             "CLAP checkpoint not found: {}".format(ckpt_path_local)
        #         )

        #     print("[OFFLINE] Loading CLAP weights from: {}".format(ckpt_path_local))
        #     ckpt = torch.load(ckpt_path_local, map_location="cpu")

        #     if isinstance(ckpt, dict) and "state_dict" in ckpt:
        #         state = ckpt["state_dict"]
        #     elif isinstance(ckpt, dict) and "model" in ckpt:
        #         state = ckpt["model"]
        #     else:
        #         state = ckpt

        #     cleaned_state = {}
        #     for k, v in state.items():
        #         if k.startswith("module."):
        #             k = k[len("module.") :]
        #         cleaned_state[k] = v

        #     incompatible = self_clap.model.load_state_dict(cleaned_state, strict=False)

        #     try:
        #         missing = incompatible.missing_keys
        #         unexpected = incompatible.unexpected_keys
        #         if missing:
        #             print(
        #                 "[OFFLINE] Warning: {} missing keys when loading CLAP (showing first 5): {}".format(
        #                     len(missing), missing[:5]
        #                 )
        #             )
        #         if unexpected:
        #             print(
        #                 "[OFFLINE] Warning: {} unexpected keys when loading CLAP (showing first 5): {}".format(
        #                     len(unexpected), unexpected[:5]
        #                 )
        #             )
        #     except Exception:
        #         pass

        #     print("[OFFLINE] CLAP weights loaded")

        # CLAP_Module.load_ckpt = _offline_load_ckpt

        # self.clap = CLAP_Module(
        #     enable_fusion=False, device=self.device, amodel="HTSAT-tiny"
        # )
        # self.clap.load_ckpt()
        # self.clap.model.eval()
        # print("CLAP loaded (offline)")
        

        # CLAP for text conditioning (online)

        self.clap = CLAP_Module(
            enable_fusion=False, device=self.device, amodel="HTSAT-tiny"
        )

        # If you have a local ckpt and want to use it:
        if clap_ckpt_path is not None and os.path.isfile(clap_ckpt_path):
            print(f"Loading CLAP weights from local checkpoint: {clap_ckpt_path}")
            self.clap.load_ckpt(clap_ckpt_path)
        else:
            print("No local CLAP checkpoint; using laion_clap's default online loading.")
            self.clap.load_ckpt()   # HF cache

        self.clap.model.eval()
        print("CLAP loaded")
        
        # Load diffusion model
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
            "target": "train_diffusion_with_film.DummyConditioningModel",
            "params": {},
        }

        unet_config = {
            "target": "src.latent_diffusion.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {
                "image_size": self.latent_h_padded,
                "in_channels": vae_config["embed_dim"],
                "out_channels": vae_config["embed_dim"],
                "model_channels": 128,
                "attention_resolutions": [4, 2],
                "num_res_blocks": self.config["num_res_blocks"],
                "channel_mult": [1, 2, 4],
                "num_heads": 8,
                "use_spatial_transformer": False,
                "extra_film_condition_dim": 512,
                "extra_film_use_concat": False,
                "dropout": self.config["dropout"],
            },
        }

        self.model = ConditionedLatentDiffusion(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            unet_config=unet_config,
            timesteps=self.config["num_timesteps"],
            channels=vae_config["embed_dim"],
            conditioning_key="film",
            cond_stage_trainable=False,
            latent_t_size=self.latent_h_padded,
            latent_f_size=self.latent_w_padded,
            base_learning_rate=self.config["lr"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(" Diffusion model with FiLM loaded")

        print("\n" + "=" * 80)
        print(" FiLM Generator Ready (decoder -> .npy only)!")
        print("=" * 80 + "\n")

     
    # Conditioning helpers
     
    def _get_text_embedding(self, text_prompts: List[str]) -> torch.Tensor:
        """Get CLAP embeddings for a list of text prompts."""
        with torch.no_grad():
            text_embeds = self.clap.get_text_embedding(text_prompts, use_tensor=True)
            if isinstance(text_embeds, np.ndarray):
                text_embeds = torch.from_numpy(text_embeds).to(self.device)
            if text_embeds.dim() == 2:
                text_embeds = text_embeds.unsqueeze(1)  # [B, 1, D]
            return text_embeds

     
    # DDIM sampling with FiLM + classifier-free guidance
     
    def _ddim_sample_film_cfg(
        self,
        text_embedding: torch.Tensor,
        num_samples: int = 1,
        ddim_steps: int = 50,
        eta: float = 0.0,
        cfg_scale: float = 3.0,
    ) -> torch.Tensor:
        """
        DDIM sampling with FiLM conditioning and classifier-free guidance (CFG).
        """
        if text_embedding.dim() == 2:  # [B, 512]
            text_embedding = text_embedding.unsqueeze(1)  # -> [B, 1, 512]

        text_embedding_expanded = text_embedding.expand(num_samples, -1, -1)
        zeros_expanded = torch.zeros_like(text_embedding_expanded)

        c_cond = {"c_film": [text_embedding_expanded]}
        c_uncond = {"c_film": [zeros_expanded]}

        x = torch.randn(
            num_samples,
            self.latent_channels,
            self.latent_h_padded,
            self.latent_w_padded,
            device=self.device,
        )

        timesteps = np.linspace(
            self.model.num_timesteps - 1, 0, ddim_steps, dtype=int
        )

        noise_pred_info: List[Dict[str, float]] = []

        for i, t in enumerate(timesteps):
            t_batch = torch.full(
                (num_samples,), int(t), device=self.device, dtype=torch.long
            )

            with torch.no_grad():
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t_batch, t_batch], dim=0)

                film_context = torch.cat(
                    [c_cond["c_film"][0], c_uncond["c_film"][0]], dim=0
                )
                cond_in = {"c_film": [film_context]}

                noise_pred = self.model.apply_model(x_in, t_in, cond_in)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)

                if i < 5:
                    diff = (noise_pred_cond - noise_pred_uncond).abs().mean().item()
                    noise_pred_info.append(
                        {
                            "step": float(i),
                            "cond_mean": float(noise_pred_cond.mean().item()),
                            "uncond_mean": float(noise_pred_uncond.mean().item()),
                            "diff": float(diff),
                        }
                    )

                noise_pred = noise_pred_uncond + cfg_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            alpha_t = self.model.alphas_cumprod[int(t)]
            if i < len(timesteps) - 1:
                alpha_t_prev = self.model.alphas_cumprod[int(timesteps[i + 1])]
            else:
                alpha_t_prev = torch.tensor(1.0, device=self.device)

            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            x_0_pred = (x - sqrt_one_minus_alpha_t * noise_pred) / torch.sqrt(alpha_t)

            dir_xt = torch.sqrt(
                1.0
                - alpha_t_prev
                - eta ** 2
                * (1.0 - alpha_t_prev)
                / (1.0 - alpha_t)
                * (1.0 - alpha_t / alpha_t_prev)
            ) * noise_pred

            x = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt

            if eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x)
                sigma_t = eta * torch.sqrt(
                    (1.0 - alpha_t_prev)
                    / (1.0 - alpha_t)
                    * (1.0 - alpha_t / alpha_t_prev)
                )
                x = x + sigma_t * noise

        if noise_pred_info:
            avg_diff = float(np.mean([info["diff"] for info in noise_pred_info]))
            print("    Average difference: {:.4f}".format(avg_diff))

        x = x * self.ref_std + self.ref_mean

        if self.pad_h > 0 or self.pad_w > 0:
            x = x[:, :, : self.latent_h, : self.latent_w]

        return x

     
    # VAE decode: latent -> mel (normalized)
     
    def _decode_to_mel(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to normalized mel-spectrograms ([-1, 1]), shape [B, 1, 80, T]."""
        with torch.no_grad():
            mel = self.vae.decode(latents)
        return mel
     
    def generate(
        self,
        text_prompt: str,
        num_samples: int = 1,
        ddim_steps: int = 50,
        eta: float = 0.0,
        cfg_scale: float = 3.0,
        npy_root: str = "generated_npy",
        start_index: int = 0,
    ) -> Tuple[np.ndarray, List[str]]:
 
        print("Generating decoder outputs .npy")
        print('Prompt (class): "{}"'.format(text_prompt))
        print("Samples: {}".format(num_samples))
        print("DDIM steps: {}".format(ddim_steps))

        class_dir_name = text_prompt.strip().replace(os.sep, "_")
        npy_dir = Path(npy_root) / class_dir_name
        npy_dir.mkdir(parents=True, exist_ok=True)

        # print("Getting text embedding...")
        text_embedding = self._get_text_embedding([text_prompt])
        # print(
        #     "  Text embedding mean: {:.4f}, std: {:.4f}".format(
        #         text_embedding.mean().item(), text_embedding.std().item()
        #     )
        # )

        # print("\nGenerating latents")
        latents = self._ddim_sample_film_cfg(
            text_embedding, num_samples, ddim_steps, eta, cfg_scale
        )

        # print("\nDecoding to mel (VAE)...")
        mel = self._decode_to_mel(latents)
        # print(
        #     "  Generated mel mean: {:.4f}, std: {:.4f}".format(
        #         mel.mean().item(), mel.std().item()
        #     )
        # )

        mel_cpu = mel.detach().cpu().numpy()  # [B, 1, 80, T]
        
        print(" shape:", mel_cpu.shape)
        
        saved_npy: List[str] = []

        for i in range(num_samples):
            global_idx = start_index + i
            idx_str = "{:03d}".format(global_idx + 1)

            npy_filename = "sample_{}.npy".format(idx_str, cfg_scale)
            npy_filepath = npy_dir / npy_filename

            np.save(npy_filepath, mel_cpu[i])
            saved_npy.append(str(npy_filepath))

        return mel_cpu, saved_npy


def main(config_path: Optional[str] = None) -> None:

    cfg = load_global_config(config_path)
    gen_cfg = cfg["generation"]
    diff_cfg = cfg["diffusion"]
    paths_cfg = cfg.get("paths", {})

    prompts: List[str] = gen_cfg["prompts"]
    num_per_class: int = gen_cfg["num_per_class"]
    batch_size: int = gen_cfg["batch_size"]
    cfg_scale: float = gen_cfg["cfg_scale"]
    ddim_steps: int = gen_cfg["ddim_steps"]
    eta: float = gen_cfg["eta"]
    device: str = gen_cfg.get("device", "cuda")
    sample_rate: int = gen_cfg.get("sample_rate", 22050)  # unused but kept for consistency
    npy_root: str = gen_cfg.get("npy_root", "generated_npy")

    diffusion_ckpt = gen_cfg.get("diffusion_ckpt")
    if diffusion_ckpt is None:
        diff_out = diff_cfg.get(
            "output_dir", "training_workspace/checkpoints/diffusion_machines_film"
        )
        diffusion_ckpt = os.path.join(diff_out, "diffusion_film_best.ckpt")

    clap_ckpt_path = paths_cfg.get("clap_ckpt")

    print("Config file:       {}".format(os.environ.get(CONFIG_ENV_VAR, CONFIG_DEFAULT_PATH)))
    print("Diffusion ckpt:    {}".format(diffusion_ckpt))
    print("Device:            {}".format(device))
    print("Classes:           {}".format(len(prompts)))
    print("Samples per class: {}".format(num_per_class))
    print("Batch size:        {}".format(batch_size))

    print("DDIM steps:        {}".format(ddim_steps))
    print("Sample rate (info):{}".format(sample_rate))
    print("Output NPY root:   {}".format(npy_root))
    print("=" * 80 + "\n")

    generator = FiLMAudioGenerator(
        diffusion_ckpt_path=diffusion_ckpt,
        device=device,
        clap_ckpt_path=clap_ckpt_path,
    )

    for prompt in prompts:
        print("\n--- Generating class: {} ---".format(prompt))
        start_idx = 0
        while start_idx < num_per_class:
            cur_bs = min(batch_size, num_per_class - start_idx)
            print("  Batch starting at {}, size {}".format(start_idx, cur_bs))

            generator.generate(
                text_prompt=prompt,
                num_samples=cur_bs,
                ddim_steps=ddim_steps,
                eta=eta,
                cfg_scale=cfg_scale,
                npy_root=npy_root,
                start_index=start_idx,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            start_idx += cur_bs

    print("ALL CLASSES GENERATED")

if __name__ == "__main__":
    main()
