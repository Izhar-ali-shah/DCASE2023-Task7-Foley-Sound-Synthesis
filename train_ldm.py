#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, importlib, argparse, re
from typing import Dict, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn.functional as F

try:
    import yaml
except Exception:
    yaml = None

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(REPO_DIR, "src")
for p in (SRC_DIR, REPO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy   # <-- use explicit DDP strategy

# ---------------- utils ----------------
def instantiate_from_config(config: Dict[str, Any]):
    if "target" not in config:
        raise KeyError("Expected key 'target' in config to instantiate module.")
    module_path, cls_name = config["target"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    params = config.get("params", {})
    return cls(**params)

def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config files.")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def freeze_module(m: nn.Module):
    if m is None: return
    for p in m.parameters():
        p.requires_grad = False
    m.eval()

def _pl_major_minor():
    m = re.match(r"^(\d+)\.(\d+)", pl.__version__)
    return (int(m.group(1)), int(m.group(2))) if m else (1, 0)

def normalize_precision_arg(arg):
    s = str(arg).lower() if arg is not None else None
    major, _ = _pl_major_minor()
    if major >= 2:
        if s in {"16", "fp16", "half"}: return "16-mixed"
        if s in {"bf16", "bfloat16"}:   return "bf16-mixed"
        if s in {"32", "fp32"}:         return "32-true"
        if s in {"64", "fp64"}:         return "64-true"
        return "32-true"
    else:
        if s in {"16", "fp16", "half"}: return 16
        if s in {"64", "fp64"}:         return 64
        return 32

# ---------------- dataset ----------------
class DCASEMelDataset(Dataset):
    def __init__(self, json_path: str, cfg: Dict[str, Any]):
        with open(json_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        pp = cfg.get("preprocess_config", {}).get("preprocessing", {})
        audio = pp.get("audio", {})
        stft  = pp.get("stft", {})
        mel   = pp.get("mel", {})

        self.sr   = int(audio.get("sampling_rate", 22050))
        self.seg  = int(audio.get("segment_size", self.sr * 4))
        self.nfft = int(stft.get("filter_length", 1024))
        self.hop  = int(stft.get("hop_length", 256))
        self.win  = int(stft.get("win_length", self.nfft))
        self.nmel = int(mel.get("n_mel_channels", 80))
        self.fmin = float(mel.get("mel_fmin", 0.0))
        fmaxv     = mel.get("mel_fmax", None)
        self.fmax = float(fmaxv) if fmaxv is not None else None
        self.Ttar = int(mel.get("target_length", 344))

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=self.nfft, hop_length=self.hop, win_length=self.win,
            f_min=self.fmin, f_max=self.fmax, n_mels=self.nmel, power=1.0, normalized=False,
            center=True, pad_mode="reflect",
        )

    def __len__(self): return len(self.items)

    def _load_wav_1T(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # [C,T]
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        T = wav.shape[-1]
        if T < self.seg:
            wav = F.pad(wav, (0, self.seg - T))
        else:
            wav = wav[..., :self.seg]
        return wav  # [1,T]

    def __getitem__(self, i: int):
        rec  = self.items[i]
        path = rec["wav"]
        text = rec["text"]
        wav  = self._load_wav_1T(path)        # [1,T]
        mel  = self.melspec(wav).clamp(min=1e-5).log()  # [1,n_mel,Tm]
        mel  = mel.squeeze(0)                 # [n_mel,Tm]
        if mel.size(1) < self.Ttar:
            mel = F.pad(mel, (0, 0, 0, self.Ttar - mel.size(1)))
        else:
            mel = mel[:, :self.Ttar]
        mel = mel.transpose(0, 1).unsqueeze(0)  # [1,Ttar,n_mel]
        return {"fbank": mel, "text": text, "wav_path": path}

def make_loaders(cfg: Dict[str, Any]):
    pth  = cfg.get("preprocess_config", {}).get("path", {})
    train_json = pth["train_data"]
    test_json  = pth["test_data"]
    bs = int(cfg.get("model", {}).get("params", {}).get("batchsize", 8))
    nw = int(os.environ.get("NUM_WORKERS", "4"))

    dtr = DCASEMelDataset(train_json, cfg)
    dvl = DCASEMelDataset(test_json,  cfg)

    def collate(b):
        fbanks = torch.stack([x["fbank"] for x in b], 0)  # [B,1,T,M]
        texts  = [x["text"] for x in b]
        return {"fbank": fbanks, "text": texts}

    train_loader = DataLoader(dtr, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(dvl, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True, collate_fn=collate)
    return train_loader, val_loader

# ---------------- build raw diffusion model ----------------
def build_model(cfg: Dict[str, Any]):
    model_cfg = cfg["model"]
    params    = model_cfg.get("params", {}) or {}

    first_stage = params.get("first_stage_config", model_cfg.get("first_stage_config"))
    cond_stage  = params.get("cond_stage_config",  model_cfg.get("cond_stage_config"))
    inst_conf   = {"target": model_cfg["target"], "params": {**params, "first_stage_config": first_stage, "cond_stage_config": cond_stage}}
    model       = instantiate_from_config(inst_conf)

    ckpt_path = model_cfg.get("reload_from_ckpt") or model_cfg.get("ckpt_path") or model_cfg.get("resume")
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for k in ("state_dict_ema", "state_dict", "model_state_dict", "model"):
            if isinstance(ckpt, dict) and k in ckpt:
                ckpt = ckpt[k]; break
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"[load] diffusion ckpt: {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")

    # Freeze VAE + CLAP
    freeze_module(getattr(model, "first_stage_model", None))
    freeze_module(getattr(model, "cond_stage_model",  None))

    lr = float(params.get("base_learning_rate", 1e-5))
    if hasattr(model, "learning_rate"):
        model.learning_rate = lr
    return model

# ---------------- Lightning wrapper ----------------
class LDMFinetuneModule(pl.LightningModule):
    def __init__(self, ldm, cfg: Dict[str, Any]):
        super().__init__()
        self.ldm = ldm
        self.cfg = cfg
        # keep only trainable params (should be UNet)
        self.save_hyperparameters(ignore=["ldm", "cfg"])

        # tiny sanity: make sure first/cond are frozen
        for mname in ["first_stage_model", "cond_stage_model"]:
            m = getattr(self.ldm, mname, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

    def _encode_condition(self, texts):
        # texts: list[str]
        try:
            return self.ldm.get_learned_conditioning(texts)
        except Exception:
            csm = getattr(self.ldm, "cond_stage_model", None)
            if csm is None:
                raise
            return csm(texts)

    def _compute_loss(self, batch):
        """
        Try a few common training APIs.
        If none are available, compute a standard diffusion noise-prediction loss.
        """
        x = batch["fbank"].to(self.device)   # [B,1,T,M]
        texts = batch["text"]

        # 1) conditioning (text -> CLAP)
        try:
            c = self.ldm.get_learned_conditioning(texts)
        except Exception:
            csm = getattr(self.ldm, "cond_stage_model", None)
            if csm is None:
                raise
            c = csm(texts)
        cond = {"y": c}

        # ---------- (A) try direct forward with return_loss ----------
        for try_kwargs in (
            {"return_loss": True},
            {"return_loss": True, "use_ema": False},
            {},  # maybe it already returns a loss
        ):
            try:
                out = self.ldm(x, cond, **{k:v for k,v in try_kwargs.items() if k in self.ldm.__call__.__code__.co_varnames})
            except Exception:
                out = None

            # tensor -> that's our loss
            if torch.is_tensor(out):
                return out, {}

            # tuple(list) -> pick the first tensor; stash a dict if present
            if isinstance(out, (tuple, list)):
                loss = next((t for t in out if torch.is_tensor(t)), None)
                loss_dict = next((d for d in out if isinstance(d, dict)), {})
                if loss is not None:
                    return loss, loss_dict

            # dict -> check common keys
            if isinstance(out, dict):
                for k in ("loss", "loss_total", "loss_simple", "loss_vlb"):
                    if k in out and torch.is_tensor(out[k]):
                        return out[k], out

        # ---------- (B) try a method named 'loss' ----------
        if hasattr(self.ldm, "loss"):
            try:
                out = self.ldm.loss(x, cond)
                if torch.is_tensor(out):
                    return out, {}
                if isinstance(out, (tuple, list)):
                    loss = next((t for t in out if torch.is_tensor(t)), None)
                    loss_dict = next((d for d in out if isinstance(d, dict)), {})
                    if loss is not None: 
                        return loss, loss_dict
                if isinstance(out, dict):
                    for k in ("loss", "loss_total", "loss_simple", "loss_vlb"):
                        if k in out and torch.is_tensor(out[k]):
                            return out[k], out
            except Exception:
                pass

        # ---------- (C) generic fallback: compute diffusion MSE on noise ----------
        # encode to first-stage latents if helpers exist
        with torch.no_grad():
            if hasattr(self.ldm, "encode_first_stage"):
                # standard LDM helper (returns DiagonalGaussianDistribution)
                try:
                    z = self.ldm.get_first_stage_encoding(self.ldm.encode_first_stage(x))
                except Exception:
                    # some forks return tensor directly
                    enc = self.ldm.encode_first_stage(x)
                    z = enc if torch.is_tensor(enc) else getattr(enc, "sample", lambda: enc)()
            elif hasattr(self.ldm, "first_stage_model") and hasattr(self.ldm.first_stage_model, "encode"):
                enc = self.ldm.first_stage_model.encode(x)
                z = enc if torch.is_tensor(enc) else getattr(enc, "sample", lambda: enc)()
            else:
                # model might already operate in input domain
                z = x

        B = z.shape[0]
        device = z.device
        num_timesteps = int(getattr(self.ldm, "num_timesteps", 1000))
        t = torch.randint(0, num_timesteps, (B,), device=device).long()
        noise = torch.randn_like(z)

        # diffuse x0 -> xt
        if hasattr(self.ldm, "q_sample"):
            zt = self.ldm.q_sample(z_start=z, t=t, noise=noise) \
                 if "z_start" in self.ldm.q_sample.__code__.co_varnames \
                 else self.ldm.q_sample(z, t, noise)
        else:
            # manual q_sample using alphas_cumprod if exposed
            ac = getattr(self.ldm, "alphas_cumprod", None)
            if ac is None:
                raise RuntimeError("Neither ldm.q_sample nor ldm.alphas_cumprod found; cannot form fallback loss.")
            at = ac.to(device)[t].view(B, *([1] * (z.dim()-1)))
            zt = at.sqrt() * z + (1.0 - at).sqrt() * noise

        # UNet predicts noise
        eps_pred = None
        # common helper
        if hasattr(self.ldm, "apply_model"):
            try:
                # some forks accept cond dict, others take cond["y"]
                try:
                    eps_pred = self.ldm.apply_model(zt, t, cond)
                except Exception:
                    eps_pred = self.ldm.apply_model(zt, t, cond["y"])
            except Exception:
                eps_pred = None
        # direct UNet
        if eps_pred is None and hasattr(self.ldm, "model") and hasattr(self.ldm.model, "diffusion_model"):
            try:
                eps_pred = self.ldm.model.diffusion_model(zt, t, cond["y"])
            except Exception:
                pass

        if eps_pred is None:
            raise RuntimeError("Could not call UNet: neither ldm.apply_model nor model.diffusion_model worked.")

        loss = F.mse_loss(eps_pred, noise)
        return loss, {"loss_simple": loss}


    def training_step(self, batch, batch_idx):
        loss, loss_dict = self._compute_loss(batch)
        # Log primary loss
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        # Forward useful keys if they exist (e.g., loss_simple_ema)
        for k in ["loss_simple", "loss_simple_ema", "loss_vlb"]:
            if k in loss_dict and torch.is_tensor(loss_dict[k]):
                self.log(f"train/{k}", loss_dict[k], on_step=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self._compute_loss(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if "loss_simple_ema" in loss_dict and torch.is_tensor(loss_dict["loss_simple_ema"]):
            self.log("val/loss_simple_ema", loss_dict["loss_simple_ema"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = getattr(self.ldm, "learning_rate", 1e-5)
        params = [p for p in self.ldm.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
        # optional cosine schedule could be added here
        return opt

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--precision", default="16")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    log_dir = cfg.get("log_directory", os.path.join(REPO_DIR, "model_logs"))
    run_name = cfg.get("id", {}).get("name", "finetune_ldm")
    out_dir = os.path.join(log_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    train_loader, val_loader = make_loaders(cfg)
    import math
    print("[debug] train_json:", cfg["preprocess_config"]["path"]["train_data"])
    print("[debug] #items:", len(train_loader.dataset), 
        " batch_size:", train_loader.batch_size, 
        " len(train_loader):", len(train_loader),
        " steps/epoch (ceil):", math.ceil(len(train_loader.dataset)/train_loader.batch_size))

    ldm = build_model(cfg)

    # Wrap with Lightning module
    module = LDMFinetuneModule(ldm, cfg)

    # Checkpoint & LR monitor
    ckpt = ModelCheckpoint(
        dirpath=os.path.join(out_dir, "checkpoints"),
        filename="{step:08d}",
        monitor="val/loss",    # generic (we also log val/loss_simple_ema if available)
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=args.val_every
    )
    lrm = LearningRateMonitor(logging_interval="step")

    # Devices & Strategy (PL 1.8.x safe) 
    from pytorch_lightning.strategies import DDPStrategy

    if torch.cuda.is_available():
        ndev = torch.cuda.device_count()
        accelerator = "gpu"
        if ndev > 1:
            devices = list(range(ndev))                 # e.g., [0,1,2]
            strategy = DDPStrategy(find_unused_parameters=False)
        else:
            devices = 1                                 # IMPORTANT: int, not [0]
            strategy = None                             # let PL choose SingleDeviceStrategy
    else:
        accelerator = "cpu"
        devices = 1
        strategy = None
        
    precision_arg = normalize_precision_arg(args.precision)

    trainer_kwargs = dict(
        default_root_dir=out_dir,
        max_steps=args.max_steps,
        val_check_interval=args.val_every,
        log_every_n_steps=50,
        callbacks=[ckpt, lrm],
        accelerator=accelerator,
        devices=devices,              # 1 or [0,1,2]
        precision=precision_arg,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
    )
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy

    trainer = pl.Trainer(**trainer_kwargs)
    print(f"[trainer] pl={pl.__version__} strategy={strategy} devices={devices} precision={precision_arg}")
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"[done] checkpoints in: {os.path.join(out_dir, 'checkpoints')}")

if __name__ == "__main__":
    main()