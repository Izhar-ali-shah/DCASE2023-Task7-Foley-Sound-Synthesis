import os, sys, json, importlib, argparse, re
from typing import Dict, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# utils
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(REPO_DIR, "src")
for p in (SRC_DIR, REPO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import yaml
except Exception:
    yaml = None

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

def load_yaml(path):
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config files.")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def instantiate_from_config(config: Dict[str, Any]):
    if "target" not in config:
        raise KeyError("Expected key 'target' in config to instantiate module.")
    module_path, cls_name = config["target"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    params = config.get("params", {}) or {}
    return cls(**params)

# dataset 
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
            f_min=self.fmin, f_max=self.fmax, n_mels=self.nmel, power=1.0,
            center=True, pad_mode="reflect",
        )

    def __len__(self): return len(self.items)

    def _load_wav_1T(self, path: str):
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
        wav  = self._load_wav_1T(rec["wav"])
        mel  = self.melspec(wav).clamp(min=1e-5).log()  # [1,n_mel,Tm]
        mel  = mel.squeeze(0)                           # [n_mel,Tm]
        if mel.size(1) < self.Ttar:
            mel = F.pad(mel, (0, 0, 0, self.Ttar - mel.size(1)))
        else:
            mel = mel[:, :self.Ttar]
        mel = mel.transpose(0, 1).unsqueeze(0)          # [1,T,n_mel]
        return {"fbank": mel}

def make_loaders(cfg: Dict[str, Any], batchsize: int):
    pth = cfg["preprocess_config"]["path"]
    nw  = int(os.environ.get("NUM_WORKERS", "4"))
    dtr = DCASEMelDataset(pth["train_data"], cfg)
    dvl = DCASEMelDataset(pth["test_data"],  cfg)

    def collate(b):
        return torch.stack([x["fbank"] for x in b], 0)  # [B,1,T,M]

    tr = DataLoader(dtr, batch_size=batchsize, shuffle=True,  num_workers=nw, pin_memory=True, collate_fn=collate)
    vl = DataLoader(dvl, batch_size=batchsize, shuffle=False, num_workers=nw, pin_memory=True, collate_fn=collate)
    return tr, vl

# Lightning wrapper
class AETrainer(pl.LightningModule):
    def __init__(self, ae, lr=1e-4, lpips_weight=0.0, kl_weight=1e-6):
        super().__init__()
        self.ae = ae
        self.lr = lr
        self.lpips_weight = lpips_weight
        self.kl_weight    = kl_weight

        # Optional LPIPS (off by default for mels)
        self.lpips = None
        if lpips_weight > 0:
            from taming.modules.losses.lpips import LPIPS
            self.lpips = LPIPS().eval()
            for p in self.lpips.parameters():
                p.requires_grad = False

    def forward(self, x):
        # Try common VAE interface
        if hasattr(self.ae, "encode") and hasattr(self.ae, "decode"):
            posterior = self.ae.encode(x)
            z = posterior.sample() if hasattr(posterior, "sample") else posterior
            x_hat = self.ae.decode(z)
            kl = torch.zeros((), device=x.device)
            if hasattr(posterior, "kl"):
                kl = posterior.kl().mean()
            return x_hat, kl
        # Fallback: plain AE
        return self.ae(x), torch.zeros((), device=x.device)

    def _loss(self, x, x_hat, kl):
        l1 = F.l1_loss(x_hat, x)
        loss = l1 + self.kl_weight * kl
        if self.lpips is not None:
            # LPIPS expects 3ch [-1,1]; approximate from mel
            a = x.permute(0,1,3,2).repeat(1,3,1,1)       # [B,3,T,M]
            b = x_hat.permute(0,1,3,2).repeat(1,3,1,1)
            def norm(u):
                umin = u.amin(dim=(2,3), keepdim=True)
                umax = u.amax(dim=(2,3), keepdim=True)
                u = (u - umin) / (umax - umin + 1e-6)
                return u * 2 - 1
            p = self.lpips(norm(a), norm(b)).mean()
            loss = loss + self.lpips_weight * p
        return loss, {"l1": l1, "kl": kl}

    def training_step(self, batch, _):
        x = batch.to(self.device)  # batch is [B,1,T,M]
        x_hat, kl = self(x)
        loss, d = self._loss(x, x_hat, kl)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/l1",   d["l1"], on_step=True)
        self.log("train/kl",   d["kl"], on_step=True)
        return loss

    def validation_step(self, batch, _):
        x = batch.to(self.device)
        x_hat, kl = self(x)
        loss, d = self._loss(x, x_hat, kl)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/l1",   d["l1"], on_epoch=True)
        self.log("val/kl",   d["kl"], on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2)

#  main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--precision", default="32")
    args = ap.parse_args()

    precision_arg = normalize_precision_arg(args.precision)
    cfg = load_yaml(args.config)

    # Instantiate AE from YAML (expects a block like 'first_stage_train_config')
    ae_conf = cfg["first_stage_train_config"]
    ae = instantiate_from_config(ae_conf)

    # Hyperparams (optionally in YAML under 'ae')
    lr      = float(cfg.get("ae", {}).get("lr", 1e-4))
    bs      = int(cfg.get("ae", {}).get("batchsize", 8))
    lpips_w = float(cfg.get("ae", {}).get("lpips_weight", 0.0))
    kl_w    = float(cfg.get("ae", {}).get("kl_weight",   1e-6))

    module = AETrainer(ae, lr=lr, lpips_weight=lpips_w, kl_weight=kl_w)

    # Data
    tr, vl = make_loaders(cfg, batchsize=bs)

    # Keep val_every <= #train batches to avoid Lightning error
    num_train_batches = len(tr)
    val_every_safe = min(args.val_every, num_train_batches) if num_train_batches > 0 else args.val_every
    if val_every_safe != args.val_every:
        print(f"[train_ae] lowering val_every from {args.val_every} to {val_every_safe} (num_train_batches={num_train_batches})")

    # Logging dirs
    log_dir = cfg.get("log_directory", os.path.join(REPO_DIR, "model_logs"))
    run_name = cfg.get("ae", {}).get("name", "ae_dcase7")
    out_dir = os.path.join(log_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Callbacks
    ckpt = ModelCheckpoint(
        dirpath=os.path.join(out_dir, "checkpoints"),
        filename="{step:08d}",
        monitor="val/loss",          # we log 'val/loss' above
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=val_every_safe
    )
    lrm = LearningRateMonitor(logging_interval="step")

    # Accelerator / devices / strategy
    if torch.cuda.is_available():
        ndev = torch.cuda.device_count()
        accelerator = "gpu"
        if ndev > 1:
            devices = list(range(ndev))                 # [0,1,2,...]
            strategy = DDPStrategy(find_unused_parameters=False)
        else:
            devices = 1                                 # int, not [0]
            strategy = None
    else:
        accelerator = "cpu"
        devices = 1
        strategy = None

    # Trainer
    trainer_kwargs = dict(
        default_root_dir=out_dir,
        max_steps=args.max_steps,
        val_check_interval=val_every_safe,
        log_every_n_steps=50,
        callbacks=[ckpt, lrm],
        accelerator=accelerator,
        devices=devices,
        precision=precision_arg,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
    )
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy

    trainer = pl.Trainer(**trainer_kwargs)
    print(f"[trainer] pl={pl.__version__} strategy={strategy} devices={devices} precision={precision_arg}")
    trainer.fit(module, train_dataloaders=tr, val_dataloaders=vl)
    print(f"[done] checkpoints in: {os.path.join(out_dir, 'checkpoints')}")

if __name__ == "__main__":
    main()