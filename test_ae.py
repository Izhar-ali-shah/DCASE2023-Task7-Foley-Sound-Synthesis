#!/usr/bin/env python3
import os, sys, json, re, importlib, argparse
from typing import Dict, Any
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio, matplotlib.pyplot as plt

# --- repo paths ---
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(REPO_DIR, "src")
for p in (SRC_DIR, REPO_DIR):
    if p not in sys.path: sys.path.insert(0, p)

import yaml

def instantiate_from_config(cfg: Dict[str, Any]):
    module_path, cls_name = cfg["target"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)(**(cfg.get("params", {}) or {}))

# ---- dataset (same as training) ----
class DCASEMelDataset(Dataset):
    def __init__(self, json_path: str, cfg: Dict[str, Any]):
        with open(json_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        pp    = cfg["preprocess_config"]["preprocessing"]
        audio = pp["audio"]; stft = pp["stft"]; mel = pp["mel"]

        self.sr   = int(audio.get("sampling_rate", 22050))
        self.seg  = int(audio.get("segment_size", self.sr * 4))
        self.nfft = int(stft.get("filter_length", 1024))
        self.hop  = int(stft.get("hop_length", 256))
        self.win  = int(stft.get("win_length", self.nfft))
        self.nmel = int(mel.get("n_mel_channels", 80))
        self.fmin = float(mel.get("mel_fmin", 0.0))
        self.fmax = float(mel.get("mel_fmax", 8000.0))
        self.Ttar = int(mel.get("target_length", 344))

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=self.nfft, hop_length=self.hop, win_length=self.win,
            f_min=self.fmin, f_max=self.fmax, n_mels=self.nmel, power=1.0, center=True, pad_mode="reflect",
        )

    def __len__(self): return len(self.items)

    def _load_wav(self, path):
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1: wav = wav.mean(0, keepdim=True)
        if sr != self.sr:   wav = torchaudio.functional.resample(wav, sr, self.sr)
        T = wav.shape[-1]
        if T < self.seg: wav = F.pad(wav, (0, self.seg - T))
        else:            wav = wav[..., :self.seg]
        return wav

    def __getitem__(self, i):
        rec  = self.items[i]
        wav  = self._load_wav(rec["wav"])
        mel  = self.melspec(wav).clamp(min=1e-5).log()   # [1,n_mel,Tm]
        mel  = mel.squeeze(0)                            # [n_mel,Tm]
        if mel.size(1) < self.Ttar: mel = F.pad(mel, (0,0,0,self.Ttar - mel.size(1)))
        else:                       mel = mel[:, :self.Ttar]
        mel  = mel.transpose(0,1).unsqueeze(0)           # [1,T,n_mel]
        return {"fbank": mel, "wav_path": rec["wav"]}

def load_yaml(p): 
    with open(p, "r") as f: 
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt",   required=True, help="Path to AE checkpoint (from train_ae.py)")
    ap.add_argument("--split",  choices=["train","test"], default="test")
    ap.add_argument("--batch",  type=int, default=8)
    ap.add_argument("--num_examples", type=int, default=8)
    ap.add_argument("--outdir", default="ae_eval")
    ap.add_argument("--save_wav", type=int, default=0, help="Save N wavs via Griffin-Lim (rough)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_yaml(args.config)
    pth = cfg["preprocess_config"]["path"]
    json_path = pth["train_data"] if args.split=="train" else pth["test_data"]

    ds = DCASEMelDataset(json_path, cfg)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2,
                    collate_fn=lambda b: torch.stack([x["fbank"] for x in b], 0))

    # build AE & load weights from Lightning ckpt
    ae_conf = cfg["first_stage_train_config"]
    ae = instantiate_from_config(ae_conf).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # keep only keys that belong to the AE inside LightningModule (prefix "ae.")
    ae_state = {k.replace("ae.", "", 1): v for k, v in state.items() if k.startswith("ae.")}
    missing, unexpected = ae.load_state_dict(ae_state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    # metrics
    l1_list, mse_list, snr_list = [], [], []

    @torch.no_grad()
    def ae_recon(x):
        if hasattr(ae, "encode") and hasattr(ae, "decode"):
            post = ae.encode(x)
            z = post.sample() if hasattr(post, "sample") else post
            xhat = ae.decode(z)
        else:
            xhat = ae(x)
        return xhat

    # run one pass
    saved = 0
    for i, x in enumerate(dl):
        x = x.to(device)                   # [B,1,T,M]
        with torch.no_grad():
            xhat = ae_recon(x)

        # metrics
        l1  = F.l1_loss(xhat, x, reduction="none").mean(dim=(1,2,3))
        mse = F.mse_loss(xhat, x, reduction="none").mean(dim=(1,2,3))
        snr = 10 * torch.log10((x.pow(2).mean(dim=(1,2,3)) + 1e-12) / (mse + 1e-12))

        l1_list.append(l1.cpu()); mse_list.append(mse.cpu()); snr_list.append(snr.cpu())

        # save a few viz
        if saved < args.num_examples:
            B = x.size(0)
            take = min(B, args.num_examples - saved)
            x_np    = x[:take].cpu().numpy()      # [b,1,T,M]
            xhat_np = xhat[:take].cpu().numpy()
            for k in range(take):
                fig, axs = plt.subplots(1, 2, figsize=(8,3), dpi=120)
                axs[0].imshow(x_np[k,0].T, aspect="auto", origin="lower"); axs[0].set_title("orig mel")
                axs[1].imshow(xhat_np[k,0].T, aspect="auto", origin="lower"); axs[1].set_title("recon mel")
                for ax in axs: ax.axis("off")
                fig.tight_layout()
                plt.savefig(os.path.join(args.outdir, f"mel_recon_{saved:03d}.png"))
                plt.close(fig)
                saved += 1
                if saved >= args.num_examples: break

        # optional Griffin-Lim audio (rough!)
        if args.save_wav > 0:
            pp = cfg["preprocess_config"]["preprocessing"]
            sr   = int(pp["audio"]["sampling_rate"])
            nfft = int(pp["stft"]["filter_length"])
            hop  = int(pp["stft"]["hop_length"])
            win  = int(pp["stft"]["win_length"])
            nmel = int(pp["mel"]["n_mel_channels"])
            fmin = float(pp["mel"]["mel_fmin"]); fmax = float(pp["mel"]["mel_fmax"])
            n_stft = nfft // 2 + 1

            invmel = torchaudio.transforms.InverseMelScale(n_stft=n_stft, n_mels=nmel, sample_rate=sr, f_min=fmin, f_max=fmax).to(device)
            griff  = torchaudio.transforms.GriffinLim(n_fft=nfft, hop_length=hop, win_length=win, power=1.0).to(device)

            # our mel was log(power^1.0); invert that
            def mel_to_wav(mel_log):  # accepts [B,1,T,M] or [B,T,M]
                # Drop the singleton channel dim if present
                if mel_log.dim() == 4:          # [B,1,T,M]
                    mel_log = mel_log.squeeze(1) # -> [B,T,M]
                # We logged magnitude mels during preprocessing (power=1.0), so invert with exp()
                mel = mel_log.exp().permute(0, 2, 1)   # [B,T,M] -> [B,M,T]
                spec = invmel(mel)                     # [B, n_stft, T]
                wav  = griff(spec)                     # [B, T']
                return wav

            bsave = min(x.size(0), args.save_wav)
            for k in range(bsave):
                w0 = mel_to_wav(x[k:k+1])
                w1 = mel_to_wav(xhat[k:k+1])
                torchaudio.save(os.path.join(args.outdir, f"orig_{i:04d}_{k}.wav"), w0.cpu(), sample_rate=sr)
                torchaudio.save(os.path.join(args.outdir, f"recon_{i:04d}_{k}.wav"), w1.cpu(), sample_rate=sr)
            args.save_wav = max(0, args.save_wav - bsave)  # only first batches

    import numpy as np
    L1  = torch.cat(l1_list).mean().item() if l1_list else float("nan")
    MSE = torch.cat(mse_list).mean().item() if mse_list else float("nan")
    SNR = torch.cat(snr_list).mean().item() if snr_list else float("nan")
    print(f"[AE EVAL] split={args.split}  L1={L1:.4f}  MSE={MSE:.6f}  SNR={SNR:.2f} dB")
    with open(os.path.join(args.outdir, "metrics.txt"), "w") as f:
        f.write(f"L1={L1:.6f}\nMSE={MSE:.8f}\nSNR_dB={SNR:.3f}\n")

if __name__ == "__main__":
    main()
