#!/usr/bin/env python3
import os, sys, re, json, argparse, importlib
from typing import Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio

# ---------------- utils ----------------
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(REPO_DIR, "src")
for p in (SRC_DIR, REPO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import yaml
except Exception:
    yaml = None

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def instantiate_from_config(conf: Dict[str, Any]):
    module_path, cls_name = conf["target"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls(**(conf.get("params", {}) or {}))

# ---------------- dataset ----------------
class DCASEMelDataset(Dataset):
    def __init__(self, json_path: str, cfg: Dict[str, Any]):
        with open(json_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

        pp   = cfg.get("preprocess_config", {}).get("preprocessing", {})
        audio= pp.get("audio", {})
        stft = pp.get("stft", {})
        mel  = pp.get("mel", {})

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
            f_min=self.fmin, f_max=self.fmax, n_mels=self.nmel, power=1.0, center=True, pad_mode="reflect"
        )

    def __len__(self): return len(self.items)

    def _load_wav_1T(self, path: str):
        wav, sr = torchaudio.load(path)  # [C,T]
        if wav.size(0) > 1: wav = wav.mean(0, keepdim=True)
        if sr != self.sr:   wav = torchaudio.functional.resample(wav, sr, self.sr)
        T = wav.shape[-1]
        if T < self.seg: wav = F.pad(wav, (0, self.seg - T))
        else:            wav = wav[..., :self.seg]
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
        mel = mel.transpose(0,1).unsqueeze(0)           # [1,T,M]
        return {"fbank": mel}

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML used for AE/LDM")
    ap.add_argument("--ckpt",   required=True, help="AE Lightning checkpoint (last.ckpt/best.ckpt)")
    ap.add_argument("--split",  default="train", choices=["train","test"])
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--max_batches", type=int, default=300, help="how many batches to average over")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["preprocess_config"]["path"]
    json_path = paths["train_data"] if args.split=="train" else paths["test_data"]

    # dataset/loader
    ds = DCASEMelDataset(json_path, cfg)
    def collate(b):
        return torch.stack([x["fbank"] for x in b], 0) # [B,1,T,M]
    nw = int(os.environ.get("NUM_WORKERS","4"))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=nw, pin_memory=True, collate_fn=collate)

    # build AE as in training (first_stage_train_config)
    ae_conf = cfg["first_stage_train_config"]
    ae = instantiate_from_config(ae_conf)
    ae.eval()

    # load weights from Lightning ckpt (strip 'ae.' prefix if present)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["state_dict"]
    if any(k.startswith("ae.") for k in sd.keys()):
        sd = {k.replace("ae.","",1): v for k,v in sd.items() if k.startswith("ae.")}
    missing, unexpected = ae.load_state_dict(sd, strict=False)
    print(f"[load AE] missing={len(missing)} unexpected={len(unexpected)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae.to(device)

    # iterate and accumulate latent std
    n_seen, std_sum = 0, 0.0
    with torch.no_grad():
        for bidx, x in enumerate(loader):
            x = x.to(device)                       # [B,1,T,M]
            posterior = ae.encode(x)               # DiagonalGaussianDistribution or Tensor
            if hasattr(posterior, "sample"):
                z = posterior.sample()             # [B,C,t,f]
            else:
                z = posterior
            std_sum += z.std().item()
            n_seen  += 1
            if n_seen >= args.max_batches:
                break

    mean_std = std_sum / max(n_seen,1)
    scale_factor = 1.0 / mean_std if mean_std > 0 else 1.0
    print(f"\nMean latent std over {n_seen} batches: {mean_std:.6f}")
    print(f"Recommended scale_factor = 1 / mean_std = {scale_factor:.6f}\n")

    # handy YAML snippet to paste
    print("Put this in your YAML:\n"
          "model:\n"
          "  params:\n"
          "    scale_by_std: false\n"
          f"    scale_factor: {scale_factor:.6f}\n")

if __name__ == "__main__":
    main()
