import os
import sys
import csv
import time
import argparse
import importlib
import traceback
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F
import torchaudio

try:
    import yaml
except Exception:
    yaml = None

# Import paths

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
# Point this to your repo's src folder (absolute path is fine)
SRC_DIR = "/home/mukh1485/media_project_update2/src"
for p in (SRC_DIR, REPO_DIR):
    if p and p not in sys.path:
        sys.path.insert(0, p)

# ------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------
def log(msg: str, device_tag: str = None):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {f'[{device_tag}] ' if device_tag else ''}{msg}", flush=True)

def resolve_path(path: str, candidates: List[str] = None) -> str:
    if path is None:
        return None
    if os.path.isfile(path):
        return path
    if not os.path.isabs(path):
        tests = [os.path.join(REPO_DIR, path)]
        for sub in ("config", "configs"):
            tests.append(os.path.join(REPO_DIR, sub, path))
        if candidates:
            tests.extend(candidates)
        for t in tests:
            if os.path.isfile(t):
                return t
    return path

def instantiate_from_config(config: Dict[str, Any]):
    if "target" not in config:
        raise KeyError("Expected key 'target' in config.")
    module_path, cls_name = config["target"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    params = config.get("params", {}) or {}
    return cls(**params)

def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config files.")
    p = resolve_path(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p, "r") as f:
        return yaml.safe_load(f)

def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def device_list_from_env() -> List[str]:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if vis is not None and vis.strip():
        ids = [s.strip() for s in vis.split(",") if s.strip()]
        return [f"cuda:{i}" for i in range(len(ids))]
    if not torch.cuda.is_available():
        return ["cpu"]
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def read_class_index_csv(csv_path: str) -> List[Dict[str, str]]:
    """
    Expected columns: index, mid, display_name (AudioSet-style).
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = {h.lower(): h for h in (reader.fieldnames or [])}
        idx_key  = headers.get("index")
        mid_key  = headers.get("mid") or headers.get("label") or headers.get("class_id")
        name_key = headers.get("display_name") or headers.get("name") or headers.get("class_name")
        for row in reader:
            rows.append({
                "index": row.get(idx_key, "") if idx_key else "",
                "mid": row.get(mid_key, ""),
                "display_name": row.get(name_key, row.get(mid_key, "")),
            })
    return rows

# Mel -> waveform decoding

@torch.no_grad()
def decode_mel_to_wav(model: nn.Module, mel: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    # Normalize to [B, n_mels, T]
    if mel.dim() == 4 and mel.size(1) == 1:
        mel_ = mel.squeeze(1)
    elif mel.dim() == 3:
        mel_ = mel
    else:
        mel_ = mel.view(mel.size(0), -1, mel.size(-1))

    if mel_.shape[1] > mel_.shape[2]:
        mel_ = mel_.transpose(1, 2)

    # Try model/first_stage direct decoder hooks
    for obj in (model, getattr(model, "first_stage_model", None)):
        if obj is None: continue
        fn = getattr(obj, "decode_to_waveform", None)
        if callable(fn):
            out = fn(mel_)
            if isinstance(out, torch.Tensor):
                return out

    def _as_b80t(x: torch.Tensor, n_mels: int = 80) -> torch.Tensor:
        if x.dim() == 4 and x.size(1) == 1: x = x.squeeze(1)
        if x.dim() == 2: x = x.unsqueeze(0)
        if x.shape[-2] != n_mels and x.shape[-1] == n_mels: x = x.transpose(-2, -1)
        if x.shape[-2] != n_mels and n_mels in x.shape:
            # move 'n_mels' axis to channel axis
            idx = [i for i, s in enumerate(x.shape) if s == n_mels][-1]
            perm = list(range(x.dim()))
            perm.pop(idx); perm.insert(-2, idx)
            x = x.permute(*perm).contiguous()
        assert x.shape[-2] == n_mels, f"mel must be [B,{n_mels},T], got {tuple(x.shape)}"
        return x

    # Try common vocoder attributes/methods
    for holder in (model, getattr(model, "first_stage_model", None)):
        if holder is None: continue
        for name in ("generator", "hifigan", "vocoder", "mel_to_wav"):
            g = getattr(holder, name, None)
            if g is None: continue
            for meth in (None, "infer", "inference", "generate", "synthesis", "mel2wav", "mel_to_wav"):
                try:
                    m = _as_b80t(mel_)
                    if meth is None and callable(g):
                        out = g(m)
                    else:
                        fn = getattr(g, meth, None)
                        if fn is None: continue
                        out = fn(m)
                    if isinstance(out, dict):
                        for k in ("audio", "wav", "waveform"):
                            if k in out and isinstance(out[k], torch.Tensor):
                                return out[k]
                    if isinstance(out, torch.Tensor):
                        return out
                except Exception:
                    pass

    # Griffin–Lim fallback
    from torchaudio import functional as AF
    pp   = cfg.get("preprocess_config", {}).get("preprocessing", {})
    aud  = pp.get("audio", {})
    stft = pp.get("stft", {})
    melc = pp.get("mel", {})

    sr   = int(aud.get("sampling_rate", 22050))
    nfft = int(stft.get("filter_length", 1024))
    hop  = int(stft.get("hop_length", 256))
    win  = int(stft.get("win_length", nfft))
    nmel = int(melc.get("n_mel_channels", 80))
    fmin = float(melc.get("mel_fmin", 0.0))
    fmxv = melc.get("mel_fmax", None)
    fmax = float(fmxv) if fmxv is not None else None

    mel_amp = mel_.detach()
    if torch.median(mel_amp).item() < 0.0:
        mel_amp = torch.exp(mel_amp)
    n_stft = nfft // 2 + 1

    mag = None
    inv_f = getattr(AF, "inverse_mel_scale", None)
    if inv_f is not None:
        try:
            mag = inv_f(mel_amp, n_stft=n_stft, sample_rate=sr, f_min=fmin, f_max=fmax)
        except Exception:
            mag = None
    if mag is None:
        InvMel = getattr(torchaudio.transforms, "InverseMelScale", None)
        if InvMel is not None:
            try:
                inv = InvMel(n_stft=n_stft, n_mels=nmel, sample_rate=sr, f_min=fmin, f_max=fmax, max_iter=128).to(mel_amp.device)
                mag = inv(mel_amp)
            except Exception:
                mag = None
    if mag is None:
        fb = AF.melscale_fbanks(n_stft, f_min=fmin, f_max=fmax if fmax is not None else sr/2, n_mels=nmel, sample_rate=sr).to(mel_amp.device)
        M = fb if fb.shape == (nmel, n_stft) else fb.transpose(0, 1)
        M_pinv = torch.linalg.pinv(M)
        mag = torch.einsum("fm,bmt->bft", M_pinv, mel_amp)

    GL = getattr(AF, "griffinlim", None)
    if GL is not None:
        return GL(mag, n_fft=nfft, win_length=win, hop_length=hop, power=1.0, n_iter=60)
    GLT = getattr(torchaudio.transforms, "GriffinLim", None)
    if GLT is None:
        raise RuntimeError("No waveform decoder and Griffin–Lim unavailable.")
    gl = GLT(n_fft=nfft, win_length=win, hop_length=hop, power=1.0, n_iter=60).to(mel_amp.device)
    return gl(mag)

# Model & sampling helpers
import inspect

def build_model(cfg: Dict[str, Any], device: str, device_tag: str):
    model_cfg = cfg["model"]
    mparams   = model_cfg.get("params", {}) or {}
    first_stage = mparams.get("first_stage_config", model_cfg.get("first_stage_config"))
    cond_stage  = mparams.get("cond_stage_config",  model_cfg.get("cond_stage_config"))
    inst_conf = {"target": model_cfg["target"], "params": {**mparams, "first_stage_config": first_stage, "cond_stage_config": cond_stage}}
    log("building model...", device_tag)
    model = instantiate_from_config(inst_conf)

    ckpt_path = model_cfg.get("reload_from_ckpt") or model_cfg.get("ckpt_path") or model_cfg.get("resume")
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd   = ckpt.get("state_dict_ema") or ckpt.get("state_dict") or ckpt
        if not isinstance(sd, dict):
            raise RuntimeError(f"Unexpected checkpoint format at {ckpt_path}")
        # strip 'ldm.' prefix from LightningModule wrapper
        new_sd = { (k[4:] if k.startswith("ldm.") else k): v for k, v in sd.items() }
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        print(f"[{device_tag}] loaded diffusion ckpt: {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:10]:    print(f"[{device_tag}] missing keys (first 10): {missing[:10]}")
        if unexpected[:10]: print(f"[{device_tag}] unexpected keys (first 10): {unexpected[:10]}")
    else:
        log("WARNING: no diffusion checkpoint loaded — using random weights!", device_tag)

    model.eval().to(device)
    log(f"Model params: {count_params(model):.2f} M", device_tag)
    return model

def call_sample_log_compat(model, cond_tensor, uncond_tensor, ddim_steps, cfg_scale, eta, batch_size):
    fn = getattr(model, "sample_log", None)
    if fn is None:
        raise AttributeError("model has no sample_log(...)")
    params = set(inspect.signature(fn).parameters.keys())

    if {"cond", "batch_size", "ddim", "ddim_steps"} <= params:
        return fn(cond=cond_tensor, batch_size=batch_size, ddim=True, ddim_steps=ddim_steps,
                  unconditional_guidance_scale=cfg_scale, unconditional_conditioning=uncond_tensor, eta=eta)

    if {"S", "conditioning"} <= params:
        cond_dict   = {"y": cond_tensor}
        uncond_dict = {"y": uncond_tensor}
        kwargs = dict(S=ddim_steps, conditioning=cond_dict, unconditional_conditioning=uncond_dict,
                      unconditional_guidance_scale=cfg_scale, eta=eta)
        if "batch_size" in params:
            kwargs["batch_size"] = batch_size
        return fn(**kwargs)

    raise TypeError(f"Unsupported sample_log signature: {params}")

@torch.no_grad()
def sample_and_decode(model, cond_y, uncond_y, cfg_scale, ddim_steps, eta, cfg):
    bs = cond_y.shape[0] if isinstance(cond_y, torch.Tensor) else 1
    out = call_sample_log_compat(model, cond_y, uncond_y, ddim_steps, cfg_scale, eta, bs)
    samples = out[0] if isinstance(out, (tuple, list)) else out
    mel = model.decode_first_stage(samples)
    wav = decode_mel_to_wav(model, mel, cfg)
    return wav, mel

def save_waveform(wav: torch.Tensor, sr: int, path: str):
    safe_mkdir(os.path.dirname(path))
    x = wav.detach().cpu().float()
    if x.dim() == 3 and x.size(1) == 1: x = x.squeeze(1)
    if x.dim() == 1: x = x.unsqueeze(0)
    elif x.dim() > 2: x = x.view(x.size(0), -1)
    peak = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    x = x / peak
    try:
        import soundfile as sf
        sf.write(path, x.t().numpy(), sr)
    except Exception:
        torchaudio.save(path, x, sample_rate=sr)

def _extract_tensor(x):
    if isinstance(x, torch.Tensor): return x
    if isinstance(x, dict):
        if "y" in x and isinstance(x["y"], torch.Tensor): return x["y"]
        for v in x.values():
            t = _extract_tensor(v)
            if isinstance(t, torch.Tensor): return t
    if isinstance(x, (list, tuple)):
        for v in x:
            t = _extract_tensor(v)
            if isinstance(t, torch.Tensor): return t
    raise TypeError(f"Could not extract a Tensor from type {type(x)}")

@torch.no_grad()
def make_film_condition_text(model, prompt: str, device: str, cfg_scale: float):
    cond   = model.get_learned_conditioning([prompt])
    uncond = model.get_learned_conditioning([""])
    cond   = _extract_tensor(cond).to(device)
    uncond = _extract_tensor(uncond).to(device)
    return cond, uncond, float(cfg_scale)

def _cfg_preproc(cfg):
    pp   = cfg.get("preprocess_config", {}).get("preprocessing", {})
    aud  = pp.get("audio", {})
    stft = pp.get("stft", {})
    melc = pp.get("mel", {})
    sr   = int(aud.get("sampling_rate", 22050))
    seg  = int(aud.get("segment_size", sr * 4))
    nfft = int(stft.get("filter_length", 1024))
    hop  = int(stft.get("hop_length", 256))
    win  = int(stft.get("win_length", nfft))
    nmel = int(melc.get("n_mel_channels", 80))
    fmin = float(melc.get("mel_fmin", 0.0))
    fmxv = melc.get("mel_fmax", None)
    fmax = float(fmxv) if fmxv is not None else None
    return sr, seg, nfft, hop, win, nmel, fmin, fmax

def _to_bmt(mel_tensor, n_mels=80):
    x = mel_tensor
    if x.dim() == 4 and x.size(1) == 1: x = x.squeeze(1)
    if x.dim() == 2: x = x.unsqueeze(0)
    if x.shape[-2] != n_mels and x.shape[-1] == n_mels: x = x.transpose(-2, -1)
    elif x.shape[1] != n_mels and x.shape[2] == n_mels: x = x.transpose(1, 2)
    if x.shape[1] != n_mels and n_mels in x.shape:
        idx = [i for i, s in enumerate(x.shape) if s == n_mels][-1]
        perm = list(range(x.dim()))
        perm.pop(idx); perm.insert(1, idx)
        x = x.permute(*perm).contiguous()
    assert x.shape[1] == n_mels, f"mel must be [B,{n_mels},T], got {tuple(x.shape)}"
    return x

def save_mel_spectrogram_image(mel, cfg, out_png_path):
    sr, seg, nfft, hop, win, nmel, fmin, fmax = _cfg_preproc(cfg)
    m = _to_bmt(mel, n_mels=nmel)[0].detach().cpu().numpy()
    T = m.shape[1]; dur = T * hop / sr
    plt.figure(figsize=(8, 3))
    plt.imshow(m, origin="lower", aspect="auto")
    plt.colorbar(label="log-mel")
    plt.title(f"Mel spectrogram  n_mels={nmel}, T={T} (~{dur:.2f}s)")
    plt.xlabel("Frames"); plt.ylabel("Mel bins")
    plt.tight_layout(); plt.savefig(out_png_path, dpi=150); plt.close()

def save_stft_spectrogram_image(wav, cfg, out_png_path):
    sr, seg, nfft, hop, win, nmel, fmin, fmax = _cfg_preproc(cfg)
    x = wav.detach().cpu().float()
    if x.dim() == 3 and x.size(1) == 1: x = x.squeeze(1)
    if x.dim() == 2: x = x[0]
    spec = torch.stft(x, n_fft=nfft, hop_length=hop, win_length=win, return_complex=True).abs()
    S = torch.log1p(spec).numpy()
    T = S.shape[1]; dur = T * hop / sr
    plt.figure(figsize=(8, 3))
    plt.imshow(S, origin="lower", aspect="auto")
    plt.colorbar(label="log|STFT|")
    plt.title(f"STFT magnitude  F={S.shape[0]}, T={T} (~{dur:.2f}s)")
    plt.xlabel("Frames"); plt.ylabel("Freq bins")
    plt.tight_layout(); plt.savefig(out_png_path, dpi=150); plt.close()

# Logging helpers
def shard_jobs(rows: List[Dict[str, str]], n_shards: int) -> List[List[Tuple[int, Dict[str, str]]]]:
    jobs = [(i, r) for i, r in enumerate(rows)]
    shards = [[] for _ in range(n_shards)]
    for i, job in enumerate(jobs):
        shards[i % n_shards].append(job)
    return shards

def init_run_log(out_dir: str, cfg: Dict[str, Any], config_path: str, devices: List[str]):
    path = os.path.join(out_dir, "generation_log.txt")
    if not os.path.exists(path):
        sr, seg, nfft, hop, win, nmel, fmin, fmax = _cfg_preproc(cfg)
        model_cfg = cfg.get("model", {})
        mparams   = model_cfg.get("params", {}) or {}
        ckpt_path = model_cfg.get("reload_from_ckpt")
        cond_cfg  = (mparams.get("cond_stage_config") or model_cfg.get("cond_stage_config") or {})
        clap_ckpt = (cond_cfg.get("params", {}) or {}).get("pretrained_path")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Generation Run Log\n")
            f.write(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"config: {resolve_path(config_path)}\n")
            f.write(f"devices: {', '.join(devices)}\n")
            f.write(f"diffusion_ckpt: {ckpt_path}\n")
            f.write(f"clap_ckpt: {clap_ckpt}\n")
            pth = cfg.get("preprocess_config", {}).get("path", {})
            f.write(f"train_data: {pth.get('train_data')}\n")
            f.write(f"test_data: {pth.get('test_data')}\n")
            f.write(f"class_label_index: {pth.get('class_label_index')}\n")
            f.write(f"audio.sampling_rate: {sr}\n")
            f.write(f"stft: n_fft={nfft} hop={hop} win={win}\n")
            f.write(f"mel: n_mels={nmel} fmin={fmin} fmax={fmax}\n")
            f.write(f"latent: t={mparams.get('latent_t_size')} f={mparams.get('latent_f_size')} ch={mparams.get('channels')}\n")
            evalp = mparams.get("evaluation_params") or {}
            f.write(f"sampling: steps={evalp.get('ddim_sampling_steps')} cfg_scale={evalp.get('unconditional_guidance_scale')} eta={evalp.get('ddim_eta', 0.0)}\n")
            f.write(f"{'-'*80}\n")
            f.write("idx\tclass\tmid\twav\tmel_img\tstft_img\twav_shape\tdur_sec\tmel_shape\n")
            f.write(f"{'-'*80}\n")

def append_item_log(out_dir: str, idx: int, disp: str, mid: str,
                    wav_path: str, mel_img: str, stft_img: str,
                    wav: torch.Tensor, mel: torch.Tensor, sr: int):
    path = os.path.join(out_dir, "generation_log.txt")
    w = wav.detach().cpu()
    if w.dim() == 3 and w.size(1) == 1: w = w.squeeze(1)
    if w.dim() == 1: w = w.unsqueeze(0)
    T = w.shape[-1]; dur = T / float(sr)
    line = f"{idx}\t{disp}\t{mid}\t{os.path.basename(wav_path)}\t{os.path.basename(mel_img)}\t{os.path.basename(stft_img)}\t{tuple(w.shape)}\t{dur:.3f}\t{tuple(mel.shape)}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)

# Worker
def worker(worker_id: int, device: str, cfg: Dict[str, Any],
           jobs: List[Tuple[int, Dict[str, str]]], out_dir: str):
    device_tag = f"cuda:{device.split(':')[-1]}" if device.startswith("cuda") else device
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[-1]))
    try:
        # Ensure we're in text mode
        mparams  = cfg.get("model", {}).get("params", {}) or {}
        cond_cfg = (mparams.get("cond_stage_config") or cfg.get("model", {}).get("cond_stage_config") or {})
        embed_mode = (cond_cfg.get("params", {}) or {}).get("embed_mode", "text")
        if embed_mode != "text":
            log(f"WARNING: embed_mode='{embed_mode}' but generator expects text. Forcing text prompts.", device_tag)

        model = build_model(cfg, device, device_tag)
        log(f"ready. Generating {len(jobs)} items.", device_tag)

        eval_p = mparams.get("evaluation_params") or {}
        ddim_steps = int(eval_p.get("ddim_sampling_steps", 200))
        eta        = float(eval_p.get("ddim_eta", 0.0))
        cfg_scale  = float(eval_p.get("unconditional_guidance_scale", 2.0))
        sr, *_ = _cfg_preproc(cfg)

        for (idx, row) in jobs:
            mid  = row.get("mid", "")
            disp = row.get("display_name", mid).strip()
            rep_i = row.get("_rep", 0)
            label_for_print = f"({(mid.replace('/', ' ').strip() + ' ' + disp).strip()})" if (disp or mid) else ""
            try:
                prompt_text = (disp or mid or "sound effect").strip()
                name_base   = (disp or mid or "sample").strip()
                cond_y, uncond_y, eff_scale = make_film_condition_text(model, prompt_text, device, cfg_scale)

                wav, mel = sample_and_decode(model, cond_y, uncond_y, eff_scale, ddim_steps, eta, cfg)

                base = name_base.lower().replace(" ", "_").replace("/", "_").replace(",", "")
                rep_suffix = f"_r{rep_i}" if rep_i else ""
                fname = f"{idx:05d}_{base}{rep_suffix}.wav"
                out_path = os.path.join(out_dir, fname)
                save_waveform(wav, sr, out_path)

                mel_png  = out_path.replace(".wav", "_mel.png")
                stft_png = out_path.replace(".wav", "_stft.png")
                save_mel_spectrogram_image(mel, cfg, mel_png)
                save_stft_spectrogram_image(wav, cfg, stft_png)

                append_item_log(out_dir, idx, disp or name_base, mid, out_path, mel_png, stft_png, wav, mel, sr)
                log(f"✔️  idx={idx} {label_for_print} -> {os.path.basename(out_path)}", device_tag)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                log(f"❌ idx={idx} {label_for_print} failed: {type(e).__name__}: {str(e)}", device_tag)
                traceback.print_exc()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log(f"FATAL worker error: {type(e).__name__}: {str(e)}", device_tag)
        print(traceback.format_exc(), flush=True)

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="/home/mukh1485/media_project_update2/configs/dcase7_finetune.yaml")
    parser.add_argument("--out", type=str, default=None,
                        help="/home/mukh1485/media_project_update2/results_dcase7_finetuned/")
    parser.add_argument("--per_class", type=int, default=1,
                        help="Number of files to generate per class.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # Output dir
    log_dir = cfg.get("log_directory") or os.path.join(REPO_DIR, "model_logs")
    out_dir = args.out or os.path.join(log_dir, "results", "generations")
    safe_mkdir(out_dir)

    # Devices & processes
    devices = device_list_from_env()
    n_workers = 1 if (not devices or devices[0] == "cpu") else len(devices)
    print(f"Using {n_workers} device(s): {', '.join(devices)}", flush=True)

    # Run log
    init_run_log(out_dir, cfg, args.config, devices)

    # Read classes from CSV
    class_csv = cfg.get("preprocess_config", {}).get("path", {}).get("class_label_index", None)
    if not class_csv or not os.path.isfile(class_csv):
        raise FileNotFoundError(f"class_label_index not found: {class_csv}")
    base_rows = read_class_index_csv(class_csv)
    if not base_rows:
        raise RuntimeError("No classes found in class_label_index CSV.")

    # Replicate rows per class
    per_class = max(1, int(args.per_class))
    if per_class > 1:
        rows = [dict(r, _rep=j) for r in base_rows for j in range(per_class)]
    else:
        rows = [dict(r, _rep=0) for r in base_rows]

    # Shard and spawn
    shards = shard_jobs(rows, max(1, n_workers))
    procs = []
    mp.set_start_method("fork", force=True)  # Linux default
    for wid in range(len(shards)):
        dev = devices[wid % len(devices)]
        p = mp.Process(target=worker, args=(wid, dev, cfg, shards[wid], out_dir), daemon=False)
        p.start()
        procs.append(p)

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()

if __name__ == "__main__":
    main()