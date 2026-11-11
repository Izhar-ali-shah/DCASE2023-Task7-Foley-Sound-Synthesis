import sys
import os
import numpy as np
import argparse
import yaml
import torch
from tqdm import tqdm
import torchaudio

# Make sure local "src" package is visible
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "src"))

from latent_diffusion.models.ddpm import LatentDiffusion


# ---------------------------------------------------------------------
# Paths and config loading
# ---------------------------------------------------------------------

CONFIG_ROOT = os.path.join(ROOT_DIR, "configs")
MODEL_LOGS_ROOT = os.path.join(ROOT_DIR, "model_logs")
RESULTS_ROOT = os.path.join(ROOT_DIR, "results")
RESULTS_MEL_ROOT = os.path.join(ROOT_DIR, "results_mel")  # <--- new root for mel specs

BIG_CONFIG_PATH = os.path.join(CONFIG_ROOT, "big.yaml")
SMALL_CONFIG_PATH = os.path.join(CONFIG_ROOT, "small.yaml")

with open(BIG_CONFIG_PATH, "r") as f:
    big_config = yaml.load(f, Loader=yaml.FullLoader)

with open(SMALL_CONFIG_PATH, "r") as f:
    small_config = yaml.load(f, Loader=yaml.FullLoader)

# Try to extract sample rates from the YAML (fallbacks just in case)
BIG_SAMPLE_RATE = (
    big_config.get("model", {})
    .get("params", {})
    .get("cond_stage_config", {})
    .get("params", {})
    .get("sampling_rate", 16000)
)

SMALL_SAMPLE_RATE = (
    small_config.get("model", {})
    .get("params", {})
    .get("cond_stage_config", {})
    .get("params", {})
    .get("sampling_rate", 22050)
)


def _load_latent_diffusion_from_ckpt(config: dict, ckpt_path: str) -> LatentDiffusion:
    """Helper: build LatentDiffusion from config and load state_dict from ckpt."""
    model = LatentDiffusion(**config["model"]["params"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    # Some checkpoints contain an extra key "transform.kernel"
    if "transform.kernel" in state_dict:
        del state_dict["transform.kernel"]
    model.load_state_dict(state_dict)
    return model


def get_big() -> LatentDiffusion:
    """Big CLAP-based model (clap_submit.ckpt)."""
    ckpt_path = os.path.join(MODEL_LOGS_ROOT, "clap_submit.ckpt")
    return _load_latent_diffusion_from_ckpt(big_config, ckpt_path)


def get_small() -> LatentDiffusion:
    """Small DCASE model (dcase_submit.ckpt)."""
    ckpt_path = os.path.join(MODEL_LOGS_ROOT, "dcase_submit.ckpt")
    return _load_latent_diffusion_from_ckpt(small_config, ckpt_path)


# ---------------------------------------------------------------------
# Instantiate models
# ---------------------------------------------------------------------

_big_model = get_big().cuda()
_small_model = get_small().cuda()

# Use torch.compile if available (PyTorch 2.x)
if hasattr(torch, "compile"):
    big_model = torch.compile(_big_model)
    small_model = torch.compile(_small_model)
else:
    big_model = _big_model
    small_model = _small_model

big_model.eval()
small_model.eval()


# ---------------------------------------------------------------------
# Labels and generation hyperparameters
# ---------------------------------------------------------------------

label_list = [
    "dog_bark",
    "footstep",
    "gunshot",
    "keyboard",
    "moving_motor_vehicle",
    "rain",
    "sneeze_cough",
]

caption_list = [
    "a dog bark",
    "foot steps",
    "gun shot",
    "someone using keyboard",
    "a moving motor",
    "rain",
    "a man cough",
]

# Thresholds for CLAP cosine similarity (used inside the original paper/code)
limit_list = [0.3, 0.03, 0.2, 0.15, 0.75, 0.2, 0.2]

# Number of internal candidates per generate_sample() call
gen_list = [3, 3, 3, 3, 10, 3, 3]

# Unconditional guidance scales
scale_list = [2.5, 2.5, 1.0, 1.0, 3.0, 1.0, 2.5]

# Target indices (used by original submission; -1 means "free")
target_list = [-1, -1, -1, -1, 0, -1, -1]

# Which model to use: 0 = small (DCASE), 1 = big (CLAP)
model_list = [0, 0, 0, 0, 1, 1, 0]


# ---------------------------------------------------------------------
# Utility: save waveform as WAV
# ---------------------------------------------------------------------

def _prepare_waveform_for_saving(waveform: torch.Tensor) -> torch.Tensor:
    """
    Normalize waveform tensor shapes for saving / feature extraction.

    Expects waveform shapes like:
      [1, T], [B, T], [B, C, T], or [T].
    Returns shape [C, T].
    """
    wav = waveform.detach().cpu()

    if wav.dim() == 3:
        # [B, C, T] -> first batch
        wav = wav[0]
    if wav.dim() == 2 and wav.size(0) >= 1:
        # [C, T]
        return wav
    if wav.dim() == 1:
        # [T] -> [1, T]
        wav = wav.unsqueeze(0)
        return wav

    raise RuntimeError(f"Unexpected waveform shape {wav.shape} for saving.")


def _save_waveform(waveform: torch.Tensor, sample_rate: int, out_path: str) -> None:
    """
    Save a single waveform tensor to disk as a .wav file.
    """
    wav = _prepare_waveform_for_saving(waveform)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchaudio.save(out_path, wav, sample_rate)


# ---------------------------------------------------------------------
# Utility: compute & save mel-spectrogram
# ---------------------------------------------------------------------

def _save_mel_spectrogram(waveform: torch.Tensor, sample_rate: int, out_path: str) -> None:
    """
    Compute a log-mel spectrogram from waveform and save as .npy file.

    - waveform: same tensor as for _save_waveform
    - out_path: e.g. results_mel/dog_bark/dog_bark_0001.npy
    """
    wav = _prepare_waveform_for_saving(waveform)  # [C, T]
    # we'll use mono for mel (first channel)
    if wav.size(0) > 1:
        wav = wav[:1, :]  # [1, T]

    # Mel spec params (you can adjust to match training setup if needed)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=64,
        center=True,
        power=2.0,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    mel = mel_transform(wav)       # [1, n_mels, time]
    mel_db = amp_to_db(mel)        # log-mel

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, mel_db.squeeze(0).numpy())  # save as [n_mels, time]


# ---------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------

def generate_sound(event_id: int, quantity: int = 100, model: LatentDiffusion = big_model):
    """
    Generate `quantity` samples for a given event ID using the provided model.

    It:
      * creates a dummy batch (the model only really uses the text condition),
      * calls model.generate_sample() in a small retry loop,
      * saves each accepted waveform into results/<label>/ as .wav,
      * saves a log-mel spectrogram into results_mel/<label>/ as .npy,
      * returns the list of `result` objects.
    """
    if event_id < 0 or event_id >= len(label_list):
        raise ValueError(f"event_id must be in [0, {len(label_list)-1}], got {event_id}")

    label = label_list[event_id]
    caption = caption_list[event_id]
    limit = limit_list[event_id]
    n_candidates = gen_list[event_id]
    scale = scale_list[event_id]
    target = target_list[event_id]

    # Choose sample rate based on which model is being used
    if model is big_model:
        sample_rate = BIG_SAMPLE_RATE
    elif model is small_model:
        sample_rate = SMALL_SAMPLE_RATE
    else:
        sample_rate = BIG_SAMPLE_RATE

    outdir = os.path.join(RESULTS_ROOT, label)
    os.makedirs(outdir, exist_ok=True)

    outdir_mel = os.path.join(RESULTS_MEL_ROOT, label)
    os.makedirs(outdir_mel, exist_ok=True)

    # Make sure the model's internal logger points somewhere valid
    model.logger_save_dir = outdir
    model.logger_project = ""
    model.logger_version = ""

    print(f"[generate_sound] Saving WAVs to: {outdir}", flush=True)
    print(f"[generate_sound] Saving mels to: {outdir_mel}", flush=True)

    result_list = []
    sam = 0

    with model.ema_scope("Plotting"):
        for j in tqdm(range(quantity), desc=f"Generating '{label}'"):
            with torch.no_grad():
                # Use text conditioning
                model.cond_stage_key = "text"
                if hasattr(model.cond_stage_model, "embed_mode"):
                    model.cond_stage_model.embed_mode = "text"

                batch = [
                    torch.rand(1, 400, 64),      # dummy mel / feature
                    torch.rand(1, 400, 512),     # dummy CLAP feature etc.
                    torch.rand(1, 527),          # dummy label embedding
                    (f"{sam}.wav",),             # filename
                    torch.rand(1, 64000),        # dummy raw audio
                    (caption,),                  # text caption
                ]

                gen_kwargs = dict(
                    name=label,
                    unconditional_guidance_scale=scale,
                    ddim_steps=model.evaluation_params["ddim_sampling_steps"],
                    n_gen=n_candidates,
                    limit=limit,
                )
                if target >= 0:
                    gen_kwargs["target"] = target

                attempts = 0
                max_attempts = 3
                saved = False
                waveform = None
                result = None

                while (not saved) and attempts < max_attempts:
                    saved, waveform, result = model.generate_sample([batch], **gen_kwargs)
                    attempts += 1

                if not saved:
                    print(
                        f"  [warn] sample {j} for '{label}' not marked as saved; "
                        f"keeping last waveform.",
                        flush=True,
                    )

                sam += 1

                # Filenames
                base_name = f"{label}_{sam:04d}"
                wav_path = os.path.join(outdir, f"{base_name}.wav")
                mel_path = os.path.join(outdir_mel, f"{base_name}.npy")

                # Save waveform and mel
                _save_waveform(waveform, sample_rate, wav_path)
                _save_mel_spectrogram(waveform, sample_rate, mel_path)

                result_list.append(result)

                if target >= 0:
                    target += 1

    return result_list


def samplying_sound(event_id: int, quantity: int = 100):
    """
    Wrapper that chooses the correct model (big/small) based on model_list[].
    """
    model_id = model_list[event_id]
    if model_id == 0:
        return generate_sound(event_id, quantity=quantity, model=small_model)
    else:
        return generate_sound(event_id, quantity=quantity, model=big_model)


# ---------------------------------------------------------------------
# Optional CLI for direct use (not used by run_generate.py, but handy)
# ---------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Low-level generator for DCASE2023 Task 7 models."
    )
    parser.add_argument(
        "--id",
        type=int,
        required=True,
        help="Event ID (0..6). See label_list.",
    )
    parser.add_argument(
        "--quantity",
        type=int,
        default=5,
        help="Number of samples to generate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = samplying_sound(args.id, quantity=args.quantity)
    print(f"Generated {len(results)} sample(s) for label '{label_list[args.id]}'")
