from __future__ import annotations
from typing import List
import os
import argparse
import pickle

import numpy as np
import pandas as pd
import librosa  # only for sanity checks

from create_embeddings_main import main as run_embedding_pipeline
import fad_utils

# The six DCASE2020 machine types
MACHINE_TYPES = ["dog_bark", "footstep", "gunshot", "keyboard", "moving_motor_vehicle", "rain", "sneeze_cough"]

# Real audio: subfolder names under the ground-truth root
REAL_DIR_NAME = {m: m for m in MACHINE_TYPES}

# Generated audio: your current folder names under generated_wav/
GEN_DIR_NAME = {
    "dog_bark": "a dog barking",
    "footstep": "the sound of footsteps",
    "gunshot": "a gunshot",
    "keyboard": "someone using keyboard",
    "moving_motor_vehicle": "a moving motor vehicle",
    "rain": "the sound of rain",
    "sneeze_cough": "sneeze and cough",
}     

class FADWrapperDCASE2020:
    def __init__(
        self,
        gt_root: str,
        gen_root: str,
        sr: int = 22050,
        length_sec: float = 10.0,
        min_files_per_class: int = 1,
        reuse_existing_embeddings: bool = True,
    ) -> None:
        """
        gt_root: directory containing subfolders fan, pump, slider, ToyCar, ToyConveyor, valve
        gen_root: directory containing your generated folders (names in GEN_DIR_NAME)
        sr: expected sample rate (DCASE2020 is 16000 Hz)
        length_sec: expected clip length in seconds (DCASE2020 is ~10 s)
        """
        self.gt_root = gt_root
        self.gen_root = gen_root
        self.sample_rate = sr
        self.length_sec = length_sec
        self.min_files_per_class = min_files_per_class
        self.reuse_existing_embeddings = reuse_existing_embeddings
        # Where we store embeddings and the final CSV
        self.output_root = "./result_dcase2023_foley_eval"

    def _list_wavs(self, set_type: str, machine: str) -> List[str]:
        """List all .wav files for a given set (gt/generated) and machine."""
        if set_type == "gt":
            cls_dir = os.path.join(self.gt_root, REAL_DIR_NAME[machine])
        else:
            cls_dir = os.path.join(self.gen_root, GEN_DIR_NAME[machine])

        files = [
            os.path.join(cls_dir, f)
            for f in sorted(os.listdir(cls_dir))
            if f.lower().endswith(".wav")
        ]

        if len(files) < self.min_files_per_class:
            raise RuntimeError(
                f"Found {len(files)} wavs in {cls_dir}, "
                f"expected at least {self.min_files_per_class}."
            )
        return files

    def _sanity_check(self, paths: List[str]) -> None:
        """
        Basic checks so we don't silently feed wrong audio into VGGish.

        - sample rate matches self.sample_rate
        - length roughly matches self.length_sec (+/- ~1% of sr)
        """
        for p in paths:
            y, sr = librosa.load(p, sr=None, mono=True)
            if sr != self.sample_rate:
                raise ValueError(
                    f"{p}: sample rate {sr} != expected {self.sample_rate}"
                )

            if self.length_sec is not None:
                expected = int(self.sample_rate * self.length_sec)
                # small tolerance for off-by-a-few-samples
                if abs(len(y) - expected) > int(self.sample_rate * 0.01):
                    raise ValueError(
                        f"{p}: length {len(y)} samples, expected about {expected}"
                    )

    def _ensure_embedding(self, set_type: str, machine: str) -> str:
        """
        Make sure we have a *_embedding.pkl for this set_type + machine.
        If it doesn't exist yet, run the embedding pipeline.
        """
        emb_dir = os.path.join(self.output_root, machine)
        os.makedirs(emb_dir, exist_ok=True)

        emb_prefix = os.path.join(emb_dir, set_type)
        emb_pkl = f"{emb_prefix}_embedding.pkl"

        # Reuse previously computed embeddings if desired
        if self.reuse_existing_embeddings and os.path.exists(emb_pkl):
            return emb_pkl

        # Build file list and sanity-check audio
        wavs = self._list_wavs(set_type, machine)
        self._sanity_check(wavs)

        list_path = os.path.join(emb_dir, f"{set_type}_files.txt")
        with open(list_path, "w") as f:
            f.write("\n".join(wavs))

        # Run the Beam + VGGish pipeline from the repo
        run_embedding_pipeline(
            input_file_list_path=list_path,
            output_path=emb_prefix,
        )

        if not os.path.exists(emb_pkl):
            raise RuntimeError(f"Expected embedding file {emb_pkl} not found.")
        return emb_pkl

    @staticmethod
    def _load_mu_sigma(emb_pkl: str):
        """Load μ and Σ from the *_embedding.pkl file."""
        with open(emb_pkl, "rb") as f:
            feature = pickle.load(f)

        mu = np.asarray(feature["mu"].float_list.value)
        emb_len = int(feature["embedding_length"].int64_list.value[0])
        sigma = np.asarray(feature["sigma"].float_list.value).reshape(emb_len, emb_len)
        return mu, sigma

    def compute_fad(self) -> pd.DataFrame:
        """Compute FAD per machine type and save to CSV."""
        result = {"Category": [], "FAD": []}

        for machine in MACHINE_TYPES:
            print(f"Computing FAD for {machine}...")

            gt_emb = self._ensure_embedding("gt", machine)
            gen_emb = self._ensure_embedding("generated", machine)

            mu_gt, sig_gt = self._load_mu_sigma(gt_emb)
            mu_gen, sig_gen = self._load_mu_sigma(gen_emb)

            fad_value = fad_utils.frechet_distance(
                mu_test=mu_gen,
                sigma_test=sig_gen,
                mu_train=mu_gt,
                sigma_train=sig_gt,
            )

            result["Category"].append(machine)
            result["FAD"].append(float(fad_value))

        df = pd.DataFrame(result)
        os.makedirs(self.output_root, exist_ok=True)
        out_csv = os.path.join(self.output_root, "fad_dcase2020.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved FAD scores to {out_csv}")
        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_dir",
        required=True,
        help="Root directory with real DCASE2020 audio "
             "(subfolders fan, pump, slider, ToyCar, ToyConveyor, valve).",
    )
    parser.add_argument(
        "--gen_dir",
        required=True,
        help="Root directory with generated audio "
             "(subfolders listed in GEN_DIR_NAME).",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Expected sample rate (DCASE2020 uses 22050).",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=4.0,
        help="Expected clip length in seconds (DCASE2020 uses ~4.0).",
    )
    args = parser.parse_args()

    wrapper = FADWrapperDCASE2020(
        gt_root=args.gt_dir,
        gen_root=args.gen_dir,
        sr=args.sr,
        length_sec=args.length,
    )
    wrapper.compute_fad()


if __name__ == "__main__":
    main()
