#!/usr/bin/env python3
"""
Script 3: PCA Analysis - Real vs Synthetic (Focused, Class-colored)
===================================================================

Saves FIVE PNGs:
1) overall_pca_combined.png
2) overall_pca_real_and_synth_panels.png
3) pca_valve_real_vs_synth.png
4) pca_pump_real_vs_synth.png
5) pca_toycar_real_vs_synth.png
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import librosa
import soundfile as sf
import torchaudio

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PCAAnalyzer:
    def __init__(self,
                 real_data_path="/home/mukh1485/dcase2023_film/data/processed_machines",
                 synthetic_wav_path="/home/mukh1485/dcase2023_film/generated_wav",
                 output_dir="/home/mukh1485/dcase2023_film/generated_vs_real_analysis/pca_outputs"):

        self.real_data_path = Path(real_data_path)
        self.synthetic_wav_path = Path(synthetic_wav_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.classes = [
            "fan machine sound",
            "pump machine sound",
            "slider machine sound",
            "toy car sound",
            "toy conveyor sound",
            "valve machine sound",
        ]
        self.class_short_names = {
            "fan machine sound": "fan",
            "pump machine sound": "pump",
            "slider machine sound": "slider",
            "toy car sound": "ToyCar",
            "toy conveyor sound": "ToyConveyor",
            "valve machine sound": "valve",
        }

        # class-wise colors
        base_colors = plt.cm.tab10(np.linspace(0, 1, len(self.classes)))
        self.class_color = {cls: base_colors[i] for i, cls in enumerate(self.classes)}

    # ---------- data ----------
    def load_mel_features(self, class_name, n_samples=200, is_real=True):
        mels = []
        if is_real:
            metadata_path = self.real_data_path / "train_metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            short_name = self.class_short_names[class_name]
            class_files = [it for it in metadata if it["class_name"] == short_name][:n_samples]
            for item in tqdm(class_files, desc=f"Loading real {class_name}", leave=False):
                audio_path = self.real_data_path / item["filename"]
                if not audio_path.exists():
                    continue
                try:
                    waveform, sr = torchaudio.load(audio_path)
                    audio = waveform[0].numpy()
                    mel = librosa.feature.melspectrogram(
                        y=audio, sr=22050, n_fft=1024, hop_length=256,
                        n_mels=80, fmin=0, fmax=8000
                    )
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    mel_norm = (np.clip((mel_db + 80) / 80, 0, 1) * 2) - 1
                    T = mel_norm.shape[1]; T_adj = T - (T % 4)
                    mels.append(mel_norm[:, :T_adj].flatten())
                except Exception:
                    continue
        else:
            class_dir = self.synthetic_wav_path / class_name
            if not class_dir.exists():
                return np.array([])
            wav_files = sorted(list(class_dir.glob("*.wav")))[:n_samples]
            for wav_file in tqdm(wav_files, desc=f"Loading synthetic {class_name}", leave=False):
                try:
                    audio, sr = sf.read(wav_file)
                    mel = librosa.feature.melspectrogram(
                        y=audio, sr=22050, n_fft=1024, hop_length=256,
                        n_mels=80, fmin=0, fmax=8000
                    )
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    mel_norm = (np.clip((mel_db + 80) / 80, 0, 1) * 2) - 1
                    T = mel_norm.shape[1]; T_adj = T - (T % 4)
                    mels.append(mel_norm[:, :T_adj].flatten())
                except Exception:
                    continue
        return np.array(mels)

    # ---------- PCA helpers ----------
    def overall_pca(self, per_class_samples=200):
        all_real, all_synth, real_labels, synth_labels = [], [], [], []
        for class_name in self.classes:
            real = self.load_mel_features(class_name, n_samples=per_class_samples, is_real=True)
            synth = self.load_mel_features(class_name, n_samples=per_class_samples, is_real=False)
            if len(real) > 0:
                all_real.append(real)
                real_labels.extend([class_name] * len(real))
            if len(synth) > 0:
                all_synth.append(synth)
                synth_labels.extend([class_name] * len(synth))

        if not all_real or not all_synth:
            raise RuntimeError("No data loaded for overall PCA.")

        X_real = np.vstack(all_real)
        X_synth = np.vstack(all_synth)

        scaler = StandardScaler()
        X_combined = np.vstack([X_real, X_synth])
        X_scaled = scaler.fit_transform(X_combined)

        pca = PCA(n_components=2)
        Z = pca.fit_transform(X_scaled)

        Z_real = Z[: len(X_real)]
        Z_synth = Z[len(X_real):]

        return {
            "real_pca": Z_real,
            "synth_pca": Z_synth,
            "real_labels": real_labels,
            "synth_labels": synth_labels,
            "pca": pca,
        }

    def per_class_pca(self, class_name, n_samples=200):
        X_real = self.load_mel_features(class_name, n_samples=n_samples, is_real=True)
        X_synth = self.load_mel_features(class_name, n_samples=n_samples, is_real=False)
        if len(X_real) == 0 or len(X_synth) == 0:
            raise RuntimeError(f"Insufficient data for {class_name}")

        scaler = StandardScaler()
        X = np.vstack([X_real, X_synth])
        Xs = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        Z = pca.fit_transform(Xs)
        Z_real = Z[: len(X_real)]
        Z_synth = Z[len(X_real):]
        return {"real_pca": Z_real, "synth_pca": Z_synth, "pca": pca}

    # ---------- Plots ----------
    def save_overall_combined(self, res):
        Zr, Zs, pca = res["real_pca"], res["synth_pca"], res["pca"]
        rlabs, slabs = np.array(res["real_labels"]), np.array(res["synth_labels"])
        fig, ax = plt.subplots(figsize=(11, 9))
        for cls in self.classes:
            color = self.class_color[cls]
            mr = (rlabs == cls); ms = (slabs == cls)
            if mr.any():
                ax.scatter(Zr[mr, 0], Zr[mr, 1], s=22, alpha=0.65, marker="o",
                           color=color, label=f"{self.class_short_names[cls]} (real)")
            if ms.any():
                ax.scatter(Zs[ms, 0], Zs[ms, 1], s=22, alpha=0.95, marker="x",
                           color=color, label=f"{self.class_short_names[cls]} (gen)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*200:.1f}%)", fontweight="bold")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*200:.1f}%)", fontweight="bold")
        ax.set_title("Overall PCA: Real vs Synthetic (Class-colored)", fontsize=18, fontweight="bold")
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), fontsize=10, ncol=2)
        ax.grid(alpha=0.3)
        out = self.output_dir / "overall_pca_combined.png"
        plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
        print(f"✅ Saved: {out}")

    def save_overall_real_vs_synth_panels(self, res):
        Zr, Zs, pca = res["real_pca"], res["synth_pca"], res["pca"]
        rlabs, slabs = np.array(res["real_labels"]), np.array(res["synth_labels"])
        fig, axes = plt.subplots(1, 2, figsize=(17, 8), sharex=True, sharey=True)
        ax1, ax2 = axes
        for cls in self.classes:
            color = self.class_color[cls]
            mr = (rlabs == cls)
            if mr.any():
                ax1.scatter(Zr[mr, 0], Zr[mr, 1], s=24, alpha=0.7, marker="o",
                            color=color, label=self.class_short_names[cls])
        ax1.set_title("PCA: Real Data Only (Class-colored)", fontsize=18, fontweight="bold")
        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*200:.1f}%)", fontweight="bold")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*200:.1f}%)", fontweight="bold")
        ax1.legend(fontsize=10, ncol=2); ax1.grid(alpha=0.3)

        for cls in self.classes:
            color = self.class_color[cls]
            ms = (slabs == cls)
            if ms.any():
                ax2.scatter(Zs[ms, 0], Zs[ms, 1], s=24, alpha=0.9, marker="x",
                            color=color, label=self.class_short_names[cls])
        ax2.set_title("PCA: Synthetic Data Only (Class-colored)", fontsize=18, fontweight="bold")
        ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*200:.1f}%)", fontweight="bold")
        ax2.legend(fontsize=10, ncol=2); ax2.grid(alpha=0.3)

        fig.suptitle("Overall PCA — Real vs Synthetic Panels (Shared PCs)", fontsize=18, fontweight="bold")
        out = self.output_dir / "overall_pca_real_and_synth_panels.png"
        plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
        print(f"✅ Saved: {out}")

    def save_per_class_plot(self, class_name, res):
        Zr, Zs, pca = res["real_pca"], res["synth_pca"], res["pca"]
        short = self.class_short_names[class_name]
        color = self.class_color[class_name]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(Zr[:, 0], Zr[:, 1], s=46, alpha=0.75, marker="o",
                   color=color, edgecolors="black", label="Real")
        ax.scatter(Zs[:, 0], Zs[:, 1], s=46, alpha=0.95, marker="x",
                   color=color, label="Synthetic")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*200:.1f}%)", fontweight="bold")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*200:.1f}%)", fontweight="bold")
        ax.set_title(f"{short}: PCA — Real vs Synthetic", fontweight="bold")
        ax.legend(fontsize=10, ncol=2); ax.grid(alpha=0.3)
        out = self.output_dir / f"pca_{short.lower()}_real_vs_synth.png"
        plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
        print(f"✅ Saved: {out}")

    # ---------- pipeline ----------
    def run(self):
        print("\n=== PCA ANALYSIS (FOCUSED, CLASS-COLORED) ===\n")
        overall = self.overall_pca(per_class_samples=200)
        self.save_overall_combined(overall)
        self.save_overall_real_vs_synth_panels(overall)

        # Per-class: Valve, Pump, ToyCar  <-- ToyCar added
        valve_res = self.per_class_pca("valve machine sound", n_samples=200)
        self.save_per_class_plot("valve machine sound", valve_res)

        pump_res = self.per_class_pca("pump machine sound", n_samples=200)
        self.save_per_class_plot("pump machine sound", pump_res)

        toycar_res = self.per_class_pca("toy car sound", n_samples=200)
        self.save_per_class_plot("toy car sound", toycar_res)


if __name__ == "__main__":
    analyzer = PCAAnalyzer()
    analyzer.run()
