#!/usr/bin/env python3
"""
Script 1: Distance Metrics - Real vs Synthetic Data (No Frechet)
================================================================

This script computes distance metrics between real and synthetic
mel-spectrograms to quantify how far apart the distributions are.

Metrics computed:
- Mean Squared Error (MSE) on flattened distributions
- Per-frequency band analysis (average energy per mel band)
- Per-time analysis (average energy per time frame)

It produces THREE PNGs:
1) mse_overview.png                 -> Distribution-level MSE per class
2) toycar_freq_time.png             -> Per-frequency & per-time for ToyCar
3) pump_freq_time.png               -> Per-frequency & per-time for Pump

Usage:
    python analysis_01_distance_metrics.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json

# Style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DistanceAnalyzer:
    def __init__(self,
                 real_data_path="/home/mukh1485/dcase2023_film/data/processed_machines",
                 synthetic_wav_path="/home/mukh1485/dcase2023_film/generated_wav",
                 synthetic_mel_path="/home/mukh1485/dcase2023_film/generated_mels",
                 output_dir="/home/mukh1485/dcase2023_film/generated_vs_real_analysis/distance_analysis"):

        self.real_data_path = Path(real_data_path)
        self.synthetic_wav_path = Path(synthetic_wav_path)
        self.synthetic_mel_path = Path(synthetic_mel_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Full class names used on disk / metadata
        self.classes = [
            "fan machine sound",
            "pump machine sound",
            "slider machine sound",
            "toy car sound",
            "toy conveyor sound",
            "valve machine sound"
        ]

        # Short labels for plots / filenames
        self.class_short_names = {
            "fan machine sound": "fan",
            "pump machine sound": "pump",
            "slider machine sound": "slider",
            "toy car sound": "ToyCar",
            "toy conveyor sound": "ToyConveyor",
            "valve machine sound": "valve"
        }

    # --------------------------
    # Loading helpers
    # --------------------------
    def load_real_mels(self, class_name):
        """Load real mel-spectrograms for a class (converted on the fly)."""
        import librosa
        import torchaudio

        metadata_path = self.real_data_path / "train_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        short_name = self.class_short_names[class_name]
        class_files = [item for item in metadata if item['class_name'] == short_name]

        mels = []
        for item in tqdm(class_files[:200], desc=f"Loading real {class_name}"):
            audio_path = self.real_data_path / item['filename']
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

                # Make time length a multiple of 4 (match generator)
                T = mel_norm.shape[1]
                T_adj = T - (T % 4)
                mels.append(mel_norm[:, :T_adj])
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue
        return np.array(mels)

    def load_synthetic_mels(self, class_name):
        """Load synthetic wavs and convert to mel-spectrograms."""
        import librosa
        import soundfile as sf

        class_dir = self.synthetic_wav_path / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist!")
            return np.array([])

        wav_files = sorted(list(class_dir.glob("*.wav")))[:200]
        mels = []
        for wav_file in tqdm(wav_files, desc=f"Loading synthetic {class_name}"):
            try:
                audio, sr = sf.read(wav_file)
                mel = librosa.feature.melspectrogram(
                    y=audio, sr=22050, n_fft=1024, hop_length=256,
                    n_mels=80, fmin=0, fmax=8000
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_norm = (np.clip((mel_db + 80) / 80, 0, 1) * 2) - 1

                T = mel_norm.shape[1]
                T_adj = T - (T % 4)
                mels.append(mel_norm[:, :T_adj])
            except Exception as e:
                print(f"Error loading {wav_file}: {e}")
                continue
        return np.array(mels)

    # --------------------------
    # Metrics
    # --------------------------
    def compute_distances(self, real_mels, synth_mels, class_name):
        """Compute MSE + per-frequency + per-time metrics (no Frechet)."""
        print(f"\n{'='*80}")
        print(f"Computing distances for: {class_name}")
        print(f"{'='*80}")
        print(f"Real samples: {len(real_mels)} | Synthetic samples: {len(synth_mels)}")

        if len(real_mels) == 0 or len(synth_mels) == 0:
            print("Skipping due to missing data")
            return None

        metrics = {
            'class': class_name,
            'n_real': len(real_mels),
            'n_synthetic': len(synth_mels)
        }

        # Flatten for distribution-level MSE
        real_flat = real_mels.reshape(len(real_mels), -1)
        synth_flat = synth_mels.reshape(len(synth_mels), -1)
        real_dist = real_flat.mean(axis=0)
        synth_dist = synth_flat.mean(axis=0)
        metrics['mse'] = float(np.mean((real_dist - synth_dist) ** 2))

        # Per-frequency (avg over samples & time)
        real_freq_energy = real_mels.mean(axis=(0, 2))
        synth_freq_energy = synth_mels.mean(axis=(0, 2))
        metrics['freq_mse'] = float(np.mean((real_freq_energy - synth_freq_energy) ** 2))
        metrics['real_freq_energy'] = real_freq_energy
        metrics['synth_freq_energy'] = synth_freq_energy

        # Per-time (avg over samples & frequency)
        real_temp_energy = real_mels.mean(axis=(0, 1))
        synth_temp_energy = synth_mels.mean(axis=(0, 1))
        metrics['temp_mse'] = float(np.mean((real_temp_energy - synth_temp_energy) ** 2))
        metrics['real_temp_energy'] = real_temp_energy
        metrics['synth_temp_energy'] = synth_temp_energy

        print(f"RESULTS for {class_name}:")
        print(f"  MSE (distribution): {metrics['mse']:.6f}")
        print(f"  Frequency MSE:      {metrics['freq_mse']:.6f}")
        print(f"  Temporal MSE:       {metrics['temp_mse']:.6f}")
        return metrics

    # --------------------------
    # Visualization
    # --------------------------
    def _save_mse_overview(self, class_metrics):
        """PNG #1: Distribution-level MSE per class."""
        valid = [m for m in class_metrics if m is not None]
        if not valid:
            print("No metrics to plot for MSE overview.")
            return

        classes = [m['class'] for m in valid]
        short = [self.class_short_names[c] for c in classes]
        mse_vals = np.array([m['mse'] for m in valid])

        fig, ax = plt.subplots(figsize=(10, 6))
        # Color scale: higher MSE = 'worse' = redder
        norm = mse_vals / (mse_vals.max() + 1e-12)
        colors = plt.cm.RdYlGn_r(norm)

        bars = ax.bar(range(len(classes)), mse_vals, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(short, rotation=45)
        ax.set_ylabel('MSE', fontweight='bold')
        ax.set_title('Distribution-level MSE (Lower = More Similar)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for i, (b, v) in enumerate(zip(bars, mse_vals)):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

        out = self.output_dir / 'mse_overview.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {out}")

    def _save_class_detail(self, metrics, filename, title_prefix):
        """PNG #2/#3: Per-frequency and Per-time for a single class."""
        if metrics is None:
            print(f"Cannot plot {title_prefix}: no metrics available.")
            return

        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
        ax_freq = fig.add_subplot(gs[0, :])  # top row (wide)
        ax_time = fig.add_subplot(gs[1, :])  # bottom row (wide)

        # Per-frequency
        freqs = np.arange(len(metrics['real_freq_energy']))
        ax_freq.plot(freqs, metrics['real_freq_energy'], label='Real', linewidth=2, alpha=0.85)
        ax_freq.plot(freqs, metrics['synth_freq_energy'], label='Synthetic', linewidth=2, alpha=0.85, linestyle='--')
        ax_freq.fill_between(freqs,
                             metrics['real_freq_energy'],
                             metrics['synth_freq_energy'],
                             alpha=0.25, color='red', label='Difference')
        ax_freq.set_xlabel('Mel Band', fontweight='bold')
        ax_freq.set_ylabel('Energy', fontweight='bold')
        ax_freq.set_title(f'{title_prefix}: Per-frequency Analysis', fontweight='bold')
        ax_freq.legend()
        ax_freq.grid(alpha=0.3)

        # Per-time
        time = np.arange(len(metrics['real_temp_energy']))
        ax_time.plot(time, metrics['real_temp_energy'], label='Real', linewidth=2, alpha=0.85)
        ax_time.plot(time, metrics['synth_temp_energy'], label='Synthetic', linewidth=2, alpha=0.85, linestyle='--')
        ax_time.set_xlabel('Time Frame', fontweight='bold')
        ax_time.set_ylabel('Energy', fontweight='bold')
        ax_time.set_title(f'{title_prefix}: Per-time Analysis', fontweight='bold')
        ax_time.legend()
        ax_time.grid(alpha=0.3)

        plt.suptitle(f'{title_prefix}: Spectral & Temporal Comparison', fontsize=14, fontweight='bold')
        out = self.output_dir / filename
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {out}")

    def visualize_results(self, class_metrics):
        """Create the 3 required PNGs."""
        # 1) Overview MSE per class
        self._save_mse_overview(class_metrics)

        # 2) ToyCar detail
        toycar = next((m for m in class_metrics if m and m['class'] == 'toy car sound'), None)
        self._save_class_detail(toycar, 'toycar_freq_time.png', 'ToyCar')

        # 3) Pump detail
        pump = next((m for m in class_metrics if m and m['class'] == 'pump machine sound'), None)
        self._save_class_detail(pump, 'pump_freq_time.png', 'Pump')

    # --------------------------
    # Pipeline
    # --------------------------
    def run_analysis(self):
        print("\n" + "="*80)
        print("DISTANCE ANALYSIS (No Frechet): Real vs Synthetic")
        print("="*80 + "\n")

        all_metrics = []
        for class_name in self.classes:
            print(f"\n{'*'*80}")
            print(f"Processing: {class_name}")
            print(f"{'*'*80}")

            real_mels = self.load_real_mels(class_name)
            synth_mels = self.load_synthetic_mels(class_name)
            metrics = self.compute_distances(real_mels, synth_mels, class_name)

            if metrics is not None:
                all_metrics.append(metrics)

        # Save JSON (arrays removed for compactness)
        results_to_save = []
        for m in all_metrics:
            mc = m.copy()
            mc.pop('real_freq_energy', None)
            mc.pop('synth_freq_energy', None)
            mc.pop('real_temp_energy', None)
            mc.pop('synth_temp_energy', None)
            results_to_save.append(mc)

        json_path = self.output_dir / 'distance_metrics_no_frechet.json'
        with open(json_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"✅ Saved: {json_path}")

        # Visualize: generates the 3 requested PNGs
        self.visualize_results(all_metrics)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"Results saved to: {self.output_dir}/")
        print("="*80 + "\n")
        return all_metrics


if __name__ == "__main__":
    analyzer = DistanceAnalyzer()
    results = analyzer.run_analysis()
