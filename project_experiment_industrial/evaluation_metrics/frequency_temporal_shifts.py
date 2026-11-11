#!/usr/bin/env python3
"""
Script 2 (Core): Frequency & Temporal Shift (Simple)
====================================================

This minimal script measures only:
- Frequency shift = mean spectral centroid difference (Hz and %)
- Temporal shift = lag (ms) from cross-correlation of average envelopes

Outputs:
- JSON: frequency_temporal_core.json
- PNG : frequency_temporal_core.png  (2 bars: frequency shift & temporal lag)

Usage:
    python analysis_02_frequency_temporal_shifts_core.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import librosa
import soundfile as sf
import torchaudio
from scipy.signal import correlate

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FrequencyTemporalCore:
    def __init__(self,
                 real_data_path="/home/mukh1485/dcase2023_film/data/processed_machines",
                 synthetic_wav_path="/home/mukh1485/dcase2023_film/generated_wav",
                 output_dir="/home/mukh1485/dcase2023_film/generated_vs_real_analysis/frequency_temporal_core",
                 sr=22050,
                 n_samples=200):
        self.real_data_path = Path(real_data_path)
        self.synthetic_wav_path = Path(synthetic_wav_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sr = sr
        self.n_samples = n_samples

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

    # -------------------
    # Loading helpers
    # -------------------
    def load_real_audio(self, class_name):
        """Load up to n_samples real audio clips for a class."""
        metadata_path = self.real_data_path / "train_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        short = self.class_short_names[class_name]
        items = [it for it in metadata if it["class_name"] == short][: self.n_samples]

        audios = []
        for it in items:
            p = self.real_data_path / it["filename"]
            if not p.exists():
                continue
            try:
                wav, _ = torchaudio.load(p)
                audios.append(wav[0].numpy())
            except Exception:
                continue
        return audios

    def load_synthetic_audio(self, class_name):
        """Load up to n_samples synthetic audio clips for a class."""
        cdir = self.synthetic_wav_path / class_name
        if not cdir.exists():
            return []

        files = sorted(list(cdir.glob("*.wav")))[: self.n_samples]
        audios = []
        for f in files:
            try:
                y, _ = sf.read(f)
                audios.append(y)
            except Exception:
                continue
        return audios

    # -------------------
    # Core metrics
    # -------------------
    def _mean_spectral_centroid(self, audios):
        """Mean spectral centroid (Hz) across clips."""
        vals = []
        for y in audios:
            try:
                c = librosa.feature.spectral_centroid(y=y, sr=self.sr)
                vals.append(float(c.mean()))
            except Exception:
                continue
        return float(np.mean(vals)) if len(vals) else float("nan")

    def _avg_envelope(self, audios, env_sr=1000):
        """Average amplitude envelope resampled to env_sr."""
        envs = []
        for y in audios:
            try:
                # simple absolute amplitude envelope
                env = np.abs(y)
                env = librosa.resample(env, orig_sr=self.sr, target_sr=env_sr)
                envs.append(env)
            except Exception:
                continue
        if not envs:
            return np.array([]), env_sr
        # pad/truncate to common length
        L = min(len(e) for e in envs)
        envs = [e[:L] for e in envs]
        return np.mean(envs, axis=0), env_sr

    def _temporal_lag_ms(self, env_real, env_synth, env_sr):
        """Lag (ms) where cross-correlation is maximal.
        Positive lag => Synthetic lags behind Real (arrives later).
        """
        if len(env_real) == 0 or len(env_synth) == 0:
            return float("nan")
        L = min(len(env_real), len(env_synth))
        r = env_real[:L]
        s = env_synth[:L]
        c = correlate(r - r.mean(), s - s.mean(), mode="full")
        lags = np.arange(-L + 1, L)
        lag_samples = int(lags[np.argmax(c)])
        lag_ms = lag_samples / env_sr * 1000.0
        return float(lag_ms)

    def analyze_class(self, class_name):
        """Return minimal metrics for one class."""
        real = self.load_real_audio(class_name)
        synth = self.load_synthetic_audio(class_name)

        if len(real) == 0 or len(synth) == 0:
            print(f"Skipping {class_name}: no data.")
            return None

        # Frequency shift via spectral centroid
        c_real = self._mean_spectral_centroid(real)
        c_synth = self._mean_spectral_centroid(synth)
        freq_shift_hz = c_synth - c_real
        freq_shift_pct = (freq_shift_hz / c_real * 100.0) if np.isfinite(c_real) and c_real != 0 else float("nan")

        # Temporal shift via cross-correlation of average envelopes
        env_r, env_sr = self._avg_envelope(real)
        env_s, _ = self._avg_envelope(synth)
        temporal_lag_ms = self._temporal_lag_ms(env_r, env_s, env_sr)

        return {
            "class": class_name,
            "short": self.class_short_names[class_name],
            "centroid_real_hz": c_real,
            "centroid_synth_hz": c_synth,
            "frequency_shift_hz": freq_shift_hz,     # +ve => synthetic brighter
            "frequency_shift_percent": freq_shift_pct,
            "temporal_lag_ms": temporal_lag_ms,      # +ve => synthetic later
        }

    # -------------------
    # Viz + Save
    
    def visualize_combined(self, results):
        import numpy as np
        import matplotlib.pyplot as plt

        R = [r for r in results if r is not None]
        if not R:
            print("Nothing to visualize.")
            return

        # ---- prep data
        labels = np.array([r["short"] for r in R])
        real   = np.array([r["centroid_real_hz"] for r in R], dtype=float)
        synth  = np.array([r["centroid_synth_hz"] for r in R], dtype=float)
        dHz    = synth - real
        lagms  = np.array([r["temporal_lag_ms"] for r in R], dtype=float)

        # order by |frequency shift|
        order = np.argsort(np.abs(dHz))[::-1]
        labels_o, real_o, synth_o, dHz_o = labels[order], real[order], synth[order], dHz[order]

        # ---- figure
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.35)

        # =========================
        # TOP: Dumbbell (centroids)
        # =========================
        ax1 = fig.add_subplot(gs[0, 0])
        y = np.arange(len(labels_o))

        # connecting lines
        for i in range(len(y)):
            ax1.plot([real_o[i], synth_o[i]], [y[i], y[i]],
                    '-', color='gray', alpha=0.6, linewidth=2)

        ax1.scatter(real_o,  y, s=70, label='Real',       zorder=3, color='#ff6f91', edgecolor='black', linewidth=0.6)
        ax1.scatter(synth_o, y, s=70, label='Synthetic',  zorder=3, color='#b39b2e', edgecolor='black', linewidth=0.6)

        ax1.set_yticks(y)
        ax1.set_yticklabels(labels_o)
        ax1.set_xlabel('Spectral Centroid (Hz)', fontweight='bold')
        ax1.set_title('Real vs Synthetic Spectral Centroids', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.legend(loc='upper right')

        # =========================
        # BOTTOM: Quadrant scatter
        # =========================
        ax2 = fig.add_subplot(gs[1, 0])

        # color by quadrant
        colors = []
        for x, yv in zip(dHz, lagms):
            if x >= 0 and yv >= 0: colors.append('tab:red')      # brighter & later
            elif x < 0 and yv >= 0: colors.append('tab:orange')  # darker & later
            elif x < 0 and yv < 0: colors.append('tab:blue')     # darker & earlier
            else: colors.append('tab:green')                      # brighter & earlier

        ax2.axvline(0, color='black', linestyle='--', linewidth=1)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)

        ax2.scatter(dHz, lagms, s=120, edgecolor='black', linewidth=0.7, color=colors, zorder=3)

        # label each point
        for xi, yi, lab in zip(dHz, lagms, labels):
            ax2.text(xi, yi, f'  {lab}', va='center', fontsize=10)

        ax2.set_xlabel('Frequency Shift (Hz)  [synthetic − real]', fontweight='bold')
        ax2.set_ylabel('Temporal Lag (ms)     [synthetic later = +]', fontweight='bold')
        ax2.set_title('Shift Map: Frequency vs Time', fontweight='bold')
        ax2.grid(alpha=0.25)

        # optional helpful bands
        ax2.fill_betweenx(ax2.get_ylim(), -100, 100, color='gray', alpha=0.08, zorder=0, label='±100 Hz band')
        ax2.fill_between(ax2.get_xlim(), -50, 50, color='gray', alpha=0.06, zorder=0, label='±50 ms band')
        ax2.legend(loc='upper left')

        out = self.output_dir / 'frequency_temporal_combined.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {out}")

    def run(self):
        print("\n" + "=" * 80)
        print("CORE FREQUENCY & TEMPORAL SHIFT ANALYSIS")
        print("=" * 80 + "\n")

        results = []
        for cls in self.classes:
            print(f"Processing: {cls}")
            r = self.analyze_class(cls)
            if r:
                results.append(r)

        # Save JSON (clean & minimal)
        out_json = self.output_dir / "frequency_temporal.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved: {out_json}")

        # Simple visualization (single PNG)
        self.visualize_combined(results)

        print("\n" + "=" * 80)
        print("DONE")
        print("=" * 80 + "\n")

        return results


if __name__ == "__main__":
    analyzer = FrequencyTemporalCore()
    analyzer.run()
