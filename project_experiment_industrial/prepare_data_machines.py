#!/usr/bin/env python3
"""
Data preparation script
Prepares audio files and creates metadata for training
"""

import os
import json
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# Configuration
DATASET_PATH = "/path/to/dataset/dev"
EVAL_PATH = "/path/to/dataset/eval"
OUTPUT_PATH = "/path/to/data/processed_machines"
SAMPLE_RATE = 22050 
DURATION = 4.0  # seconds
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)  # 88200 samples

# 6 machine sound classes (for my case / change)
CLASS_MAPPING = {
    "fan": 0,
    "pump": 1,
    "slider": 2,
    "ToyCar": 3,
    "ToyConveyor": 4,
    "valve": 5
}

# Can be use any or later at generation.

TEXT_PROMPTS = {
    0: "fan machine sound",
    1: "pump machine sound",
    2: "slider machine sound",
    3: "toy car sound",
    4: "toy conveyor sound",
    5: "valve machine sound"
}


def process_audio(audio_path, target_length=TARGET_LENGTH, sr=SAMPLE_RATE):
    """Load and process audio to standard format"""
    try:
        # Load audio
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        # Ensure correct length (pad or trim)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        return audio
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def prepare_dataset():
    """Prepare dataset for training"""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    metadata = []
    file_counter = 0
    
    print("Processing DCASE 2020 Machine Sounds Dataset...")
    # print(f"Dev path: {DATASET_PATH}")
    # print(f"Eval path: {EVAL_PATH}")
    print(f"Output path: {OUTPUT_PATH}\n")
    
    # Process both dev and eval directories
    for dataset_type, base_path in [("dev", DATASET_PATH), ("eval", EVAL_PATH)]:
        print(f"\n{'='*80}")
        print(f"Processing {dataset_type.upper()} set")
        print(f"{'='*80}")
        
        for class_name, class_id in CLASS_MAPPING.items():
            class_path = Path(base_path) / class_name
            
            if not class_path.exists():
                print(f" Warning: {class_path} does not exist, skipping...")
                continue
            
            # Find all .wav files (including subdirectories)
            audio_files = list(class_path.rglob("*.wav"))
            print(f"\n{class_name}: Found {len(audio_files)} files")
            
            if len(audio_files) == 0:
                print(f" No wav files found in {class_path}")
                continue
            
            for audio_file in tqdm(audio_files, desc=f"  Processing {class_name}"):
                # Process audio
                audio = process_audio(str(audio_file))
                
                if audio is None:
                    continue
                
                # Save processed audio
                output_filename = f"{class_name}_{file_counter:05d}.wav"
                output_path = os.path.join(OUTPUT_PATH, output_filename)
                sf.write(output_path, audio, SAMPLE_RATE)
                
                # Add to metadata
                metadata.append({
                    "filename": output_filename,
                    "class_id": class_id,
                    "class_name": class_name,
                    "text_prompt": TEXT_PROMPTS[class_id],
                    "original_file": str(audio_file),
                    "dataset_type": dataset_type,
                    "duration": DURATION,
                    "sample_rate": SAMPLE_RATE
                })
                
                file_counter += 1
    
    # Save full metadata
    metadata_path = os.path.join(OUTPUT_PATH, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f" Processed {file_counter} files total")
    print(f" Metadata saved to {metadata_path}")
    print(f"{'='*80}")
    
    # Create train/val split (90/10)
    np.random.seed(42)
    indices = np.random.permutation(len(metadata))
    split_idx = int(0.9 * len(metadata))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_metadata = [metadata[i] for i in train_indices]
    val_metadata = [metadata[i] for i in val_indices]
    
    # Save splits
    with open(os.path.join(OUTPUT_PATH, "train_metadata.json"), 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(os.path.join(OUTPUT_PATH, "val_metadata.json"), 'w') as f:
        json.dump(val_metadata, f, indent=2)
    
    print(f"\n Train set: {len(train_metadata)} samples")
    print(f" Validation set: {len(val_metadata)} samples")
    
    # Print class distribution
    print(f"\n{'='*80}")
    print("Class distribution:")
    print(f"{'='*80}")
    
    for class_name, class_id in CLASS_MAPPING.items():
        train_count = sum(1 for m in train_metadata if m['class_id'] == class_id)
        val_count = sum(1 for m in val_metadata if m['class_id'] == class_id)
        total_count = train_count + val_count
        print(f"  {class_name:15s}: {total_count:4d} total (train: {train_count:4d}, val: {val_count:3d})")
    
    print(f"\n{'='*80}")
    print(" Data preparation complete!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    prepare_dataset()