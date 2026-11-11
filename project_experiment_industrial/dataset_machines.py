#!/usr/bin/env python3
"""
Dataset loader for training
Properly shapes mel-spectrograms 
"""

import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa


class MachineDataset(Dataset):
    
    def __init__(self, metadata_path, data_root, transform=None):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.data_root = data_root
        self.transform = transform
        self.sample_rate = 22050
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 80
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load audio
        audio_path = f"{self.data_root}/{item['filename']}"
        waveform, sr = torchaudio.load(audio_path)
        
        # Ensure correct shape [1, T]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Convert to mel-spectrogram
        mel_spec = self._wav_to_mel(waveform[0].numpy())
        
        # Add channel dimension: [80, 345] -> [1, 80, 345]
        mel_spec = mel_spec[np.newaxis, ...]
        
        return {
            'mel': torch.FloatTensor(mel_spec),
            'class_id': item['class_id'],
            'text_prompt': item['text_prompt'],
            'filename': item['filename']
        }
    
    def _wav_to_mel(self, audio):
        """Convert waveform to mel-spectrogram"""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=0,
            fmax=min(8000, self.sample_rate // 2)
        )
        
        # Convert to log scale
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to [-1, 1]
        mel_normalized = (mel_db + 80) / 80  # Assuming -80 to 0 dB range
        mel_normalized = np.clip(mel_normalized, 0, 1) * 2 - 1
        
        # Adjust time dimension to be divisible by 4
        T = mel_normalized.shape[1]
        T_adj = T - (T % 4)  # e.g., 345 -> 344
        mel_normalized = mel_normalized[:, :T_adj]
        
        return mel_normalized


def get_dataloaders(batch_size=8, num_workers=4, data_root='data/processed_machines'):
    """Create train and validation dataloaders"""
    
    train_dataset = MachineDataset(
        metadata_path=f'{data_root}/train_metadata.json',
        data_root=data_root
    )
    
    val_dataset = MachineDataset(
        metadata_path=f'{data_root}/val_metadata.json',
        data_root=data_root
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing dataset...")
    train_loader, val_loader = get_dataloaders(batch_size=2)
    
    # Get a batch
    batch = next(iter(train_loader))
    
    print(f"\nBatch info:")
    # print(f"  Mel shape: {batch['mel'].shape}") 
    # print(f"  Class IDs: {batch['class_id']}")
    # print(f"  Text prompts: {batch['text_prompt']}")
    
    print(f"\nExpected shape: [batch_size, 1, 80, 344]")
    print(f"Actual shape: {batch['mel'].shape}")
    
    if batch['mel'].shape[1:] == (1, 80, 344):
        print(" Shape is correct")
    else:
        print(" Shape might need adjustment")