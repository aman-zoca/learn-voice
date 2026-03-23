"""
Base Dataset Module
===================
Base classes and utilities for TTS datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Callable
from dataclasses import dataclass
import json
import random

from ...02_tts_core.preprocessing.audio.audio_loader import AudioLoader, AudioConfig
from ...02_tts_core.preprocessing.audio.feature_extractor import MelSpectrogramExtractor, FeatureConfig
from ...02_tts_core.preprocessing.text.tokenizer import TextTokenizer


@dataclass
class DatasetConfig:
    """Configuration for TTS datasets."""
    # Paths
    data_dir: str = ""
    metadata_file: str = "metadata.csv"

    # Audio settings
    sample_rate: int = 22050
    n_mels: int = 80
    n_fft: int = 2048
    hop_length: int = 256
    win_length: int = 1024
    f_min: float = 0.0
    f_max: float = 8000.0

    # Text settings
    language: str = 'en'
    use_phonemes: bool = False

    # Processing
    max_audio_len: int = 10 * 22050  # 10 seconds
    max_text_len: int = 200
    min_audio_len: int = 22050 // 2  # 0.5 seconds

    # Training
    batch_size: int = 16
    num_workers: int = 4


class TTSDataset(Dataset):
    """
    Base TTS dataset.

    Loads audio files and corresponding text, extracts features,
    and prepares batches for training.
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: TextTokenizer,
        split: str = 'train'
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split

        # Initialize processors
        self.audio_loader = AudioLoader(AudioConfig(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels
        ))

        self.mel_extractor = MelSpectrogramExtractor(FeatureConfig(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max
        ))

        # Load metadata
        self.samples = self._load_metadata()

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load dataset metadata."""
        data_dir = Path(self.config.data_dir)
        metadata_path = data_dir / self.config.metadata_file

        samples = []

        if metadata_path.suffix == '.csv':
            samples = self._load_csv_metadata(metadata_path)
        elif metadata_path.suffix == '.json':
            samples = self._load_json_metadata(metadata_path)
        elif metadata_path.suffix == '.txt':
            samples = self._load_txt_metadata(metadata_path)

        # Filter by split if needed
        if self.split:
            samples = [s for s in samples if s.get('split', 'train') == self.split]

        return samples

    def _load_csv_metadata(self, path: Path) -> List[Dict[str, Any]]:
        """Load metadata from CSV file."""
        samples = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('|')
                if len(parts) >= 2:
                    samples.append({
                        'id': parts[0],
                        'text': parts[1],
                        'audio_path': str(Path(self.config.data_dir) / 'wavs' / f"{parts[0]}.wav")
                    })

        return samples

    def _load_json_metadata(self, path: Path) -> List[Dict[str, Any]]:
        """Load metadata from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'samples' in data:
            return data['samples']

        return []

    def _load_txt_metadata(self, path: Path) -> List[Dict[str, Any]]:
        """Load metadata from text file (LJSpeech format)."""
        samples = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('|')
                if len(parts) >= 2:
                    samples.append({
                        'id': parts[0],
                        'text': parts[-1],  # Use last column for normalized text
                        'audio_path': str(Path(self.config.data_dir) / 'wavs' / f"{parts[0]}.wav")
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load and process audio
        audio_path = sample['audio_path']
        audio, sr = self.audio_loader.load(audio_path)

        # Trim/pad audio
        if len(audio) > self.config.max_audio_len:
            audio = audio[:self.config.max_audio_len]

        # Extract mel spectrogram
        mel = self.mel_extractor.mel_spectrogram(audio, normalize=True)

        # Process text
        text = sample['text']
        tokens = self.tokenizer.encode(text)

        # Trim text if needed
        if len(tokens) > self.config.max_text_len:
            tokens = tokens[:self.config.max_text_len]

        return {
            'text': torch.LongTensor(tokens),
            'text_length': len(tokens),
            'mel': torch.FloatTensor(mel),
            'mel_length': mel.shape[1],
            'audio': torch.FloatTensor(audio),
            'audio_length': len(audio),
            'id': sample['id']
        }


class TTSCollator:
    """
    Collator for TTS batches.

    Handles padding and creates attention masks.
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Get batch info
        batch_size = len(batch)

        # Get max lengths
        max_text_len = max(item['text_length'] for item in batch)
        max_mel_len = max(item['mel_length'] for item in batch)
        max_audio_len = max(item['audio_length'] for item in batch)
        n_mels = batch[0]['mel'].shape[0]

        # Initialize tensors
        text_padded = torch.full((batch_size, max_text_len), self.pad_token_id, dtype=torch.long)
        mel_padded = torch.zeros(batch_size, n_mels, max_mel_len)
        audio_padded = torch.zeros(batch_size, max_audio_len)

        text_lengths = torch.zeros(batch_size, dtype=torch.long)
        mel_lengths = torch.zeros(batch_size, dtype=torch.long)
        audio_lengths = torch.zeros(batch_size, dtype=torch.long)

        ids = []

        # Fill tensors
        for i, item in enumerate(batch):
            text_len = item['text_length']
            mel_len = item['mel_length']
            audio_len = item['audio_length']

            text_padded[i, :text_len] = item['text']
            mel_padded[i, :, :mel_len] = item['mel']
            audio_padded[i, :audio_len] = item['audio']

            text_lengths[i] = text_len
            mel_lengths[i] = mel_len
            audio_lengths[i] = audio_len

            ids.append(item['id'])

        # Sort by text length (descending) for efficient packing
        sorted_indices = torch.argsort(text_lengths, descending=True)

        return {
            'text': text_padded[sorted_indices],
            'text_lengths': text_lengths[sorted_indices],
            'mel': mel_padded[sorted_indices],
            'mel_lengths': mel_lengths[sorted_indices],
            'audio': audio_padded[sorted_indices],
            'audio_lengths': audio_lengths[sorted_indices],
            'ids': [ids[i] for i in sorted_indices]
        }


class LJSpeechDataset(TTSDataset):
    """
    LJSpeech dataset.

    A popular single-speaker English TTS dataset.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: TextTokenizer,
        split: str = 'train',
        train_ratio: float = 0.95
    ):
        config = DatasetConfig(
            data_dir=data_dir,
            metadata_file='metadata.csv'
        )

        # Load all samples first
        super().__init__(config, tokenizer, split=None)

        # Split dataset
        random.seed(42)
        all_samples = self.samples.copy()
        random.shuffle(all_samples)

        split_idx = int(len(all_samples) * train_ratio)

        if split == 'train':
            self.samples = all_samples[:split_idx]
        elif split == 'val' or split == 'valid':
            self.samples = all_samples[split_idx:]
        elif split == 'all':
            self.samples = all_samples


class VCTKDataset(TTSDataset):
    """
    VCTK dataset.

    Multi-speaker English dataset with various accents.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: TextTokenizer,
        split: str = 'train',
        speakers: Optional[List[str]] = None
    ):
        self.speakers = speakers
        self.speaker_to_id = {}

        config = DatasetConfig(
            data_dir=data_dir,
            metadata_file='metadata.txt'
        )

        super().__init__(config, tokenizer, split)

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load VCTK metadata with speaker info."""
        data_dir = Path(self.config.data_dir)
        samples = []

        # VCTK structure: wavs/pXXX/pXXX_XXX.wav
        wavs_dir = data_dir / 'wavs'

        if wavs_dir.exists():
            for speaker_dir in wavs_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue

                speaker_id = speaker_dir.name

                if self.speakers and speaker_id not in self.speakers:
                    continue

                if speaker_id not in self.speaker_to_id:
                    self.speaker_to_id[speaker_id] = len(self.speaker_to_id)

                # Find corresponding text files
                txt_dir = data_dir / 'txt' / speaker_id

                for wav_file in speaker_dir.glob('*.wav'):
                    txt_file = txt_dir / f"{wav_file.stem}.txt"

                    if txt_file.exists():
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()

                        samples.append({
                            'id': wav_file.stem,
                            'text': text,
                            'audio_path': str(wav_file),
                            'speaker': speaker_id,
                            'speaker_id': self.speaker_to_id[speaker_id]
                        })

        return samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        sample = self.samples[idx]

        item['speaker_id'] = sample['speaker_id']
        item['speaker'] = sample['speaker']

        return item


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pad_token_id: int = 0
) -> DataLoader:
    """
    Create a DataLoader for TTS training.

    Args:
        dataset: TTS dataset
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of worker processes
        pad_token_id: Padding token ID

    Returns:
        DataLoader
    """
    collator = TTSCollator(pad_token_id=pad_token_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )
