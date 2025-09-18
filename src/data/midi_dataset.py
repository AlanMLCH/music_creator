from typing import List
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from .midi_tokenizer import midi_to_tokens, VOCAB_SIZE, BOS, EOS


class MidiSequenceDataset(Dataset):
    def __init__(self, midi_dir: str, seq_len: int, min_tokens: int = 256):
        self.seq_len = seq_len
        self.samples: List[List[int]] = []
        paths = list(Path(midi_dir).glob('**/*.mid')) + list(Path(midi_dir).glob('**/*.midi'))
        for p in paths:
            toks = midi_to_tokens(str(p))
            if len(toks) < max(min_tokens, seq_len + 1):
                continue
            for i in range(0, len(toks) - (seq_len + 1), seq_len):
                self.samples.append(toks[i:i+seq_len+1])
        if not self.samples:
            raise RuntimeError(f"No sufficiently long MIDI sequences found in {midi_dir}")
        random.shuffle(self.samples)


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int):
        seq = self.samples[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y