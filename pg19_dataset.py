import numpy as np
import torch
from torch.utils.data import Dataset


class Pg19Dataset(Dataset):
    def __init__(self, data_path: str, seq_length: int, sliding_window: int = 256):
        assert seq_length >= sliding_window, f"Sliding window '{sliding_window}' must be smaller than sequence length '{seq_length}'"

        self.seq_length = seq_length
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.start_indices = list(range(0, len(self.data) - seq_length, sliding_window))

        assert len(self) > 0, "Dataset is empty"

    def __len__(self):
        # return len(self.start_indices)
        return 8

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        start = self.start_indices[index]
        end = start + self.seq_length

        input_id = torch.from_numpy(self.data[start: end].astype(np.int64))
        label = torch.from_numpy(self.data[start+1: end+1].astype(np.int64))
        return {
            "input_ids": input_id,
            "labels": label
        }

    def num_tokens(self):
        return len(self.data)
