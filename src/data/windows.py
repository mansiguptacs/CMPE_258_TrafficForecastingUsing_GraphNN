import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficWindowDataset(Dataset):
    """Sliding windows over one split.

    Each item is (x, y, y_mask):
      x      [input_len, N, 1]  normalized, imputed speeds
      y      [output_len, N]    raw speeds in mph (targets stay unnormalized for metrics)
      y_mask [output_len, N]    True where the target reading is real
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray, observed: np.ndarray,
                 start: int, end: int, input_len: int, output_len: int):
        self.inputs = torch.from_numpy(inputs[start:end])
        self.targets = torch.from_numpy(targets[start:end])
        self.observed = torch.from_numpy(observed[start:end])
        self.input_len = input_len
        self.output_len = output_len
        self.length = (end - start) - input_len - output_len + 1
        if self.length <= 0:
            raise ValueError("split is shorter than one window")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        j = i + self.input_len
        k = j + self.output_len
        return self.inputs[i:j].unsqueeze(-1), self.targets[j:k], self.observed[j:k]
