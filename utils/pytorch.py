import torch
from torch.utils.data import DataLoader
from typing import Union

from .easydict import EasyDict


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPUs
    elif torch.backends.mps.is_available():  # Apple Silicon (MPS)
        return torch.device("mps")
    elif torch.has_mps:  # Older versions of PyTorch may use this
        return torch.device("mps")
    elif torch.backends.rocm.is_available():  # AMD GPUs (ROCm)
        return torch.device("rocm")
    elif torch.xpu.is_available():  # Intel GPUs (XPU)
        return torch.device("xpu")
    else:
        return torch.device("cpu")  # Default to CPU


def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(
        value, shape=shape, dtype=dtype, device=device, memory_format=memory_format
    )


class AcDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=16, shuffle=True, num_workers=0):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to stack dictionary values by their keys.
        """
        return AcDataSample.stack(batch)


class AcDataSample(EasyDict):

    def __getitem__(self, idx):
        sample = AcDataSample()
        for key, val in self.items():
            sample[key] = val[idx]
        return sample

    def __len__(self):
        if not self:
            return 0
        for key, val in self.items():
            return val.shape[0]

    @classmethod
    def stack(cls, samples):
        stacked = AcDataSample()
        if samples:
            for key in samples[0]:
                stacked[key] = torch.stack([sample[key] for sample in samples])
        return stacked
