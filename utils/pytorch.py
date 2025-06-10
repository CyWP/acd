import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Union, Iterable

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


def inspect_model_parameters(model):
    for name, param in model.named_parameters():
        print(
            f"Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}"
        )


class TanhRelu(nn.Module):
    def forward(self, x):
        return torch.where(x > 0, torch.tanh(x), torch.zeros_like(x))


def apply_to_modules(model, condition_fn, apply_fn):
    """
    Recursively traverses all submodules of `model` and applies `apply_fn`
    to the module if `condition_fn(module)` is True.

    Parameters:
        model (nn.Module): The root model to traverse.
        condition_fn (Callable[[nn.Module], bool]): A lambda or function that returns True for modules of interest.
        apply_fn (Callable[[str, nn.Module], None]): A function that will be called with (name, module) if condition_fn is True.
    """

    def recursive_traverse(prefix, module):
        for name, submodule in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if condition_fn(submodule):
                apply_fn(full_name, submodule)
            recursive_traverse(full_name, submodule)

    recursive_traverse("", model)


# ----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (
        value.shape,
        value.dtype,
        value.tobytes(),
        shape,
        dtype,
        device,
        memory_format,
    )
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(
        value, shape=shape, dtype=dtype, device=device, memory_format=memory_format
    )


class AcDataset(Dataset):

    def __getitem__(self, idx):
        if isinstance(idx, Iterable):
            return AcDataSample.stack([self.get_sample(i) for i in idx])
        return self.get_sample(idx)

    def get_sample(self, idx):
        raise NotImplementedError("This is an abstract base class.")


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

    def copy(self):
        return AcDataSample(super().copy())

    def to(self, device):
        new = {}
        for key, val in self.items():
            new[key] = val.to(device)
        return new

    def __str__(self):
        s = ""
        for key, val in self.items():
            s += key
            try:
                s += f": {val.shape}\n"
            except:
                s += f": {val}\n"
        return s

    def __getitem__(self, idx):
        """
        Override indexing behavior for slices, arrays, integers, and strings.
        """
        if isinstance(idx, str):
            for key, val in self.items():
                if key == idx:
                    return val
            raise AttributeError(f"Attribute {idx} not found.")
        elif isinstance(idx, (int, slice, Iterable)):
            try:
                return AcDataSample({key: val[idx] for key, val in self.items()})
            except:
                breakpoint()
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    @classmethod
    def stack(cls, samples):
        try:
            stacked = AcDataSample()
            if len(samples) == 1:
                return samples[0]
            for key in samples[0].keys():
                stacked[key] = torch.stack([sample[key] for sample in samples])
            return stacked
        except Exception as e:
            for key in samples[0].keys():
                print(
                    [
                        f"{key}: {[(sample[key].shape, sample[key].device) for sample in samples]}"
                    ]
                )
            raise e
