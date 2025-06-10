import torch

from .utils.pytorch import get_device


class State:

    _device = get_device()

    @classmethod
    def device(cls):
        return cls._device

    @classmethod
    def set_device(cls, device):
        if isinstance(device, str):
            cls._device = torch.device(device)
        if isinstance(device, torch.device):
            cls._device = device
