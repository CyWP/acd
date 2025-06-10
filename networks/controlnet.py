import torch
import torch.nn as nn

from ..utils import IMG_KEY, LABEL_KEY, CTL_KEY, persistence
from .edm2 import Block


@persistence.persistent_class
class ControlNetEDM2(nn.Module):

    def __init__(self, edm2_model):
        super().__init__()
        self.net = apply_controlnet(edm2_model, Block)

    def forward(
        self,
        x,
        sigma,
        class_labels,
        force_fp32=False,
        return_logvar=True,
        **unet_kwargs
    ):
        ctl = x[CTL_KEY]
        noise = x[IMG_KEY]
        labels = x[LABEL_KEY]
        ControlNetBlock.set_ctl_image(ctl)
        out = self.model(
            noise,
            sigma,
            class_labels=labels,
            force_fp32=force_fp32,
            return_logvar=return_logvar,
            **unet_kwargs
        )
        ControlNetBlock.clear_ctl_image()
        return out


def apply_controlnet(model, class_to_wrap, init_std=1e-4):
    for name, module in model.named_children():
        if isinstance(module, class_to_wrap):
            new_module = ControlNetBlock(module, module.out_channels, init_std=init_std)
            setattr(model, name, new_module)
        if len(list(module.children())) > 0:
            apply_controlnet(module, class_to_wrap=class_to_wrap, init_std=init_std)


@persistence.persistent_class
class ControlNetBlock(nn.Module):
    """
    Light, magnitude-preserving implementation of ControlNet
    for small EDM2 models.
    """

    _ctl_embedding = None

    def __init__(self, block, channels, init_std=1e-4):
        super().__init__()
        self.zero_conv = nn.Conv2d(channels, channels, (1, 1), bias=False)
        nn.init.normal_(self.zero_conv.weight, mean=0.0, std=init_std)

        assert isinstance(block, nn.Module)
        self.block = block

    def forward(self, x, label):
        # Check that the control embedding is set
        if ControlNetBlock._ctl_embedding is None:
            raise RuntimeError("ControlNetBlock: _ctl_embedding is None")

        ctl_emb = self.block(ControlNetBlock._ctl_embedding, label)
        block_out = self.block(x, label)
        ctl_out = self.zero_conv(ctl_emb)

        ControlNetBlock.set_ctl_image(
            ctl_emb
        )  # Prevents having to edit the module being wrapped.

        return mp_finetune_sum(block_out, ctl_out)

    @classmethod
    def set_ctl_image(cls, img):
        cls._ctl_embedding = img

    @classmethod
    def get_ctl_image(cls):
        return cls._ctl_embedding

    @classmethod
    def clear_ctl_image(cls):
        cls.set_ctl_image(None)


def mp_finetune_sum(a, b, alpha=1.0):
    """
    Sum of a and b, but rescaled to the magnitude of a.
    Meant to maintain the assumptions for which EDM2 is built
    upon, but in a finetuning context.
    """
    combined = block_out + alpha * ctl_out
    block_magnitude = torch.norm(block_out, p=2, dim=1, keepdim=True)
    combined_magnitude = torch.norm(combined, p=2, dim=1, keepdim=True)
    if combined_magnitude < 1e-8:
        return combined * (block_magnitude / (combined_magnitude + 1e-8))
    return combined * (block_magnitude / (combined_magnitude))
