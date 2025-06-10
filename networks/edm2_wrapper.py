"""
Tehse are base classes meant for wrapping the EDM2 architecture for editing/finetuning.
These classes do nothing.
"""

import os

import torch
import torch.nn as nn

from .edm2 import Precond, UNet, Block, get_edm2


class Finetunable:
    def finetune(self):
        raise NotImplementedError("This is an interface. Override this function.")


class BlockWrapper(nn.Module, Finetunable):

    def __init__(self, block: Block):
        super().__init__()
        self.w_net = block

    def forward(self, x, emb):
        return self.w_net(x, emb)

    def finetune(self):
        pass


class UNetWrapper(nn.Module, Finetunable):

    def __init__(self, unet: UNet):
        super().__init__()
        self.w_net = unet

    def forward(self, x, noise_labels, class_labels):
        return self.w_net(x, noise_labels, class_labels)

    def finetune(self):
        pass


class PrecondWrapper(nn.Module, Finetunable):

    def __init__(self, precond: Precond):
        super().__init__()
        self.w_net = precond

    def forward(
        self,
        x,
        sigma,
        class_labels=None,
        force_fp32=False,
        return_logvar=False,
        **unet_kwargs,
    ):
        return self.w_net(
            x,
            sigma,
            class_labels=class_labels,
            force_fp32=force_fp32,
            return_logvar=return_logvar,
            **unet_kwargs,
        )

    def finetune(self):
        pass


class EDM2Wrapper(nn.Module, Finetunable):

    def __init__(
        self,
        size="xs",
        precond=PrecondWrapper,
        unet=UNetWrapper,
        block=BlockWrapper,
        precond_kwargs={},
        unet_kwargs={},
        block_kwargs={},
        **edm2_kwargs,
    ):
        super().__init__()
        self.net = get_edm2(size=size, **edm2_kwargs)
        wrap_edm2(self, precond=precond, unet=unet, block=block)

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def finetune(self):
        self.requires_grad_(False)
        children = list(self.named_children())
        while len(children) > 0:
            name, module = children.pop()
            if isinstance(module, Finetunable):
                module.finetune()
            children += list(module.named_children())

    def save_checkpoint(
        self,
        path,
        epoch,
        optimizer,
        scheduler=None,
        scaler=None,
        guidance_model=None,
        guidance_optimizer=None,
        opt=None,
    ):

        # Base checkpoint
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "opt": dict(opt) if opt else {},
        }

        # Include scheduler state if it exists
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Include scaler state for mixed precision if it exists
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        # Deal with guidance models
        if guidance_model:
            if not guidance_optimizer:
                raise ValueError(
                    "Guidance optimizer must be provided with guidance model."
                )
        checkpoint["guidance_model_state_dict"] = (
            guidance_model.state_dict() if guidance_model else None
        )
        checkpoint["guidance_optimizer_state_dict"] = (
            guidance_optimizer.state_dict() if guidance_model else None
        )

        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save checkpoint
        torch.save(checkpoint, path)


def wrap_edm2(
    net,
    precond=PrecondWrapper,
    unet=UNetWrapper,
    block=BlockWrapper,
    precond_kwargs={},
    unet_kwargs={},
    block_kwargs={},
):
    to_adapt = [(None, net, "")]

    while len(to_adapt):
        parent, module, name = to_adapt.pop()
        og_children = list(module.named_children())

        if isinstance(module, Block):
            setattr(parent, name, block(module, **block_kwargs))
        elif isinstance(module, UNet):
            setattr(parent, name, unet(module, **unet_kwargs))
        elif isinstance(module, Precond):
            setattr(parent, name, precond(module, **precond_kwargs))

        if len(og_children) > 0:
            to_adapt += [
                (module, child, child_name) for child_name, child in og_children
            ]


def mp_finetune_sum(a, b, alpha=1.0):
    """
    Sum of a and b, but rescaled to the magnitude of a.
    Meant to maintain the assumptions for which EDM2 is built
    upon, but in a finetuning context.
    """
    combined = a + alpha * b
    a_magnitude = torch.norm(a, p=2, dim=1, keepdim=True)
    combined_magnitude = torch.norm(combined, p=2, dim=1, keepdim=True)
    return combined * (a_magnitude / (torch.clamp(combined_magnitude, min=1e-8)))
