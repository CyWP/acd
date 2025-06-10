import numpy as np
import torch
import torch.nn as nn

from ..utils import IMG_KEY, LABEL_KEY, CTL_KEY
from .edm2 import mp_sum, mp_silu, mp_cat
from .edm2_wrapper import (
    EDM2Wrapper,
    PrecondWrapper,
    UNetWrapper,
    BlockWrapper,
    mp_finetune_sum,
)

CTL_URLS = {"xs": "path/to/pretrained"}


def get_controlnet(opts, guidance=False, pretrained=False):
    model_args = opts.model_args
    p_args = model_args["precond"] if "precond" in model_args else {}
    u_args = model_args["unet"] if "unet" in model_args else {}
    b_args = model_args["block"] if "block" in model_args else {}
    return EDM2Wrapper(
        size=opts.guidance_model_size if guidance else opts.model_size,
        folder=opts.run_dir,
        pretrained=not pretrained,
        force_download=False,
        precond=PaintPrecond,
        unet=PaintUNet,
        block=PaintBlock,
        precond_kwargs=p_args,
        unet_kwargs=u_args,
        block_kwargs=b_args,
    )


class PaintPrecond(PrecondWrapper):

    def forward(self, x, sigma, force_fp32=False, return_logvar=True, **unet_kwargs):
        ctl = x.ctl
        noise = x.img
        labels = x.label
        PaintBlock.set_ctl_image(ctl)
        out = self.w_net(
            noise,
            sigma,
            class_labels=labels,
            force_fp32=force_fp32,
            return_logvar=return_logvar,
            **unet_kwargs,
        )
        PaintBlock.clear_ctl_image()
        return out


class PaintUNet(UNetWrapper):

    def forward(self, x, noise_labels, class_labels):
        w = self.w_net
        # Embedding.
        emb = w.emb_noise(w.emb_fourier(noise_labels))
        if w.emb_label is not None:
            emb = mp_sum(
                emb,
                w.emb_label(class_labels * np.sqrt(class_labels.shape[1])),
                t=w.label_balance,
            )
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        ctl = PaintBlock.get_ctl_image()
        PaintBlock.set_ctl_image(ctl)
        skips = []
        ctl_skips = []
        for name, block in w.enc.items():
            if "conv" in name:
                x = block(x)
                PaintBlock.set_ctl_image(block(PaintBlock.get_ctl_image()))
            else:
                x = block(x, emb)
            skips.append(x)
            ctl_skips.append(PaintBlock.get_ctl_image())

        # Decoder.
        for name, block in w.dec.items():
            if "block" in name:
                x = mp_cat(x, skips.pop(), t=w.concat_balance)
                PaintBlock.set_ctl_image(
                    torch.cat([PaintBlock.get_ctl_image(), ctl_skips.pop()], dim=1)
                )
            x = block(x, emb)
        x = w.out_conv(x, gain=w.out_gain)
        return x


class PaintBlock(BlockWrapper):
    """
    Light, magnitude-preserving implementation of ControlNet
    for small EDM2 models.
    """

    _ctl_embedding = None

    def __init__(self, block, init_std=1e-4):
        super().__init__(block=block)
        self.zero_conv = nn.Conv2d(
            block.out_channels, block.out_channels, (1, 1), bias=False
        )
        nn.init.normal_(self.zero_conv.weight, mean=0.0, std=init_std)

    def forward(self, x, label):
        ctl = PaintBlock.get_ctl_image()
        ctl_emb = self.w_net(ctl, label)
        block_out = self.w_net(x, label)
        ctl_out = self.zero_conv(ctl_emb)
        PaintBlock.set_ctl_image(ctl_out)
        res = mp_finetune_sum(block_out, ctl_out)
        return res

    def finetune(self):
        self.zero_conv.requires_grad_(True)

    @classmethod
    def set_ctl_image(cls, img):
        cls._ctl_embedding = img

    @classmethod
    def get_ctl_image(cls):
        return cls._ctl_embedding

    @classmethod
    def clear_ctl_image(cls):
        cls.set_ctl_image(None)
