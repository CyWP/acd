import numpy as np
import torch
import torch.nn as nn

from ..utils import IMG_KEY, LABEL_KEY, CTL_KEY
from ..utils.data import (
    extract_patches,
    assemble_patches,
    flip_vertical,
    flip_horizontal,
    one_hot,
)
from ..utils.pytorch import apply_to_modules, TanhRelu
from .edm2 import mp_sum, mp_silu, mp_cat
from .edm2_wrapper import (
    EDM2Wrapper,
    PrecondWrapper,
    UNetWrapper,
    BlockWrapper,
    mp_finetune_sum,
)

CTL_URLS = {"xs": "path/to/pretrained"}


def get_paintnet(opts, guidance=False, pretrained=False):
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

    def __init__(self, precond):
        super().__init__(precond)
        self.label = nn.Parameter(one_hot(42, 1000))
        self.label_proc = nn.Tanh()
        self.nbhd_encoder = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=1),
            nn.Conv2d(in_channels=12, out_channels=6, kernel_size=1),
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x, sigma, force_fp32=False, return_logvar=True, **unet_kwargs):
        ctl = x.ctl
        nbhd = x.nbhd
        noise = x.noise

        nbhd = self.nbhd_encoder(self.format_neighbourhood(nbhd))
        # labels = torch.square(self.label_proc(self.label)).repeat(ctl.shape[0], 1)
        labels = self.label.repeat(ctl.shape[0], 1)
        x.label = labels
        PaintBlock.set_ctl_image(torch.cat([nbhd, ctl], dim=1))

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

    def format_neighbourhood(self, nbhd):
        patches = extract_patches(nbhd, 64, 64, as_grid=False)
        nbhd_list = []
        for i in list(range(4)) + list(range(5, 9)):
            row, col = i // 3, i % 3
            flip_v, flip_h = row % 2 == 0, col % 2 == 0
            img = patches[:, i, :, :, :]
            if flip_v:
                img = flip_vertical(img)
            if flip_h:
                img = flip_horizontal(img)
            nbhd_list.append(img)
        return torch.cat(nbhd_list, dim=1)


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

    def __init__(self, block, num_convs=1):
        super().__init__(block=block)

        zc_list = []
        C = block.out_channels
        for _ in range(num_convs):
            zc_list.append(
                nn.Conv2d(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=3,
                    padding=1,
                    groups=C,
                    bias=False,
                )
            )
            zc_list.append(nn.ReLU(inplace=True))
            zc_list.append(
                nn.Conv2d(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=1,
                    padding=0,
                    groups=1,
                    bias=False,
                )
            )

        for layer in zc_list:
            if isinstance(layer, nn.Conv2d):
                nn.init.zeros_(layer.weight)
        self.zero_conv = nn.Sequential(*zc_list)

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
        self.w_net.requires_grad_(False)

    @classmethod
    def set_ctl_image(cls, img):
        cls._ctl_embedding = img

    @classmethod
    def get_ctl_image(cls):
        return cls._ctl_embedding

    @classmethod
    def clear_ctl_image(cls):
        cls.set_ctl_image(None)
