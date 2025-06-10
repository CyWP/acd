import torch
import torch.nn as nn

from ..utils import IMG_KEY
from ..utils.easydict import EasyDict
from ..networks.hed import get_hed
from ..networks.dino import get_dino


def build_loss(train_opts):
    inference_kwargs = train_opts.get("inference_kwargs")
    if not inference_kwargs:
        influence_kwargs = {}
    return LossBuilder(train_opts.loss, **inference_kwargs)


def get_loss(loss):
    name = loss["name"]
    multiplier = loss["multiplier"] if loss.get("multiplier") else 1.0
    loss_args = loss["args"] if loss.get("args") else {}

    if name == "mse":
        return EasyDict(loss=MSELoss(), multiplier=multiplier)
    elif name == "hed":
        return EasyDict(loss=HEDLoss(**loss_args), multiplier=multiplier)
    elif name == "perception":
        return EasyDict(loss=PerceptionLoss(), multiplier=multiplier)
    elif name == "label_sparsity":
        return EasyDict(loss=LabelL1Loss(), multiplier=multiplier)


class LossBuilder(nn.Module):
    """
    loss function that also automatically calls backward()
    """

    def __init__(self, losses, **inference_kwargs):
        super().__init__()
        self.inference = Inference(**inference_kwargs)
        self.loss_fns = [get_loss(l) for l in losses]

    def __call__(self, net, x):
        denoised = self.inference(net, x)
        loss = 0.0
        for fn in self.loss_fns:
            l = fn.multiplier * fn.loss(denoised, x)
            loss += l
        return loss


class Inference(nn.Module):
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, x):
        images = x.img
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        x.noise = torch.randn_like(images) * sigma
        return net(x, sigma, return_logvar=False, force_fp32=True)


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, denoised, x):
        return ((x.img - denoised) ** 2).mean()


class HEDLoss(nn.Module):
    def __init__(self, folder=".", force_download=False):
        super().__init__()
        self.hed = get_hed(folder=folder, force_download=force_download)

    def __call__(self, denoised, x):
        return ((x.ctl - self.hed(denoised)) ** 2).mean()


class PerceptionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = get_dino()

    def __call__(self, denoised, x):
        batch = torch.cat([x.img, denoised], dim=0)
        batch = self.dino(batch)
        return ((batch[0] - batch[1]) ** 2).mean()


class LabelL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, denoised, x):
        return x.label.abs().sum()
