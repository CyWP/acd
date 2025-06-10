import timm
import torch
import torch.nn as nn

from ..utils import IMG_KEY


class DiffusionMSELoss(nn.Module):
    def __init__(self, **inference_kwargs):
        super().__init__()
        self.inference = Inference(**inference_kwargs)
        self.mse = MSELoss()

    def __call__(self, net, x):
        return self.mse(x[IMG_KEY], self.inference(net, x))


class FeatRegLoss(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.model = feature_extractor

    def forward(self, pred_img, target_img):
        feat_pred = self.model(pred_img)
        feat_target = self.model(target_img)
        return ((feat_pred - feat_target) ** 2).mean()


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, target, pred):
        return ((target - pred) ** 2).mean()


class Inference(nn.Module):
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def _call__(self, net, x):
        images = x.img
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        noise = torch.randn_like(images) * sigma
        noised_x = x.copy()
        noised_x.img = images + noise
        return net(noised_x, sigma, return_logvar=False, force_fp32=True)
