import torch.nn as nn
from transformers import ViTModel


def get_dino():
    if DinoViT._instance is None:
        vit = DinoViT()
        vit.eval()
        vit.requires_grad_(False)
        DinoViT._instance = vit
    return DinoViT._instance


class DinoViT(nn.Module):

    _instance = None

    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained("facebook/dino-vits8")

    def forward(self, x):
        return self.model(x).last_hidden_state
