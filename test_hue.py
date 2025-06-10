import torch
import torch.nn as nn
from .utils.data import tensors_to_pil

IMG_PATH = "/home/cyvvp/Train/acd/data_controlnet/data/train/349542.pt"
INVS2 = 1/(2**0.5)
RWGT = 0.3086
GWGT = 0.6094
BWGT = 0.0820

def test_hue_rotation():
    img = torch.load(IMG_PATH)["img"]
    rots = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    for r in rots:
        rt = hue_rot(tensor, r)
        ri = tensors_to_pil(rt)
        ri.save(f"rot_{r}.png")

def hue_rot(img, deg):
