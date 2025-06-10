import random
import sys
import torch
import numpy as np

from .ctl_imgnet import CtlNetDS
from ..networks.deepfill import get_deepfill
from ..utils.pytorch import get_device, AcDataSample
from ..utils import IMG_KEY, LABEL_KEY, CTL_KEY, NBHD_KEY
from ..utils.data import (
    extract_center_img,
    resize_tensor_img,
    pil_to_tensor,
    one_hot,
    bw_to_rgb,
    flip_horizontal,
    flip_vertical,
)


class CtlNbhdDs(CtlNetDS):

    def __init__(
        self,
        root_dir=".",
        hed_path=None,
        should_cache=False,
        cached=False,
        split="train",
        max_size=int(sys.maxsize),
        ctl_dropout=0.15,
        nbhd_dropout=0.25,
        img_size=64,
    ):
        super().__init__(
            root_dir,
            hed_path,
            should_cache,
            cached,
            split,
            max_size,
            ctl_dropout,
            img_size,
        )
        self.nbhd_dropout = nbhd_dropout
        self.inpainter = get_deepfill(root_dir).to(get_device())
        self.inpainter.requires_grad_(False)
        self.img_size = img_size

        # init nbhd blend mask
        nbm = torch.ones((self.img_size * 3,))
        nbm[: self.img_size // 2 + 1] = 0
        nbm[-self.img_size // 2 :] = 0
        nbm[self.img_size // 2 : self.img_size] = np.cos(
            np.linspace(0, 1, self.img_size // 2)
        )
        nbm[2 * self.img_size : -self.img_size // 2 + 1] = nbm[
            self.img_size // 2 : self.img_size
        ][::-1]
        self.nbh_blend_mask = nbm

    def get_sample(self, i):
        self.new_indices()
        idx = self.shuffled_indices[self.split].pop()
        filepath = self.cache_path(idx)
        if self.check_cached(idx):
            data = torch.load(filepath)
            return AcDataSample(data).to(get_device())
        data = self.imgnet[self.split].select([idx])
        img = pil_to_tensor(data["image"][0]).to(get_device())
        label = one_hot(int(data["label"][0]), CtlNetDS._label_size).to(get_device())

        nbhd = self.generate_neighborhood(self)

        ctl = (
            self.hed(img).squeeze(0).to(get_device())
            if random.random() < self.ctl_dropout
            else torch.zeros(1, *img.shape[1:]).to(get_device())
        )

        sample = AcDataSample(
            {
                IMG_KEY: img,
                LABEL_KEY: label,
                CTL_KEY: ctl,
                NBHD_KEY: nbhd,
            }
        )

        if self.should_cache:
            torch.save(
                sample.to(torch.device("cpu")),
                filepath,
            )

        return sample.to(get_device())

        def generate_neighborhood(self, img):
            positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
            # Create and fill neighborhood
            nbhd = torch.empty(3, self.img_size * 3, self.img_size * 3)
            for row, col in positions:
                fill_region = img
                even_row, even_col = row % 2 == 0, col % 2 == 0
                if even_row and even_col:
                    fill_region = flip_horizontal(flip_vertical(fill_region))
                elif even_row:
                    fill_region = flip_horizontal(fill_region)
                elif even_col:
                    fill_region = flip_vertical(fill_region)
                nbhd[
                    :,
                    row * self.img_size : (1 + row) * self.img_size,
                    col * self.img_size : (1 + col) * self.img_size,
                ] = fill_region

            # Randomly blanks outpatches in the 3x3 neighborhood
            mask = torch.ones(1, self.img_size, self.img_size)
            k = random.randint(0, 8)
            filled = random.sample(positions, k)
            # blanked.append((1, 1))  # Always blank out center
            for row, col in filled:
                ys = row * self.img_size
                xs = col * self.img_size
                nbhd[:, ys : ys + self.img_size, xs : xs + self.img_size] = 0

            nbhd[
                :, self.img_size : 2 * self.img_size, self.img_size : 2 * self.img_size
            ] = img
