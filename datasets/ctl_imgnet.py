import os
import sys
import random

from datasets import load_dataset
import PIL
import torch
from torch.utils.data import Dataset
from collections.abc import Iterable

from ..networks.hed import get_hed
from ..utils.pytorch import get_device, AcDataSample, AcDataset
from ..utils.data import pil_to_tensor, one_hot, bw_to_rgb
from ..utils import IMG_KEY, LABEL_KEY, CTL_KEY


class CtlNetDS(AcDataset):
    _root_name = "data_controlnet"
    _cache_path = "data"
    _label_size = 1000
    _max_sizes = {"train": 1281167, "test": 100000, "validation": 50000}

    def __init__(
        self,
        root_dir=".",
        hed_path=None,
        should_cache=False,
        cached=False,
        split="train",
        max_size=int(sys.maxsize),
        ctl_dropout=0.1,
        img_size=64,
    ):
        self.root = os.path.join(root_dir, CtlNetDS._root_name)
        self.cache = self.path(CtlNetDS._cache_path)
        self.split = split
        self.should_cache = should_cache
        self.max_size = max_size
        self.ctl_dropout = ctl_dropout
        self.shuffled_indices = {key: [] for key in CtlNetDS._max_sizes}
        self.new_indices()
        os.makedirs(self.root, exist_ok=True)
        if not cached:
            self.imgnet = load_dataset(
                f"benjamin-paine/imagenet-1k-{img_size}x{img_size}", cache_dir=self.root
            )
            self.hed = (
                get_hed(hed_path) if hed_path else get_hed(root_dir)
            )  # HED encoder
            if should_cache:
                for split in CtlNetDS._max_sizes:
                    os.makedirs(os.path.join(self.cache, split), exist_ok=True)

    def path(self, subpath):
        return os.path.join(self.root, subpath)

    def cache_path(self, idx):
        return os.path.join(self.cache, self.split, f"{idx}.pt")

    def check_cached(self, idx):
        return os.path.exists(self.cache_path(idx))

    def hed_to(self, device):
        self.hed = self.hed.to(
            device
        )  # May be useful to keep HED encoder on cpu to make space sometimes

    def release_hed(self):
        self.hed = None  # Releases hed, and especially memory. Use after all data has been cached.

    def train(self):
        self.split = "train"

    def test(self):
        self.split = "test"

    def val(self):
        self.split = "validation"

    def __len__(self):
        if self.split == "train":
            return min(self.max_size, 1281167)
        if self.split == "test":
            return min(self.max_size, 100000)
        if self.split == "validation":
            return min(self.max_size, 50000)

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
        hed_encoding = self.hed(img).squeeze(0).repeat(3, 1, 1).to(get_device())
        sample = AcDataSample({IMG_KEY: img, LABEL_KEY: label, CTL_KEY: hed_encoding})
        if self.should_cache:
            torch.save(
                sample.to(torch.device("cpu")),
                filepath,
            )
        # Random dropout for ctl input
        if random.random() < self.ctl_dropout:
            sample.ctl = torch.zeros_like(sample.ctl) - 1.0
        return sample

    def new_indices(self):
        for key, val in CtlNetDS._max_sizes.items():
            if len(self.shuffled_indices[key]) == 0:
                ilist = list(range(val))
                random.shuffle(ilist)
                self.shuffled_indices[key] = ilist
