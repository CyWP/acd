import os
import sys

from datasets import load_dataset
import PIL
import torch
from torch.utils.data import Dataset
from collections.abc import Iterable

from ..networks.hed import get_hed
from ..utils.pytorch import get_device, AcDataSample
from ..utils.data import pil_to_tensor, one_hot, bw_to_rgb
from ..utils import IMG_KEY, LABEL_KEY, CTL_KEY


class CtlNetDS(Dataset):
    _root_name = "data_controlnet"
    _cache_path = "data"
    _label_size = 1000
    _possible_splits = ["train", "test", "validation"]

    def __init__(
        self,
        root_dir=".",
        hed_path=None,
        should_cache=True,
        cached=False,
        split="train",
        max_size=int(sys.maxsize),
    ):
        self.root = os.path.join(root_dir, CtlNetDS._root_name)
        self.cache = self.path(CtlNetDS._cache_path)
        self.split = split
        self.should_cache = should_cache
        self.max_size = max_size
        os.makedirs(self.root, exist_ok=True)
        if not cached:
            self.imgnet = load_dataset("benjamin-paine/imagenet-1k-64x64")
            self.hed = (
                get_hed(hed_path) if hed_path else get_hed(root_dir)
            )  # HED encoder
            if should_cache:
                for split in self.__class__._possible_splits:
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

    def __getitem__(self, idx):
        if isinstance(idx, Iterable):
            return AcDataSample.stack([self.get_sample(i) for i in idx])
        return self.get_sample(idx)

    def get_sample(self, idx):
        assert isinstance(idx, int)
        filepath = self.cache_path(idx)
        if self.check_cached(idx):
            data = torch.load(filepath)
            return AcDataSample(data)
        data = self.imgnet[self.split].select([idx])
        img = pil_to_tensor(data["image"][0])  # .unsqueeze(0)
        label = one_hot(int(data["label"][0]), CtlNetDS._label_size)
        hed_encoding = self.hed(img).squeeze(0).repeat(3, 1, 1)
        if self.should_cache:
            torch.save(
                {
                    IMG_KEY: img,
                    LABEL_KEY: label,
                    CTL_KEY: hed_encoding,
                },
                filepath,
            )
        return AcDataSample({IMG_KEY: img, LABEL_KEY: label, CTL_KEY: hed_encoding})
