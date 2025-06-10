import random

from .ctl_imgnet import CtlNetDS
from ..networks.deepfill import get_deepfill
from ..utils.pytorch import get_device, AcDataSample
from ..utils import IMG_KEY, LABEL_KEY, CTL_KEY, NBHD_KEY
from ..utils.data import extract_center_img, resize_tensor_img


class CtlNbhdDs(CtlNetDS):

    def __init__(
        self,
        root_dir=".",
        hed_path=None,
        should_cache=False,
        cached=False,
        split="train",
        max_size=int(sys.maxsize),
        ctl_dropout=0.1,
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
            img_size * 2,
        )
        self.nbhd_dropout = nbhd_dropout
        self.inpainter = get_deepfill(root_dir)
        self.patch_size = patch_size

    def get_sample(self, i):
        x = super().get_sample(i)

        img = x.img1
        label = x.label
        ctl = x.ctl

        if random.random() < self.nbhd_dropout:
            
        else:
            nbhd = torch.ones(ctl.shape) * -1
        return AcDataSample(
            {IMG_KEY: img, LABEL_KEY: label, CTL_KEY: ctl, NBHD_KEY: nbhd}
        )
