import sys

from .loop import train_loop
from .loss import DiffusionMSELoss
from .train_options import TrainOptions
from ..networks.controlnet import ControlNetEDM2
from ..networks.edm2 import get_edm2
from ..datasets.imagenet_64 import CtlNetDS

CTL_DEFAULTS = {
    "epochs": 1000,
    "run_name": "ctlnet-edm2-xs-finetuning",
    "model_size": "xs",
    "autoguidance": False,
    "guidance_size": "xs",
    "cache_dataset": True,
    # "dataset_max_size": int(sys.maxsize),
    "dataset_max_size": 128,
    "hed_path": None,
    "previously_cached": False,
    "split": "train",
}


def train_controlnet(opts):
    train_args = CTL_DEFAULTS.copy()
    train_args.update(opts)
    train_opts = TrainOptions(train_args)

    model = ControlNetEDM2(
        edm2_model=get_edm2(size=train_opts.model_size, folder=train_opts.run_dir)
    )

    guidance_model = (
        ControlNetEDM2(
            get_edm2(size=train_opts.guidance_size, folder=train_opts.run_dir)
        )
        if train_opts.autoguidance
        else None
    )

    dataset = CtlNetDS(
        root_dir=train_opts.run_dir,
        hed_path=train_opts.hed_path,
        should_cache=train_opts.cache_dataset,
        cached=train_opts.previously_cached,
        split=train_opts.split,
        max_size=train_opts.dataset_max_size,
    )

    train_loop(
        model=model, dataset=dataset, opt=train_opts, guidance_model=guidance_model
    )
