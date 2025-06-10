import sys

from ..utils import inspect_model_parameters
from .loop import train_loop
from .loss import DiffusionMSELoss
from .train_options import TrainOptions
from ..networks.controlnet import get_controlnet
from ..networks.edm2 import get_edm2
from ..datasets.imagenet_64 import CtlNetDS

CTL_DEFAULTS = {
    "epochs": 1000,
    "run_name": "ctlnet-edm2-xs-finetuning",
    "model_size": "xs",
    "guidance_model_size": "xs",
    "autoguidance": False,
    "guidance_size": "xs",
    "cache_dataset": True,
    # "dataset_max_size": int(sys.maxsize),
    "dataset_max_size": 16,
    "hed_path": None,
    "previously_cached": False,
    "split": "train",
    "model_args": {"block": {"init_std": 0.0}},
}


def train_controlnet(opts):
    train_args = CTL_DEFAULTS.copy()
    train_args.update(opts)
    train_opts = TrainOptions(train_args)

    model = get_controlnet(
        size=train_opts.model_size,
        folder=train_opts.run_dir,
        model_args=train_opts.model_args,
    )
    model.finetune()
    inspect_model_parameters(model)
    breakpoint()

    # Get guidance model if relevant
    guidance_model = (
        get_controlnet(
            size=train_opts.guidance_model_size,
            folder=train_opts.run_dir,
            model_args=train_opts.model_args,
        )
        if train_opts.autoguidance
        else None
    )
    if train_opts.autoguidance:
        guidance_model.finetune()

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
