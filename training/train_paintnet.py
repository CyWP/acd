import sys

from ..utils.pytorch import inspect_model_parameters, get_device
from .loop import TrainLoop
from .train_options import TrainOptions
from ..networks.paintnet import get_paintnet
from ..datasets.patch import PatchNeighborhoodDataset

PAINT_DEFAULTS = {
    "epochs": 1000,
    "run_name": "paintnet-edm2-xs-finetuning",
    "model_size": "xs",
    "guidance_model_size": "xs",
    "autoguidance": False,
    "guidance_size": "xs",
    "cache_dataset": True,
    "dataset_max_size": int(sys.maxsize),
    "hed_path": None,
    "previously_cached": False,
    "split": "train",
    "ctl_dropout": 0.15,
    "nbhd_dropout": 0.25,
    "second_phase_start": 100,
    "second_phase_lr": 0.00001,
    "scheduler": {
        "name": "reducelronplateau",
        "args": {
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
        },
    },
    "loss": [
        {
            "name": "mse",
            "multiplier": 1.0,
            "args": {},
        }
    ],
    "data_dir": None,
}


def train_paintnet(opts):
    train_args = PAINT_DEFAULTS.copy()
    train_args.update(opts)
    train_opts = TrainOptions(train_args)
    train_opts.epoch_end_actions = [second_phase]

    model = get_paintnet(train_opts)
    model = model.to(get_device())
    finetune_label(model)

    # Get guidance model if relevant
    guidance_model = (
        get_paintnet(
            size=train_opts.guidance_model_size,
            folder=train_opts.run_dir,
        )
        if train_opts.autoguidance
        else None
    )
    if train_opts.autoguidance:
        guidance_model = guidance_model.to(get_device())
        guidance_model.finetune()

    # dataset = CtlNbhdDs(
    #     root_dir=train_opts.run_dir,
    #     hed_path=train_opts.hed_path,
    #     should_cache=train_opts.cache_dataset,
    #     cached=train_opts.previously_cached,
    #     split=train_opts.split,
    #     max_size=train_opts.dataset_max_size,
    #     ctl_dropout=train_opts.ctl_dropout,
    #     nbhd_dropout=train_opts.nbhd_dropout,
    # )
    dataset = PatchNeighborhoodDataset(train_opts.data_dir, scale=train_opts.scale)

    train_loop = TrainLoop()
    train_loop.start(
        model=model, dataset=dataset, opt=train_opts, guidance_model=guidance_model
    )


def second_phase(loop: TrainLoop):
    if loop.current_epoch == loop.second_phase_start:
        print("Initiating second training phase.")
        finetune_nbhd(loop.model)
        lr = loop.second_phase_lr
        for g in loop.optim.param_groups:
            g["lr"] = lr
        try:
            loop.sched.patience *= 2
        except:
            pass


def finetune_label(model):
    model.requires_grad_(False)
    model.net.label.requires_grad_(True)


def finetune_nbhd(model):
    model.finetune()
    model.net.label.requires_grad_(False)
