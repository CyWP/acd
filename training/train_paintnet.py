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
        },
        # {
        #     "name": "label_sparsity",
        #     "multiplier": 0.01,
        #     "args": {},
        # },
    ],
    "data_dir": None,
}


def train_paintnet(opts):
    train_args = PAINT_DEFAULTS.copy()
    train_args.update(opts)
    train_opts = TrainOptions(train_args)
    # train_opts.epoch_end_actions = [second_phase, additional_stats]

    model = get_paintnet(train_opts)
    model = model.to(get_device())
    finetune_nbhd(model)

    inspect_model_parameters(model)

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


def additional_stats(loop: TrainLoop):
    if loop.current_epoch <= loop.second_phase_start:
        lab = loop.model.net.label
        print(f"LABEL--max:{lab.max()} min:{lab.min()} L1:{lab.abs().sum()}")


def finetune_label(model):
    model.requires_grad_(False)
    model.net.label.requires_grad_(True)


def finetune_nbhd(model):
    model.finetune()
    model.net.label.requires_grad_(False)
