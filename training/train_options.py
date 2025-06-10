from ..utils.easydict import EasyDict


class TrainOptions(EasyDict):

    _defaults = {
        "epochs": 1000,
        "run_name": "train",
        "run_dir": ".",
        "seed": 42,
        "batch_size": 16,
        "optimizer": {
            "name": "adam",
            "args": {},
        },
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
        "inference_kwargs": {"P_mean": -0.8, "P_std": 1.6},
        "lr": 1e-3,
        "grad_clip": -1.0,
        "guidance_train_interval": 1,
        "save_interval": 1,
        "output_interval": 1,
        "guidance_strengths": [1.0],
        "gen_size": 4,
        "sanity_epochs": [],
        "setup_actions": [],
        "epoch_start_actions": [],
        "epoch_end_actions": [],
        "model_args": {},
    }

    def __init__(self, opts=None):
        self.update(TrainOptions._defaults)
        if opts:
            self.update(opts)
