import torch

from .loss import DiffusionMSELoss


def get_optimizer(model, opts):
    optimizer_name = opts.optimizer["name"].lower()
    kwargs = opts.optimizer["args"]
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), **kwargs)
    elif optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), **kwargs)
    elif optimizer_name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), **kwargs)
    elif optimizer_name == "adamax":
        return torch.optim.Adamax(model.parameters(), **kwargs)
    elif optimizer_name == "nadam":
        return torch.optim.NAdam(model.parameters(), **kwargs)
    elif optimizer_name == "lbfgs":
        return torch.optim.LBFGS(model.parameters(), **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_lr_scheduler(optimizer, opts):
    scheduler_name = opts.scheduler["name"].lower()
    kwargs = opts.scheduler["args"]
    kwargs["lr"] = opts.lr
    if scheduler_name == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == "exponentiallr":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_name == "multisteplr":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler_name == "cosineannealingwarmrestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def get_loss_fn(opts):
    name = opts.loss_fn["name"].lower()
    kwargs = opts.loss_fn["args"]
    if name == "mse":
        return DiffusionMSELoss(**kwargs)
    raise ValueError(f"Unsupported loss function: {name}")


def detach_dict(d):
    dic = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            dic[k] = v.detach()
        else:
            dic[k] = v
    return dic
