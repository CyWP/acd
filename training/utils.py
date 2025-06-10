import torch


def get_optimizer(model, opts):
    optimizer_name = opts.optimizer["name"].lower()
    kwargs = opts.optimizer["args"]
    kwargs["lr"] = opts.lr
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
    return Scheduler(scheduler_name, optimizer, lr=opts.lr, **kwargs)


class Scheduler:

    def __init__(self, name, optimizer, lr, **kwargs):
        self.name = name
        if name == "reducelronplateau":
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
        elif name == "steplr":
            self.sched = torch.optim.lr_scheduler.StepLR(optimizer, lr=lr, **kwargs)
        elif name == "exponentiallr":
            self.sched = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, lr=lr, **kwargs
            )
        elif name == "cosineannealinglr":
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, lr=lr, **kwargs
            )
        elif name == "multisteplr":
            self.sched = torch.optim.lr_scheduler.MultisStepLR(
                optimizer, lr=lr, **kwargs
            )
        elif name == "cosineannealingwarmrestarts":
            self.sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, lr=lr, **kwargs
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def state_dict(self):
        return self.sched.state_dict()

    def load_state_dict(self, state_dict):
        self.sched.load_state_dict(state_dict)

    def step(self, train_loop):
        if self.name == "reducelronplateau":
            self.sched.step(train_loop.running_loss)
        else:
            self.sched.step()


def detach_dict(d):
    dic = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            dic[k] = v.detach()
        else:
            dic[k] = v
    return dic
