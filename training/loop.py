import json
import os
import torch

try:
    from IPython import get_ipython

    if get_ipython() is not None:  # Check if running in a Jupyter Notebook
        from tqdm.notebook import tqdm
    else:
        raise ImportError  # Force fallback to standard tqdm
except ImportError:
    from tqdm import tqdm

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from .utils import detach_dict, get_optimizer, get_lr_scheduler, get_loss_fn
from ..utils.pytorch import AcDataLoader


def train_loop(
    model,
    dataset,
    opt,
    guidance_model=None,
    optimizer=None,
    scheduler=None,
    loss_fn=None,
):

    dataloader = AcDataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    optimizer = get_optimizer(model, opt) if optimizer is none else optimizer
    scheduler = get_lr_scheduler(optimizer, opt) if scheduler is None else scheduler
    loss_fn = get_loss_fn(opt) if loss_fn is None else loss_fn

    # Make a clone of the main model's optimizer
    if guidance_model:
        optimizer_class = type(optimizer)
        optimizer_kwargs = optimizer.defaults
        guidance_optimizer = optimizer_class(
            guidance_model.parameters(), **optimizer_kwargs
        )

    # Setup Directories
    run_dir = os.path.join(opt.run_dir, opt.run_name)
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    img_dir = os.path.join(run_dir, "img")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    stats_path = os.path.join(run_dir, "training_stats.jsonl")

    exec_list(opt.setup_actions)

    with open(os.path.join(run_dir, "options.json"), "w") as f:
        json.dump(opt, f, indent=4)

    epoch_bar = tqdm(range(opt.start_epoch, opt.epochs + 1), desc="Epochs", position=0)

    last_batch = None
    best_loss = float("inf")

    for epoch in epoch_bar:
        batch_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            leave=False,
            position=1,
        )

        running_loss = 0.0
        running_guidance_loss = 0.0

        for i, batch in enumerate(batch_bar):
            exec_list(opt.epoch_start_actions)
            last_batch = batch
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            if opt.grad_clip > 0.0:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), clip_value=opt.grad_clip
                )
            optimizer.step()

            running_loss += loss.item()

            if guidance_model and epoch % opt.guidance_train_interval == 0:
                guidance_optimizer.zero_grad()
                guidance_loss = loss_fn(guidance_model, detach_dict(batch)).sum()
                guidance_loss.backward()
                if opt.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        guidance_model.parameters(), max_norm=opt.grad_clip
                    )
                guidance_optimizer.step()
                running_guidance_loss += guidance_loss.item()
            else:
                guidance_loss = None

            batch_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "guidance_loss": (
                        f"{guidance_loss.item():.4f}"
                        if guidance_loss is not None
                        else "—"
                    ),
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                }
            )

        scheduler.step()
        exec_list(opt.epoch_end_actions)

        if epoch % opt.save_interval == 0 or running_loss < best_loss:
            name_suffix = "best" if running_loss < best_loss else "last"
            # Save model
            checkpoint_file = os.path.join(
                checkpoint_dir, f"model_{opt.run_name}_{name_suffix}.pth"
            )
            model.save_checkpoint(
                checkpoint_file,
                epoch,
                optimizer,
                scheduler=scheduler,
                scaler=None,
                guidance_model=guidance_model,
                guidance_optimizer=guidance_optimizer,
                opt=opt,
            )

        if epoch % opt.output_interval == 0 or epoch in opt.sanity_epochs:
            # Generate a batch of images
            batch_size = min(len(last_batch), opt.gen_size)
            for g in opt.guidance_strengths:
                img_epoch_dir = os.path.join(img_dir, f"epoch_{epoch}_guidance_{g}")
                os.makedirs(img_epoch_dir, exist_ok=True)
                generate_images(
                    model,
                    num_images=batch_size,
                    additional_input=last_batch[:batch_size],
                    gnet=guidance_model,
                    outdir=img_epoch_dir,
                    guidance_strength=g,
                )

        avg_loss = running_loss / len(dataloader)
        avg_guidance_loss = (
            running_guidance_loss / len(dataloader)
            if guidance_model and epoch % opt.guidance_train_interval == 0
            else None
        )
        epoch_bar.set_postfix(
            {
                "avg_loss": f"{avg_loss:.4f}",
                "avg_guidance_loss": (
                    f"{avg_guidance_loss:.4f}" if avg_guidance_loss is not None else "—"
                ),
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            }
        )

        # Save stats to JSONL
        with open(stats_path, "a") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "loss": avg_loss,
                    "guidance_loss": avg_guidance_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                f,
            )
            f.write("\n")

    print(f"Training run '{run_name}' done.")


def exec_list(actions):
    for action in actions:
        func = action["func"]
        if action.get("params"):
            func(action["params"])
        else:
            func()
