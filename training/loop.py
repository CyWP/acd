import json
import os
from random import randint

import torch
from tqdm import tqdm

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from ..utils.easydict import EasyDict
from .utils import detach_dict, get_optimizer, get_lr_scheduler
from .loss import build_loss
from ..utils.pytorch import AcDataLoader, AcDataSample
from ..generate import generate_images


class TrainLoop(EasyDict):

    def loop(self):
        self.exec_list(self.setup_actions)
        if not self.get("current_epoch"):
            self.current_epoch = 1
        self.start_epoch = self.current_epoch
        self.best_loss = float("inf")
        self.avg_guidance_loss = None

        self.epoch_bar = tqdm(
            range(self.start_epoch, self.epochs + 1), desc="Epochs", position=0
        )

        for epoch in self.epoch_bar:
            self.current_epoch = epoch
            batch_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}",
                leave=False,
                position=1,
            )

            self.running_loss = 0.0
            self.running_guidance_loss = 0.0

            for i, batch in enumerate(batch_bar):
                self.exec_list(self.epoch_start_actions)

                batch_loss = self.optimization_step(batch, guidance=False)
                self.running_loss += batch_loss

                if self.guidance_model and epoch % self.guidance_train_interval == 0:
                    batch_guidance_loss = self.optimization_step(batch, guidance=True)
                    self.running_guidance_loss += batch_guidance_loss
                else:
                    batch_guidance_loss = None

                batch_bar.set_postfix(
                    {
                        "loss": f"{batch_loss:.4f}",
                        "guidance_loss": (
                            f"{batch_guidance_loss:.4f}"
                            if batch_guidance_loss is not None
                            else "—"
                        ),
                        "lr": f"{self.optim.param_groups[0]['lr']:.6f}",
                    }
                )

            self.sched.step(self)

            self.exec_list(self.epoch_end_actions)

            self.avg_loss = self.running_loss / len(self.dataloader)
            if self.guidance_model:
                self.avg_guidance_loss = self.running_guidance_loss / len(
                    self.dataloader
                )

            if epoch % self.save_interval == 0 or self.avg_loss < self.best_loss:
                self.epoch_bar.set_postfix_str(f"Saving checkpoint...")
                self.save(suffix="best" if self.avg_loss < self.best_loss else "last")
                self.epoch_bar.set_postfix_str("")

            if epoch % self.output_interval == 0 or epoch in self.sanity_epochs:
                self.epoch_bar.set_postfix_str(f"Generating images...")
                gen_batch = AcDataSample.stack(
                    [
                        self.dataset[randint(0, len(self.dataset) - 1)]
                        for _ in range(self.gen_size)
                    ]
                )
                for g in self.guidance_strengths:
                    img_epoch_dir = os.path.join(
                        self.img_dir, f"epoch_{epoch}_guidance_{g}"
                    )
                    os.makedirs(img_epoch_dir, exist_ok=True)
                    generate_images(
                        self.model,
                        gen_batch[: self.gen_size],
                        num_images=self.gen_size,
                        gnet=self.guidance_model,
                        outdir=img_epoch_dir,
                        guidance_strength=g,
                    )
                self.epoch_bar.set_postfix_str("")

            self.best_loss = min(self.best_loss, self.avg_loss)

            self.epoch_bar.set_postfix(
                {
                    "best_loss": f"{self.best_loss:.4f}",
                    "last_loss": f"{self.avg_loss:.4f}",
                    "avg_guidance_loss": (
                        f"{self.avg_guidance_loss:.4f}"
                        if self.avg_guidance_loss is not None
                        else "—"
                    ),
                    "lr": f"{self.optim.param_groups[0]['lr']:.6f}",
                }
            )

            # Save stats to JSONL
            with open(self.stats_path, "a") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "loss": self.avg_loss,
                        "guidance_loss": self.avg_guidance_loss,
                        "lr": self.optim.param_groups[0]["lr"],
                    },
                    f,
                )
                f.write("\n")

        print(f"Training run '{self.run_name}' done.")

    def exec_list(self, actions):
        for action in actions:
            action(self)

    def save(self, suffix="best"):
        save_dir = self.best_dir if suffix == "best" else self.last_dir
        model_path = os.path.join(save_dir, f"model.pth")
        torch.save(self.model.state_dict(), model_path)
        if self.guidance_model:
            guidance_path = os.path.join(save_dir, f"guidance.pth")
            torch.save(self.guidance_model.state_dict(), guidance_path)
        loop_dict = {}
        for key, val in self.items():
            if key in ["model", "guidance_model", "epoch_bar"]:
                continue
            try:
                loop_dict[key] = val.state_dict()
            except:
                loop_dict[key] = val
        loop_path = os.path.join(save_dir, "loop.pth")
        torch.save(loop_dict, loop_path)

    def initialize(self):
        self.dataloader = AcDataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        self.optim = get_optimizer(self.model, self)
        self.sched = get_lr_scheduler(self.optim, self)
        self.loss_fn = build_loss(self)

        # Make a clone of the main model's optimizer
        if self.guidance_model:
            optimizer_class = type(self.optim)
            optimizer_kwargs = self.optim.defaults
            self.guidance_optim = optimizer_class(
                self.guidance_model.parameters(), **optimizer_kwargs
            )
        else:
            self.guidance_optim = None

    def load(self):
        print(f"\n---\nLoading previous run from {self.run_dir}.\n---")
        self.model.load_state_dict(
            torch.load(os.path.join(self.last_dir, "model.pth"), weights_only=False)
        )
        guidance_path = os.path.join(self.last_dir, "guidance.pth")
        self.guidance_model = (
            torch.load(guidance_path, weights_only=False)
            if os.path.exists(guidance_path)
            else None
        )
        self.update(
            torch.load(os.path.join(self.last_dir, "loop.pth"), weights_only=False)
        )
        self.dataloader = AcDataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        optim = get_optimizer(self.model, self)
        optim.load_state_dict(self.optim)
        self.optim = optim
        sched = get_lr_scheduler(self.optim, self)
        sched.load_state_dict(self.sched)
        self.sched = sched
        loss_fn = build_loss(self)
        loss_fn.load_state_dict(self.loss_fn)
        self.loss_fn = loss_fn

        self.start_epoch = self.current_epoch

        # Make a clone of the main model's optimizer
        if self.guidance_model:
            optimizer_class = type(self.optim)
            optimizer_kwargs = self.optimizer.defaults
            self.guidance_optim = optimizer_class(
                self.guidance_model.parameters(), **optimizer_kwargs
            )
        else:
            self.guidance_optim = None

    def start(self, model, dataset, opt, guidance_model=None):
        self.update(opt)
        self.setup_dirs()
        load = os.path.exists(os.path.join(self.last_dir, "model.pth"))
        self.make_dirs()
        self.model = model
        self.dataset = dataset
        self.guidance_model = guidance_model
        if load:
            self.load()
        else:
            self.initialize()
        self.loop()

    def optimization_step(self, batch, guidance=False):
        if guidance:
            mod, opt = self.guidance_model, self.guidance_optim
        else:
            mod, opt = self.model, self.optim
        opt.zero_grad()
        loss = self.loss_fn(self.model, batch)
        loss.backward()
        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_value_(mod.parameters(), clip_value=self.grad_clip)
        opt.step()
        return loss.item()

    def setup_dirs(self):
        # Setup Directories
        self.run_dir = os.path.join(self.run_dir, self.run_name)
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.best_dir = os.path.join(self.checkpoint_dir, "best")
        self.last_dir = os.path.join(self.checkpoint_dir, "last")
        self.img_dir = os.path.join(self.run_dir, "img")
        self.stats_path = os.path.join(self.run_dir, "training_stats.jsonl")

    def make_dirs(self):
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_dir, exist_ok=True)
        os.makedirs(self.last_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

        def default_serializer(obj):
            """Convert non-serializable objects to something JSON can handle."""
            try:
                return obj.__dict__  # Try converting to dict first
            except AttributeError:
                return str(obj)  # Fallback to string

        with open(os.path.join(self.run_dir, "options.json"), "w+") as f:
            json.dump(self, f, indent=4, default=default_serializer)
