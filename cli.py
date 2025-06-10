import argparse
import yaml

from .training.train_controlnet import train_controlnet
from .training.train_paintnet import train_paintnet
from .generate import generate_images


def cli():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="CLI for running tasks with a YAML config file."
    )
    parser.add_argument("--task", type=str, help="Name of the task to execute.")
    parser.add_argument(
        "--config", type=str, help="Path to the YAML configuration file.", default={}
    )
    args = parser.parse_args()

    # Load YAML config
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = args.config

    if args.task == "generate":
        pass
    elif args.task == "train_controlnet":
        train_controlnet(opts=config)
    elif args.task == "train_paintnet":
        train_paintnet(opts=config)
    else:
        raise Error(f"Task '{args.task}' not a valid option.")
