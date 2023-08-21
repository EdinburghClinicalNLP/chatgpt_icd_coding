import os
import random
from datetime import datetime
from typing import Tuple

import numpy as np
import yaml

from .configs import Configs


def load_yaml(filepath: str) -> dict:
    """
    Utility function to load yaml file, mainly for config files.

    Args:
        filepath (str): Path to the config file.

    Raises:
        exc: Stop process if there is a problem when loading the file.

    Returns:
        dict: Training configs.
    """
    with open(filepath, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc


def setup_random_seed(seed: int) -> None:
    """
    Utility function to setup random seed. Apply this function early on the training script.

    Args:
        seed (int): Integer indicating the desired seed.
        is_deterministic (bool, optional): Set deterministic flag of CUDNN. Defaults to True.
    """
    # set the seeds
    random.seed(seed)
    np.random.seed(seed)


def setup_experiment_folder(outputs_dir: str) -> Tuple[str, str]:
    """
    Utility function to create and setup the experiment output directory.

    Args:
        outputs_dir (str): The parent directory to store
            all outputs across experiments.

    Returns:
        Tuple[str, str]:
            outputs_dir: Directory of the outputs.
    """
    now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    outputs_dir = os.path.join(outputs_dir, now)
    os.makedirs(outputs_dir, exist_ok=True)

    return outputs_dir


def save_training_configs(configs: Configs, output_dir: str):
    """
    Save training config
    Args:
        configs (Configs): Configs used during training for reproducibility
        output_dir (str): Path to the output directory
    """
    filepath = os.path.join(output_dir, "configs.yaml")
    with open(filepath, "w") as file:
        _ = yaml.dump(configs.dict(), file)
