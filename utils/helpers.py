"""Helper utilities for training and evaluation."""

import os
import json
import logging
import random
import torch
import numpy as np


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(use_gpu=True):
    """Get device (GPU if available and requested, else CPU)."""
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def create_logger(log_dir, name='model'):
    """Create a logger for training/evaluation."""
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'{name}.log')

    logger = logging.getLogger(name)
    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path,
                    is_best=False):
    """Save model checkpoint.  Always saves; optionally copies as best."""
    os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = os.path.join(os.path.dirname(checkpoint_path),
                                 'best_model.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(model, optimizer, checkpoint_path, device='cpu'):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint.get('metrics', {})


def log_metrics(logger, epoch, metrics_dict, phase='train', duration=None):
    """Log metrics for an epoch."""
    msg = f"Epoch {epoch:3d} [{phase}]:"
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            msg += f" {key}={value:.6f}"
        else:
            msg += f" {key}={value}"

    if duration is not None:
        msg += f" time={duration:.2f}s"

    logger.info(msg)


def save_hp_config(hp_dict, save_path):
    """Save hyperparameter configuration to JSON."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(hp_dict, f, indent=2)


def load_hp_config(config_path):
    """Load hyperparameter configuration from JSON."""
    with open(config_path, 'r') as f:
        return json.load(f)
