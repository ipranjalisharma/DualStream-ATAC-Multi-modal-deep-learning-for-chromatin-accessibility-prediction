"""Data loading utilities for ATAC signal regression model."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# DNA nucleotide encoding
NUCLEOTIDE_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}


def reverse_complement(sequence):
    """Return the reverse complement of a DNA sequence."""
    return "".join(COMPLEMENT.get(base, 'N') for base in reversed(sequence))


def one_hot_encode(sequence, max_length=1000):
    """
    One-hot encode a DNA sequence.

    Args:
        sequence: DNA sequence string (uppercase).
        max_length: Maximum sequence length (pads with N if shorter).

    Returns:
        np.ndarray: One-hot encoded array of shape (max_length, 5).
    """
    if len(sequence) < max_length:
        sequence = sequence + 'N' * (max_length - len(sequence))

    sequence = sequence[:max_length].upper()
    one_hot = np.zeros((max_length, 5), dtype=np.float32)

    for i, nucleotide in enumerate(sequence):
        idx = NUCLEOTIDE_TO_INDEX.get(nucleotide, 4)
        one_hot[i, idx] = 1.0

    return one_hot


class ATACSignalDataset(Dataset):
    """PyTorch Dataset for DNA sequences + RNA-seq → continuous ATAC signal."""

    def __init__(self, sequences, expressions, targets, encoding='onehot',
                 max_length=1000, mean=None, std=None, augment=False):
        """
        Args:
            sequences: List of DNA sequence strings.
            expressions: List/array of expression vectors (one per sample).
            targets: List/array of continuous log2 signal values.
            encoding: Sequence encoding type ('onehot').
            max_length: Maximum sequence length.
            mean: Pre-computed expression mean for normalisation (optional).
            std: Pre-computed expression std for normalisation (optional).
            augment: Whether to perform reverse complement augmentation.
        """
        self.sequences = sequences
        self.expressions = np.array(expressions, dtype=np.float32)
        self.targets = np.array(targets, dtype=np.float32)
        self.encoding = encoding
        self.max_length = max_length
        self.augment = augment

        assert len(sequences) == len(targets) == len(self.expressions), (
            f"Length mismatch: sequences={len(sequences)}, "
            f"expressions={len(self.expressions)}, targets={len(targets)}"
        )

        # Normalise expression features
        if mean is None or std is None:
            mean = self.expressions.mean(axis=0, keepdims=True)
            std = self.expressions.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
        self.mean = mean
        self.std = std
        self.expressions = (self.expressions - mean) / std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        expression = torch.from_numpy(self.expressions[idx])
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        # Reverse complement augmentation (50% chance)
        if self.augment and np.random.random() > 0.5:
            sequence = reverse_complement(sequence)

        if self.encoding == 'onehot':
            encoded = one_hot_encode(sequence, self.max_length)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

        return {
            'sequence': torch.from_numpy(encoded),
            'expression': expression,
            'target': target,
        }


def get_data_loaders(sequences_train, expressions_train, targets_train,
                     sequences_val, expressions_val, targets_val,
                     sequences_test, expressions_test, targets_test,
                     batch_size=32, num_workers=4,
                     encoding='onehot', pin_memory=True,
                     use_gpu=True, augment_train=True):
    """
    Create PyTorch DataLoaders for train/val/test.

    Normalises expression features using training-set statistics.

    Returns:
        tuple: (train_loader, val_loader, test_loader, norm_stats)
               norm_stats = {'mean': ndarray, 'std': ndarray}
    """
    # Compute normalisation stats from training set
    train_expr_arr = np.array(expressions_train, dtype=np.float32)
    mean = train_expr_arr.mean(axis=0, keepdims=True)
    std = train_expr_arr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    train_dataset = ATACSignalDataset(
        sequences_train, expressions_train, targets_train, encoding,
        mean=mean, std=std, augment=augment_train
    )
    val_dataset = ATACSignalDataset(
        sequences_val, expressions_val, targets_val, encoding,
        mean=mean, std=std, augment=False
    )
    test_dataset = ATACSignalDataset(
        sequences_test, expressions_test, targets_test, encoding,
        mean=mean, std=std, augment=False
    )

    if not use_gpu:
        num_workers = 0
        pin_memory = False
    else:
        num_workers = min(num_workers, 4)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )

    norm_stats = {'mean': mean, 'std': std}
    return train_loader, val_loader, test_loader, norm_stats
