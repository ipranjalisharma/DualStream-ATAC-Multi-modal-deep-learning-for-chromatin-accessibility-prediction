"""Hyperparameter configuration for ATAC signal regression model."""

import os

# ─── Paths (hardcoded defaults — override via CLI if needed) ─────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

PATHS = {
    'fasta': os.path.join(PARENT_DIR, 'hg38.fa'),
    'gtf': os.path.join(PARENT_DIR, 'gencode.v45.annotation.gtf'),
    'data_dir': os.path.join(PARENT_DIR, 'data'),
    'cache_dir': os.path.join(BASE_DIR, 'cache'),
}

# ─── Default hyperparameters ─────────────────────────────────────────────────
DEFAULT_HYPERPARAMS = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'dropout_rate': 0.4,
    'l2_regularization': 0.0001,
    'num_filters': 64,
    'kernel_size': 12,
    'num_conv_layers': 6,
    'hidden_dim': 128,
}

# ─── Training configuration ─────────────────────────────────────────────────
TRAINING_CONFIG = {
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 1e-4,
    'lr_scheduler_patience': 5,
    'lr_scheduler_factor': 0.5,
    'seed': 42,
    'val_split': 0.15,
    'save_interval': 1,        # save checkpoint every N epochs
    'log_interval': 10,
}

# ─── Data configuration ─────────────────────────────────────────────────────
DATA_CONFIG = {
    'sequence_length': 1000,
    'encoding': 'onehot',
    'test_chromosomes': ['chr8', 'chr9'],
    'random_seed': 42,
    'signal_column': 6,          # 0-indexed col 6 = narrowPeak signalValue (col 7 in 1-based)
    'negative_positive_ratio': 1.0,
    'num_neighbor_genes': 5,      # Top-N neighbors to track (V6)
    'neighbor_window_size': 200000, # ±100kb search window
}

