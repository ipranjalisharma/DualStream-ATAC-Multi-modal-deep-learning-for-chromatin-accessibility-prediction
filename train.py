"""
Training script for ATAC signal regression model.

Usage:
    python train.py --conditions 0,1,2
    python train.py --conditions all
    python train.py --conditions 0 --epochs 50 --batch-size 64
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import DualStreamRegressor
from data_preparation import (
    DataPreparationPipeline, discover_conditions, load_condition_paths,
    _cache_key, save_cache, load_cache,
)
from utils import (
    create_logger, set_seed, get_device, save_checkpoint,
    compute_all_metrics, save_hp_config,
)
from utils.data_loader import get_data_loaders
from config import DEFAULT_HYPERPARAMS, TRAINING_CONFIG, PATHS, DATA_CONFIG


# ═══════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════

class Trainer:
    """Training loop with early stopping for regression model."""

    def __init__(self, model, device, logger, hyperparams, config):
        self.model = model
        self.device = device
        self.logger = logger
        self.hyperparams = hyperparams
        self.config = config

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams['learning_rate'],
            weight_decay=hyperparams['l2_regularization'],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min',
            factor=config['lr_scheduler_factor'],
            patience=config['lr_scheduler_patience'],
            min_lr=1e-6,
        )
        self.criterion = nn.HuberLoss(delta=1.0)

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0

    # ── single epoch ──────────────────────────────────────────────────
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss, n_batches = 0.0, 0

        pbar = tqdm(train_loader, desc='  train', leave=False)
        for batch in pbar:
            seq = batch['sequence'].to(self.device)
            expr = batch['expression'].to(self.device)
            tgt = batch['target'].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(seq, expr)
            loss = self.criterion(pred, tgt)

            l2 = self.model.get_l2_loss()
            (loss + l2).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'mse': total_loss / n_batches})

        return total_loss / max(n_batches, 1)

    # ── validation ────────────────────────────────────────────────────
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                seq = batch['sequence'].to(self.device)
                expr = batch['expression'].to(self.device)
                tgt = batch['target'].to(self.device)

                pred = self.model(seq, expr)
                loss = self.criterion(pred, tgt)

                total_loss += loss.item()
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(tgt.cpu().numpy())

        avg_loss = total_loss / max(len(val_loader), 1)
        metrics = compute_all_metrics(np.array(all_targets),
                                      np.array(all_preds))
        return avg_loss, metrics, np.array(all_preds), np.array(all_targets)

    # ── full training loop ────────────────────────────────────────────
    def train(self, train_loader, val_loader, save_dir):
        """Train with early stopping.  Save ALL epoch checkpoints."""
        os.makedirs(save_dir, exist_ok=True)
        ckpt_dir = os.path.join(save_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)

        self.logger.info("=" * 70)
        self.logger.info("Starting training  |  early-stop patience = "
                         f"{self.config['early_stopping_patience']}")
        self.logger.info(f"Hyperparams: {self.hyperparams}")
        self.logger.info("=" * 70)

        best_metrics = None

        for epoch in range(self.hyperparams['epochs']):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, metrics, _, _ = self.validate(val_loader)
            dt = time.time() - t0

            self.scheduler.step(val_loss)

            msg = (f"Epoch {epoch+1:3d}/{self.hyperparams['epochs']}  |  "
                   f"train_mse={train_loss:.6f}  val_mse={val_loss:.6f}  |  "
                   f"R²={metrics['r2']:.4f}  "
                   f"pearson={metrics['pearson_r']:.4f}  "
                   f"MAE={metrics['mae']:.4f}  |  {dt:.1f}s")
            self.logger.info(msg)
            print(msg)

            # Save EVERY epoch checkpoint
            is_best = val_loss < self.best_val_loss - \
                self.config['early_stopping_min_delta']

            ckpt_path = os.path.join(ckpt_dir,
                                     f'epoch_{epoch+1:04d}.pth')
            save_checkpoint(self.model, self.optimizer, epoch + 1,
                            metrics, ckpt_path, is_best=is_best)

            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.no_improve_count = 0
                best_metrics = metrics
                self.logger.info(
                    f"  ★ New best model at epoch {epoch+1}")
            else:
                self.no_improve_count += 1

            if self.no_improve_count >= self.config['early_stopping_patience']:
                self.logger.info(
                    f"\nEarly stopping at epoch {epoch+1}.  "
                    f"Best epoch: {self.best_epoch}  "
                    f"val_mse={self.best_val_loss:.6f}")
                break

        # Label the best checkpoint
        if best_metrics is not None:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Best model: epoch {self.best_epoch}")
            self.logger.info(f"Metrics: {best_metrics}")
            self.logger.info(f"Saved as: {save_dir}/checkpoints/best_model.pth")
            self.logger.info(f"{'='*70}\n")

        return best_metrics


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train ATAC signal regression model',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--conditions', type=str, required=True,
                        help='Condition indices (e.g. 0,1,2) or "all"')
    parser.add_argument('--data-dir', type=str, default=PATHS['data_dir'],
                        help='Data directory')
    parser.add_argument('--fasta', type=str, default=PATHS['fasta'],
                        help='Genome FASTA path')
    parser.add_argument('--gtf', type=str, default=PATHS['gtf'],
                        help='GTF annotation path')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--cache-dir', type=str, default=PATHS['cache_dir'],
                        help='Cache directory for processed datasets')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-frac-cond', type=float, default=0.3,
                        help='Fraction of conditions to hold out for testing')
    parser.add_argument('--val-frac-cond', type=float, default=0.1,
                        help='Fraction of conditions (from remaining) for validation')

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(not args.cpu)
    os.makedirs(args.output_dir, exist_ok=True)

    logger = create_logger(args.output_dir, 'training')
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Data:   {args.data_dir}")

    # ── Discover conditions ───────────────────────────────────────
    available = discover_conditions(args.data_dir)
    if not available:
        logger.error(f"No conditions found in {args.data_dir}")
        sys.exit(1)

    # Parse selection pool
    if args.conditions.lower() == 'all':
        selection_idx = list(range(len(available)))
    else:
        selection_idx = [int(x.strip()) for x in args.conditions.split(',')]

    pool_names = [available[i] for i in selection_idx if i < len(available)]
    if not pool_names:
        logger.error("No valid conditions selected")
        sys.exit(1)

    # ── Partition Conditions ──────────────────────────────────────
    # Shuffle and split into Train / Val / Test pools (Condition-wise)
    np.random.seed(args.seed)
    all_names = sorted(pool_names)
    np.random.shuffle(all_names)

    n = len(all_names)
    
    if n >= 3:
        # High-diversity mode: dedicated cell-type holdout
        n_test = max(1, int(n * args.test_frac_cond))
        n_val = max(1, int((n - n_test) * args.val_frac_cond))
        
        test_pool = all_names[:n_test]
        val_pool = all_names[n_test : n_test + n_val]
        train_pool = all_names[n_test + n_val:]
        mode = "Condition-Holdout (Cross-Cell)"
    else:
        # Low-diversity mode: intra-condition split fallback
        train_pool = all_names
        val_pool = []
        test_pool = []
        mode = "Intra-Condition (Shuffle-Split)"

    logger.info("=" * 70)
    logger.info(f"STRATEGY: {mode}")
    logger.info(f"  Train Pool: {len(train_pool)} cell types")
    logger.info(f"  Val Pool:   {len(val_pool)} cell types")
    logger.info(f"  Test Pool:  {len(test_pool)} cell types")
    logger.info("-" * 70)
    if test_pool: logger.info(f"  Test Cell Types: {test_pool}")
    logger.info("=" * 70)

    # ── Data Loading Logic ────────────────────────────────────────
    def get_pool_data(names, prefix, split_mode='combine'):
        if not names: return [], [], [], []
        
        c_name = _cache_key(names, args.seed, prefix=prefix)
        cached = load_cache(args.cache_dir, c_name)
        
        if cached is not None:
            logger.info(f"Loaded {prefix} pool ({len(names)} conds) from cache")
            return cached['sequences'], cached['expressions'], cached['targets'], [
                {'chrom': c, 'start': s, 'end': e, 'gene_id': g}
                for c, s, e, g in zip(cached['chroms'], cached['starts'], cached['ends'], cached['gene_ids'])
            ]
        
        logger.info(f"Building {prefix} pool from scratch …")
        
        all_s, all_e, all_t, all_i = [], [], [], []
        v_s, v_e, v_t, v_i = [], [], [], [] # for internal split
        te_s, te_e, te_t, te_i = [], [], [], [] # for internal split
        
        for cond_name in names:
            atac, rna = load_condition_paths(cond_name, args.data_dir)
            if not atac or not rna: continue
            
            p = DataPreparationPipeline(
                args.fasta, atac, rna, gtf_path=args.gtf,
                seq_length=DATA_CONFIG['sequence_length'], seed=args.seed,
                signal_column=DATA_CONFIG['signal_column'],
                num_neighbors=DATA_CONFIG['num_neighbor_genes'],
                neighbor_window=DATA_CONFIG['neighbor_window_size'] // 2 # ±100kb
            )
            splits = p.prepare_data(val_frac=TRAINING_CONFIG['val_split'])
            
            if split_mode == 'combine':
                # Traditional V4 logic: combine all three internal splits into one pool
                for sn in ('train', 'val', 'test'):
                    s, e, t, i = splits[sn]
                    all_s.extend(s); all_e.extend(e); all_t.extend(t); all_i.extend(i)
            else:
                # Fallback logic: keep them separate
                s, e, t, i = splits['train']
                all_s.extend(s); all_e.extend(e); all_t.extend(t); all_i.extend(i)
                s, e, t, i = splits['val']
                v_s.extend(s); v_e.extend(e); v_t.extend(t); v_i.extend(i)
                s, e, t, i = splits['test']
                te_s.extend(s); te_e.extend(e); te_t.extend(t); te_i.extend(i)
        
        # Cache management
        if split_mode == 'combine':
            if all_s: save_cache(args.cache_dir, c_name, all_s, all_e, all_t, all_i)
            return all_s, np.array(all_e, dtype=np.float32), np.array(all_t, dtype=np.float32), all_i
        else:
            # For intra-split, we don't cache the combined multi-output to keep it simple
            return (all_s, np.array(all_e), np.array(all_t), all_i), \
                   (v_s, np.array(v_e), np.array(v_t), v_i), \
                   (te_s, np.array(te_e), np.array(te_t), te_i)

    # ── Orchestrate Split ─────────────────────────────────────────
    if n >= 3:
        tr_s, tr_e, tr_t, tr_i = get_pool_data(train_pool, 'train')
        val_s, val_e, val_t, val_i = get_pool_data(val_pool, 'val')
        te_s, te_e, te_t, te_i = get_pool_data(test_pool, 'test')
    else:
        # Use internal 85/15/test splitting
        (tr_s, tr_e, tr_t, tr_i), (val_s, val_e, val_t, val_i), (te_s, te_e, te_t, te_i) = \
            get_pool_data(train_pool, 'mixed', split_mode='internal')

    # ── Final Processing ───────────────────────────────────────────
    # Standardize based on TRAIN pool stats
    tr_t = np.asarray(tr_t, dtype=np.float32)
    val_t = np.asarray(val_t, dtype=np.float32)
    te_t = np.asarray(te_t, dtype=np.float32)

    target_mean, target_std = tr_t.mean(), tr_t.std()
    if target_std == 0: target_std = 1.0
    
    logger.info(f"Target stats (train): mean={target_mean:.4f}, std={target_std:.4f}")
    
    # Scale ALL pools by TRAIN stats
    tr_t = (tr_t - target_mean) / target_std
    val_t = (val_t - target_mean) / target_std
    te_t = (te_t - target_mean) / target_std

    # Build Dataloaders
    train_loader, val_loader, test_loader, norm_stats = get_data_loaders(
        tr_s, tr_e, tr_t,
        val_s, val_e, val_t,
        te_s, te_e, te_t,    # out-of-condition test set
        batch_size=args.batch_size,
        use_gpu=not args.cpu,
        augment_train=True,
    )

    # Save stats
    np.savez(os.path.join(args.output_dir, 'norm_stats.npz'),
             mean=norm_stats['mean'], std=norm_stats['std'],
             target_mean=target_mean, target_std=target_std)

    # ── Build & Train model ───────────────────────────────────────
    # Synchronize CLI arguments with hyperparameters
    hyperparams = DEFAULT_HYPERPARAMS.copy()
    hyperparams['learning_rate'] = args.learning_rate
    hyperparams['batch_size'] = args.batch_size
    hyperparams['epochs'] = args.epochs
    
    training_cfg = TRAINING_CONFIG.copy()

    expr_dim = tr_e.shape[1] if len(tr_e) > 0 else 1
    model = DualStreamRegressor(
        seq_input_dim=5, seq_len=DATA_CONFIG['sequence_length'],
        expression_dim=expr_dim,
        num_filters=hyperparams['num_filters'],
        kernel_size=hyperparams['kernel_size'],
        num_conv_layers=hyperparams['num_conv_layers'],
        hidden_dim=hyperparams['hidden_dim'],
        dropout_rate=hyperparams['dropout_rate'],
        l2_reg=hyperparams['l2_regularization'],
    ).to(device)

    trainer = Trainer(model, device, logger, hyperparams, training_cfg)
    best_metrics = trainer.train(train_loader, val_loader, args.output_dir)


    # ── Final Test Evaluation (Held-out Cell Types) ───────────────
    if len(te_s) > 0:
        logger.info("\nEvaluating on HELD-OUT CONDITION test set …")
        _, metrics_std, preds_std, targets_std = trainer.validate(test_loader)
        
        preds_raw = (preds_std * target_std) + target_mean
        targets_raw = (targets_std * target_std) + target_mean
        metrics_raw = compute_all_metrics(targets_raw, preds_raw)

        logger.info(f"Test (Standardized): {metrics_std}")
        logger.info(f"Test (Raw Log₂):      {metrics_raw}")

        np.savez(os.path.join(args.output_dir, 'test_predictions.npz'),
                 predictions=preds_std, targets=targets_std,
                 predictions_raw=preds_raw, targets_raw=targets_raw,
                 test_conditions=test_pool)

    logger.info("\n✓ V4 Training complete (Condition Holdout active).")


if __name__ == '__main__':
    main()
