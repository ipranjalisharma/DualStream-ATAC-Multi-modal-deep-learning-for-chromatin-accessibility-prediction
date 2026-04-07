"""
Test (evaluate) a trained ATAC signal model on held-out test data.

Usage:
    python test.py --model-path output/checkpoints/best_model.pth \
                   --conditions 0,1,2

If conditions are the same as training, the cached dataset is reused
automatically — no re-processing needed.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch

from model import DualStreamRegressor
from data_preparation import (
    discover_conditions, load_condition_paths,
    DataPreparationPipeline, _cache_key, load_cache, save_cache,
)
from utils import (
    set_seed, get_device, compute_all_metrics, create_logger,
)
from utils.data_loader import get_data_loaders, ATACSignalDataset
from config import PATHS, DATA_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained ATAC signal model on test data',
    )
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--conditions', type=str, required=True,
                        help='Condition indices (e.g. 0,1,2) or "all"')
    parser.add_argument('--data-dir', type=str, default=PATHS['data_dir'])
    parser.add_argument('--fasta', type=str, default=PATHS['fasta'])
    parser.add_argument('--gtf', type=str, default=PATHS['gtf'])
    parser.add_argument('--cache-dir', type=str, default=PATHS['cache_dir'])
    parser.add_argument('--output-dir', type=str, default='test_results')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(not args.cpu)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = create_logger(args.output_dir, 'testing')

    # ── Discover & select conditions ──────────────────────────────
    available = discover_conditions(args.data_dir)
    if not available:
        logger.error(f"No conditions in {args.data_dir}")
        sys.exit(1)

    if args.conditions.lower() == 'all':
        selected_idx = list(range(len(available)))
    else:
        selected_idx = [int(x.strip()) for x in args.conditions.split(',')]

    selected_names = []
    for i in selected_idx:
        if i < len(available):
            selected_names.append(available[i])

    if not selected_names:
        logger.error("No valid conditions")
        sys.exit(1)

    logger.info(f"Testing conditions: {selected_names}")

    # ── Load or prepare data ──────────────────────────────────────
    # Try multiple common prefixes used in train.py
    cached = None
    for prefix in ['test', 'mixed', None]:
        c_name = _cache_key(selected_names, args.seed, prefix=prefix)
        cached = load_cache(args.cache_dir, c_name)
        if cached is not None:
            logger.info(f"Using cache from prefix: '{prefix}'")
            break

    if cached is not None:
        all_seq = cached['sequences']
        all_expr = cached['expressions']
        all_tgt = cached['targets']
        all_info = [{'chrom': c, 'start': s, 'end': e, 'gene_id': g} 
                    for c, s, e, g in zip(cached['chroms'], cached['starts'], cached['ends'], cached['gene_ids'])]
        logger.info(f"Loaded {len(all_seq)} samples from cache")
    else:
        logger.info("Cache miss — preparing data from scratch …")

        all_seq, all_expr, all_tgt, all_info = [], [], [], []

        for cond_name in selected_names:
            atac_file, rna_file = load_condition_paths(cond_name,
                                                       args.data_dir)
            if not atac_file or not rna_file:
                logger.warning(f"Skipping {cond_name}: missing files")
                continue

            pipeline = DataPreparationPipeline(
                fasta_path=args.fasta,
                atac_bed_path=atac_file,
                rna_tsv_path=rna_file,
                gtf_path=args.gtf,
                seq_length=DATA_CONFIG['sequence_length'],
                seed=args.seed,
                signal_column=DATA_CONFIG['signal_column'],
                num_neighbors=DATA_CONFIG['num_neighbor_genes'],
                neighbor_window=DATA_CONFIG['neighbor_window_size'] // 2 # ±100kb
            )
            splits = pipeline.prepare_data()

            for split in ('train', 'val', 'test'):
                seqs, exprs, tgts, infos = splits[split]
                all_seq.extend(seqs)
                all_expr.extend(exprs)
                all_tgt.extend(tgts)
                all_info.extend(infos)

        all_expr = np.array(all_expr, dtype=np.float32)
        all_tgt = np.array(all_tgt, dtype=np.float32)

        save_cache(args.cache_dir, _cache_key(selected_names, args.seed),
                   all_seq, all_expr, all_tgt, all_info)


    # ── Extract test set (chr8, chr9) ─────────────────────────────
    test_chroms = set(DATA_CONFIG['test_chromosomes'])
    all_expr = np.asarray(all_expr, dtype=np.float32)
    all_tgt = np.asarray(all_tgt, dtype=np.float32)

    test_s, test_e, test_t = [], [], []
    train_s, train_e, train_t = [], [], []   # needed for norm stats

    for i, inf in enumerate(all_info):
        if inf['chrom'] in test_chroms:
            test_s.append(all_seq[i])
            test_e.append(all_expr[i])
            test_t.append(float(all_tgt[i]))
        else:
            train_s.append(all_seq[i])
            train_e.append(all_expr[i])
            train_t.append(float(all_tgt[i]))

    if not test_s:
        logger.error("No test-chromosome data found")
        sys.exit(1)

    logger.info(f"Test samples: {len(test_s)}")

    # ── Load norm stats (try saved; fallback to computing) ────────
    model_dir = os.path.dirname(args.model_path)
    # Walk up to find norm_stats in parent dirs
    norm_path = None
    for d in [model_dir, os.path.dirname(model_dir), '.']:
        candidate = os.path.join(d, 'norm_stats.npz')
        if os.path.exists(candidate):
            norm_path = candidate
            break

    if norm_path:
        ns = np.load(norm_path)
        mean, std = ns['mean'], ns['std']
        target_mean = float(ns.get('target_mean', 0.0))
        target_std = float(ns.get('target_std', 1.0))
        logger.info(f"Loaded norm stats from {norm_path}")
    else:
        logger.info("Computing norm stats from non-test data")
        arr = np.array(train_e, dtype=np.float32)
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        target_mean, target_std = 0.0, 1.0

    # ── Standardize Targets (BUG FIX) ─────────────────────────────
    # Model predicts Z-scores; targets must be Z-scores before the comparison loop.
    test_t = np.array(test_t, dtype=np.float32)
    test_t = (test_t - target_mean) / target_std

    # ── Build test DataLoader ─────────────────────────────────────
    from torch.utils.data import DataLoader

    test_dataset = ATACSignalDataset(
        test_s, test_e, test_t, mean=mean, std=std)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0 if args.cpu else 4,
        pin_memory=not args.cpu,
    )

    # ── Load model ────────────────────────────────────────────────
    checkpoint = torch.load(args.model_path, map_location=device,
                            weights_only=False)

    # Infer model dimensions from checkpoint
    state = checkpoint['model_state_dict']
    
    # Robustly detect expression dimension from layer names
    if 'expr_fc1.weight' in state:
        expr_dim = state['expr_fc1.weight'].shape[1]
    elif 'expr_branch.0.weight' in state:
        expr_dim = state['expr_branch.0.weight'].shape[1]
    else:
        # Fallback/Debug: print keys if we can't find it
        layers = list(state.keys())
        raise KeyError(f"Could not find expression layer in state dict. Available layers: {layers[:10]}...")


    model = DualStreamRegressor(
        seq_input_dim=5,
        seq_len=DATA_CONFIG['sequence_length'],
        expression_dim=expr_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    logger.info(f"Loaded model from {args.model_path}  "
                f"(epoch {checkpoint.get('epoch', '?')})")

    # ── Evaluate ──────────────────────────────────────────────────
    all_preds, all_targets = [], []
    criterion = torch.nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            seq = batch['sequence'].to(device)
            expr = batch['expression'].to(device)
            tgt = batch['target'].to(device)

            pred = model(seq, expr)
            total_loss += criterion(pred, tgt).item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(tgt.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # ── Calculate Metrics (Standardised Space) ───────────────────
    # This is the "apples-to-apples" comparison with your train.py logs.
    metrics_std = compute_all_metrics(all_targets, all_preds)

    # ── Calculate Metrics (Raw Log₂ Space) ──────────────────────
    # This reflects the real biological signal error.
    all_preds_raw = (all_preds * target_std) + target_mean
    all_targets_raw = (all_targets * target_std) + target_mean
    metrics_raw = compute_all_metrics(all_targets_raw, all_preds_raw)

    # ── Report ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST RESULTS: STANDARDISED SPACE (Z-Scores)")
    print("-" * 70)
    print(f"  MSE:          {metrics_std['mse']:.6f}")
    print(f"  R²:           {metrics_std['r2']:.6f}")
    print(f"  Pearson r:    {metrics_std['pearson_r']:.6f}")
    print(f"  Spearman ρ:   {metrics_std['spearman_rho']:.6f}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("TEST RESULTS: RAW SIGNAL SPACE (Log₂ Fold-Enrichment)")
    print("-" * 70)
    print(f"  Samples:      {len(all_preds)}")
    print(f"  MSE:          {metrics_raw['mse']:.6f}")
    print(f"  RMSE:         {metrics_raw['rmse']:.6f}")
    print(f"  MAE:          {metrics_raw['mae']:.6f}")
    print(f"  R²:           {metrics_raw['r2']:.6f}")
    print(f"  Pearson r:    {metrics_raw['pearson_r']:.6f}")
    print(f"  Spearman ρ:   {metrics_raw['spearman_rho']:.6f}")
    print("=" * 70)

    # Save
    results = {
        'model_path': args.model_path,
        'conditions': selected_names,
        'n_test_samples': len(all_preds),
        'metrics_std': metrics_std,
        'metrics_raw': metrics_raw,
    }
    results_path = os.path.join(args.output_dir, 'test_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {results_path}")

    np.savez(os.path.join(args.output_dir, 'test_predictions.npz'),
             predictions=all_preds, targets=all_targets)
    logger.info("Predictions saved → test_predictions.npz")


if __name__ == '__main__':
    main()
