"""
Predict ATAC signal for new DNA sequences using a trained model.

Workflow:
  1. User provides DNA sequence(s) + RNA-seq expression file
  2. Align each sequence to hg38 using minimap2 (mappy) to get genomic coords
  3. Find nearest gene(s) via GTF annotation
  4. Look up expression in user's RNA-seq file
  5. Feed one-hot DNA + expression into model → predicted log2 ATAC signal

Usage:
    # Single sequence
    python predict.py --sequence ATCGATCG... \\
                      --rna-tsv my_rnaseq.tsv \\
                      --model-path output/checkpoints/best_model.pth

    # Batch from FASTA
    python predict.py --fasta-input sequences.fa \\
                      --rna-tsv my_rnaseq.tsv \\
                      --model-path output/checkpoints/best_model.pth

    # Provide expression directly (skip alignment + gene lookup)
    python predict.py --sequence ATCGATCG... \\
                      --expression 5.3 \\
                      --model-path output/checkpoints/best_model.pth
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

from model import DualStreamRegressor
from utils.data_loader import one_hot_encode
from config import PATHS, DATA_CONFIG


# ═══════════════════════════════════════════════════════════════════════════
# Alignment helpers
# ═══════════════════════════════════════════════════════════════════════════

def align_sequence_minimap2(sequence, fasta_path):
    """
    Align a DNA sequence to the reference genome using minimap2 (mappy).

    Returns:
        dict with keys: chrom, start, end, strand, mapq
        or None if no alignment found.
    """
    try:
        import mappy as mp
    except ImportError:
        print("ERROR: 'mappy' is not installed.  Install with:")
        print("  pip install mappy")
        print("Or install minimap2 system-wide and use the Python binding.")
        sys.exit(1)

    aligner = mp.Aligner(fasta_path, preset='sr')  # short-read preset
    if not aligner:
        print(f"ERROR: Could not load index for {fasta_path}")
        return None

    best_hit = None
    for hit in aligner.map(sequence):
        if best_hit is None or hit.mapq > best_hit.mapq:
            best_hit = hit

    if best_hit is None:
        return None

    return {
        'chrom': best_hit.ctg,
        'start': best_hit.r_st,
        'end': best_hit.r_en,
        'strand': '+' if best_hit.strand == 1 else '-',
        'mapq': best_hit.mapq,
    }


def load_gtf_genes(gtf_path, gene_names=None):
    """
    Load gene coordinates from GTF.

    Args:
        gtf_path: Path to GTF annotation.
        gene_names: Optional set of gene names to filter (for speed).

    Returns:
        dict: {gene_name: {'chrom', 'start', 'end', 'strand'}}
    """
    genes = {}
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            if parts[2] != 'gene':
                continue
            info = parts[8]
            # Extract gene_name
            import re
            m = re.search(r'gene_name "([^"]+)"', info)
            if not m:
                continue
            gn = m.group(1)
            if gene_names and gn not in gene_names:
                continue
            if gn not in genes:
                genes[gn] = {
                    'chrom': parts[0],
                    'start': int(parts[3]),
                    'end': int(parts[4]),
                    'strand': parts[6],
                }
    return genes


def find_neighborhood_genes(chrom, start, end, gene_coords, num_neighbors=5, window=100000):
    """Find the N nearest genes within a window around a genomic region."""
    centre = (start + end) // 2
    w_start = centre - window
    w_end = centre + window
    
    matches = []
    for gene, c in gene_coords.items():
        if c['chrom'] != chrom:
            continue
        
        # Overlap check with the search window
        if max(c['start'], w_start) <= min(c['end'], w_end):
            tss = c['start'] if c.get('strand') == '+' else c['end']
            dist = abs(centre - tss)
            matches.append((dist, gene))
            
    matches.sort(key=lambda x: x[0])
    return matches[:num_neighbors]


def get_neighborhood_expression(neighbors, rna_df, num_neighbors=5):
    """
    Look up expression for a list of genes. 
    Returns a flat vector of size num_neighbors.
    """
    expr_vals = []
    for dist, gene_name in neighbors:
        if gene_name in rna_df.index:
            val = rna_df.loc[gene_name].values[0]
            expr_vals.append(val)
            
    # Padding
    if len(expr_vals) < num_neighbors:
        pad_size = num_neighbors - len(expr_vals)
        expr_vals.extend([0.0] * pad_size)
        
    return np.array(expr_vals[:num_neighbors], dtype=np.float32)


def parse_fasta(path):
    """Parse a FASTA file → list of (name, sequence) tuples."""
    records = []
    name, seq_parts = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    records.append((name, ''.join(seq_parts)))
                name = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.upper())
    if name is not None:
        records.append((name, ''.join(seq_parts)))
    return records


# ═══════════════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════════════

def predict_single(model, device, sequence, expression, norm_mean, norm_std,
                   target_mean=0.0, target_std=1.0, seq_length=1000):
    """
    Run model on a single sequence + expression vector.

    Returns:
        float: predicted log2 ATAC signal
    """
    model.eval()

    # Pad / trim sequence
    seq = sequence.upper()
    if len(seq) < seq_length:
        seq = seq + 'N' * (seq_length - len(seq))
    seq = seq[:seq_length]

    encoded = one_hot_encode(seq, seq_length)              # (L, 5)
    seq_tensor = torch.from_numpy(encoded).unsqueeze(0)    # (1, L, 5)

    expr = np.array(expression, dtype=np.float32).reshape(1, -1)
    expr = (expr - norm_mean) / norm_std
    expr_tensor = torch.from_numpy(expr)

    with torch.no_grad():
        seq_tensor = seq_tensor.to(device)
        expr_tensor = expr_tensor.to(device)
        pred = model(seq_tensor, expr_tensor)

    # De-standardise
    final_pred = (float(pred.cpu().item()) * target_std) + target_mean
    return final_pred


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Predict ATAC signal for new DNA sequences',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--sequence', type=str,
                     help='Single DNA sequence string')
    grp.add_argument('--fasta-input', type=str,
                     help='FASTA file with one or more sequences')

    parser.add_argument('--rna-tsv', type=str, default=None,
                        help='RNA-seq expression file (gene × samples TSV)')
    parser.add_argument('--expression', type=float, nargs='+', default=None,
                        help='Provide expression value(s) directly, '
                             'skipping alignment + gene lookup')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--genome-fasta', type=str, default=PATHS['fasta'],
                        help='Genome FASTA for alignment (default: hg38.fa)')
    parser.add_argument('--gtf', type=str, default=PATHS['gtf'],
                        help='GTF annotation (default: gencode v45)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output', type=str, default=None,
                        help='Save predictions to TSV file')

    args = parser.parse_args()

    device = torch.device('cpu') if args.cpu else (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))

    # ── Load model ────────────────────────────────────────────────
    checkpoint = torch.load(args.model_path, map_location=device,
                            weights_only=False)
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

    print(f"Model loaded from {args.model_path}  "
          f"(epoch {checkpoint.get('epoch', '?')})")

    # ── Load norm stats ───────────────────────────────────────────
    model_dir = os.path.dirname(args.model_path)
    norm_path = None
    for d in [model_dir, os.path.dirname(model_dir), '.']:
        candidate = os.path.join(d, 'norm_stats.npz')
        if os.path.exists(candidate):
            norm_path = candidate
            break

    if norm_path:
        ns = np.load(norm_path)
        norm_mean, norm_std = ns['mean'], ns['std']
        target_mean = float(ns.get('target_mean', 0.0))
        target_std = float(ns.get('target_std', 1.0))
    else:
        print("WARNING: norm_stats.npz not found — using zeros/ones")
        norm_mean = np.zeros((1, expr_dim), dtype=np.float32)
        norm_std = np.ones((1, expr_dim), dtype=np.float32)
        target_mean, target_std = 0.0, 1.0

    # ── Prepare RNA-seq + GTF if needed ───────────────────────────
    rna_df = None
    gene_coords = None

    if args.expression is None:
        # Need alignment → gene lookup → expression
        if args.rna_tsv is None:
            print("ERROR: Provide --rna-tsv or --expression")
            sys.exit(1)

        rna_df = pd.read_csv(args.rna_tsv, sep='\t', index_col=0)
        rna_df = rna_df.iloc[:, [0]].fillna(0)
        print(f"RNA-seq: {rna_df.shape[0]} genes")

        gtf_path = args.gtf
        if os.path.exists(gtf_path):
            print("Loading GTF annotation (this may take a minute) …")
            gene_coords = load_gtf_genes(gtf_path,
                                         gene_names=set(rna_df.index))
            print(f"  {len(gene_coords)} genes mapped")
        else:
            print(f"WARNING: GTF not found at {gtf_path}")
            gene_coords = {}

    # ── Collect sequences ─────────────────────────────────────────
    sequences = []
    if args.sequence:
        sequences = [('input_seq', args.sequence.upper())]
    elif args.fasta_input:
        sequences = parse_fasta(args.fasta_input)
        print(f"Loaded {len(sequences)} sequences from {args.fasta_input}")

    # ── Predict ───────────────────────────────────────────────────
    results = []
    seq_length = DATA_CONFIG['sequence_length']

    for name, seq in sequences:
        if args.expression is not None:
            # User-provided expression — no alignment needed
            expr_vec = np.array(args.expression, dtype=np.float32)
            if len(expr_vec) != expr_dim:
                print(f"WARNING: --expression has {len(expr_vec)} values "
                      f"but model expects {expr_dim}. Padding/truncating.")
                padded = np.zeros(expr_dim, dtype=np.float32)
                n = min(len(expr_vec), expr_dim)
                padded[:n] = expr_vec[:n]
                expr_vec = padded

            pred = predict_single(model, device, seq, expr_vec,
                                  norm_mean, norm_std, 
                                  target_mean, target_std, seq_length)
            results.append({
                'name': name, 'seq_len': len(seq),
                'chrom': '-', 'start': '-', 'end': '-',
                'nearest_gene': '-',
                'expression': expr_vec.tolist(),
                'predicted_log2_signal': pred,
            })
        else:
            # Align → find neighborhood → lookup expressions
            alignment = align_sequence_minimap2(seq, args.genome_fasta)
            if alignment is None:
                print(f"  {name}: no alignment found — skipping")
                continue

            chrom = alignment['chrom']
            start = alignment['start']
            end = alignment['end']

            neighbors = find_neighborhood_genes(chrom, start, end, 
                                               gene_coords, 
                                               num_neighbors=expr_dim)
            expr_vec = get_neighborhood_expression(neighbors, rna_df, 
                                                  num_neighbors=expr_dim)
            
            # Report neighbor status
            if neighbors:
                gene_list = ", ".join([f"{n[1]} ({n[0]/1000:.1f}kb)" for n in neighbors])
                print(f"  {name} mapped to {chrom}:{start}-{end} near {gene_list}")
            else:
                print(f"  {name} mapped to {chrom}:{start}-{end} (no nearby genes)")


            pred = predict_single(model, device, seq, expr_vec,
                                  norm_mean, norm_std, 
                                  target_mean, target_std, seq_length)
            
            results.append({
                'name': name, 'seq_len': len(seq),
                'chrom': chrom, 'start': start, 'end': end,
                'nearest_genes': [n[1] for n in neighbors],
                'predicted_log2_signal': pred,
            })

    # ── Output ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PREDICTIONS")
    print("=" * 70)
    for r in results:
        sig = r['predicted_log2_signal']
        sig_str = f"{sig:.4f}" if sig is not None else "N/A"
        
        genes_str = ",".join(r.get('nearest_genes', []))
        if not genes_str:
            genes_str = "None"
            
        print(f"  {r['name']:30s}  "
              f"{r['chrom']}:{r['start']}-{r['end']}  "
              f"genes={genes_str:20s}  "
              f"log2_signal={sig_str}")
    print("=" * 70)

    if args.output:
        rows = []
        for r in results:
            rows.append({
                'name': r['name'],
                'seq_len': r['seq_len'],
                'chrom': r['chrom'],
                'start': r['start'],
                'end': r['end'],
                'nearest_genes': ",".join(r.get('nearest_genes', [])),
                'predicted_log2_signal': r['predicted_log2_signal'],
            })
        df = pd.DataFrame(rows)
        df.to_csv(args.output, sep='\t', index=False)
        print(f"\nResults saved → {args.output}")


if __name__ == '__main__':
    main()
