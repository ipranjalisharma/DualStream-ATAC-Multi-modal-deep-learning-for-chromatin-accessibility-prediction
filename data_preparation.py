"""
Data preparation pipeline for ATAC signal regression.

Reads narrowPeak files (signal column = fold-enrichment), applies
log2(signal + 1) transformation, generates positive (peak) and negative
(non-peak) examples, and caches processed datasets for fast re-use.
"""

import os
import hashlib
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm
import math
import warnings

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _cache_key(condition_names, seed, prefix=None):
    """Deterministic cache key from sorted condition names + seed."""
    sorted_names = sorted(condition_names)
    tag = ','.join(sorted_names) + f'_seed{seed}'
    if prefix:
        tag = prefix + '_' + tag
    h = hashlib.md5(tag.encode()).hexdigest()[:12]
    
    # Avoid "File name too long" issues (Linux limit is 255 chars)
    p = f"{prefix}_" if prefix else ""
    if len(sorted_names) > 3:
        # e.g., train_multicell_32_16b961fcd1a4
        return f"{p}multicell_{len(sorted_names)}_{h}"
    
    # e.g., val_conditions_12Z_143B_16b961fcd1a4
    return f"{p}conditions_{'_'.join(sorted_names)}_{h}"


def save_cache(cache_dir, cache_name, sequences, expressions, targets, info):
    """Save processed dataset to an .npz file (uncompressed for maximum load speed)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, cache_name + '.npz')

    # info is a list of dicts at this stage
    chroms = np.array([d['chrom'] for d in info])
    starts = np.array([d['start'] for d in info], dtype=np.int64)
    ends = np.array([d['end'] for d in info], dtype=np.int64)
    gene_ids = np.array([d.get('gene_id', 'unknown') for d in info])

    np.savez(
        path,
        sequences=np.array(sequences, dtype=object),
        expressions=np.array(expressions, dtype=np.float32),
        targets=np.array(targets, dtype=np.float32),
        chroms=chroms,
        starts=starts,
        ends=ends,
        gene_ids=gene_ids,
    )
    print(f"  ✓ Cached dataset → {path}  ({len(sequences)} samples)")
    return path


def load_cache(cache_dir, cache_name):
    """Load cached dataset instantly via memory mapping."""
    path = os.path.join(cache_dir, cache_name + '.npz')
    if not os.path.exists(path):
        return None

    print(f"  ✓ Loading cached dataset ← {path}")
    data = np.load(path, allow_pickle=True, mmap_mode='r')

    # Return raw arrays instead of building a list of dictionaries (the old bottleneck)
    return {
        'sequences': data['sequences'],
        'expressions': data['expressions'],
        'targets': data['targets'],
        'chroms': data['chroms'],
        'starts': data['starts'],
        'ends': data['ends'],
        'gene_ids': data['gene_ids']
    }


# ---------------------------------------------------------------------------
# Condition discovery
# ---------------------------------------------------------------------------

def discover_conditions(data_dir):
    """Return sorted list of condition names that have both ATAC/ and RNA/."""
    if not os.path.isdir(data_dir):
        return []
    conditions = []
    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if not os.path.isdir(item_path):
            continue
        atac = os.path.join(item_path, 'ATAC')
        rna = os.path.join(item_path, 'RNA')
        if os.path.isdir(atac) and os.path.isdir(rna):
            conditions.append(item)
    return conditions


def load_condition_paths(condition_name, data_dir):
    """Return (atac_file, rna_file) or (None, None)."""
    cond = os.path.join(data_dir, condition_name)
    atac_dir = os.path.join(cond, 'ATAC')
    rna_dir = os.path.join(cond, 'RNA')

    atac_file = None
    if os.path.isdir(atac_dir):
        for f in sorted(os.listdir(atac_dir)):
            if f.endswith('.bed') or f.endswith('.narrowPeak'):
                atac_file = os.path.join(atac_dir, f)
                break

    rna_file = None
    if os.path.isdir(rna_dir):
        for f in sorted(os.listdir(rna_dir)):
            if f.endswith('.tsv'):
                rna_file = os.path.join(rna_dir, f)
                break

    if atac_file and rna_file:
        return atac_file, rna_file
    return None, None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class DataPreparationPipeline:
    """Prepare regression training data from ATAC + RNA + genome."""

    def __init__(self, fasta_path, atac_bed_path, rna_tsv_path,
                 gtf_path=None, seq_length=1000, seed=42,
                 signal_column=6, num_neighbors=5, 
                 neighbor_window=100000):

        """
        Args:
            fasta_path:      Path to genome FASTA (hg38.fa).
            atac_bed_path:   Path to ATAC narrowPeak / BED file.
            rna_tsv_path:    Path to RNA-seq expression matrix (gene × samples).
            gtf_path:        Path to GTF (optional, for gene→coord mapping).
            seq_length:      Window size in bp.
            seed:            Random seed.
            signal_column:   0-indexed column in BED for signal value
                             (default 6 = narrowPeak signalValue).
            num_neighbors:   Number of closest genes to use in expression vector.
            neighbor_window: Window size (one direction) to look for neighbors.
        """

        self.fasta_path = fasta_path
        self.atac_bed_path = atac_bed_path
        self.rna_tsv_path = rna_tsv_path
        self.gtf_path = gtf_path
        self.seq_length = seq_length
        self.seed = seed
        self.signal_column = signal_column
        self.num_neighbors = num_neighbors
        self.neighbor_window = neighbor_window

        np.random.seed(seed)

        # ── Load genome ────────────────────────────────────────────
        print(f"[1/4] Loading FASTA: {os.path.basename(fasta_path)}")
        self.fasta = Fasta(fasta_path)
        print(f"  ✓ {len(self.fasta.keys())} chromosomes")

        # ── Load ATAC peaks + signal ───────────────────────────────
        print(f"[2/4] Loading ATAC peaks: {os.path.basename(atac_bed_path)}")
        self.peaks_df = self._load_peaks(atac_bed_path, signal_column)
        print(f"  ✓ {len(self.peaks_df)} peaks  |  "
              f"signal range {self.peaks_df['signal'].min():.2f} – "
              f"{self.peaks_df['signal'].max():.2f}")

        # ── Load RNA-seq ───────────────────────────────────────────
        print(f"[3/4] Loading RNA-seq: {os.path.basename(rna_tsv_path)}")
        rna = pd.read_csv(rna_tsv_path, sep='\t', index_col=0)
        self.rna_df = rna.iloc[:, [0]].fillna(0).copy()
        print(f"  ✓ {self.rna_df.shape[0]} genes × {self.rna_df.shape[1]} samples")

        # ── Load GTF ───────────────────────────────────────────────
        print(f"[4/4] Loading GTF annotation …")
        self.gene_coords = {}
        self.genes_by_chrom = {} # Indexed by chrom for fast window lookups
        if gtf_path and os.path.exists(gtf_path):
            self._load_gtf(gtf_path)
        else:
            print("  ⚠ GTF not provided — will use mean expression as fallback")

        print("\n✓ Pipeline initialised\n")

    # ------------------------------------------------------------------ io
    @staticmethod
    def _load_peaks(path, signal_col):
        """Read narrowPeak / BED and extract signal column."""
        # Detect total columns
        with open(path) as fh:
            first = fh.readline()
        ncols = len(first.strip().split('\t'))

        col_names = [f'col{i}' for i in range(ncols)]
        col_names[0] = 'chrom'
        col_names[1] = 'start'
        col_names[2] = 'end'

        df = pd.read_csv(path, sep='\t', header=None, names=col_names,
                         dtype={'start': int, 'end': int})

        if signal_col >= ncols:
            raise ValueError(
                f"signal_column={signal_col} but file only has {ncols} columns"
            )
        df['signal'] = pd.to_numeric(df[col_names[signal_col]],
                                     errors='coerce').fillna(0).astype(float)
        return df[['chrom', 'start', 'end', 'signal']].copy()

    def _load_gtf(self, gtf_path):
        """Load gene coordinates and index them by chromosome."""
        try:
            gtf = pd.read_csv(
                gtf_path, sep='\t', header=None, comment='#',
                usecols=[0, 2, 3, 4, 6, 8],
                names=['chrom', 'feature', 'start', 'end', 'strand', 'info'],
                dtype={'start': int, 'end': int},
            )
            # Keep only gene-level rows for speed
            gtf = gtf[gtf['feature'] == 'gene'].copy()
            gtf['gene_name'] = gtf['info'].str.extract(r'gene_name "([^"]+)"')
            valid = set(self.rna_df.index)
            gtf = gtf[gtf['gene_name'].isin(valid)]

            for _, row in gtf.iterrows():
                gn = row['gene_name']
                chrom = row['chrom']
                
                if gn in self.rna_df.index:
                    tpm_val = self.rna_df.loc[gn].values.astype(np.float32)
                    coords = {
                        'chrom': chrom,
                        'start': row['start'],
                        'end': row['end'],
                        'strand': row['strand'],
                        'gene_name': gn,
                        'tpm': tpm_val
                    }
                    self.gene_coords[gn] = coords
                    self.genes_by_chrom.setdefault(chrom, []).append(coords)
                
            print(f"  ✓ Mapped {len(self.gene_coords)} genes with expression data from GTF")
        except Exception as e:
            print(f"  ⚠ GTF parsing error: {e}")

    # -------------------------------------------------------- sequence I/O
    def extract_sequence(self, chrom, start, end):
        """Extract and validate sequence from genome."""
        if chrom not in self.fasta:
            return None
        if end - start != self.seq_length:
            return None
        try:
            seq = str(self.fasta[chrom][start:end]).upper()
        except Exception:
            return None
        if len(seq) != self.seq_length:
            return None
        if seq.count('N') / len(seq) > 0.2:
            return None
        return seq

    # --------------------------------------------------- expression lookup
    def get_expression(self, chrom, start, end, num_neighbors=5, max_distance=100000):
        """
        Return a vector of expressions for the N nearest genes within max_distance.
        Sorted by distance to TSS (nearest first).
        """
        if not hasattr(self, 'genes_by_chrom') or not self.genes_by_chrom or chrom not in self.genes_by_chrom:
            # Fallback: global mean padding (repeated N times)
            mean_val = self.rna_df.mean(axis=0).values.astype(np.float32)
            return np.tile(mean_val, (num_neighbors,))

        window_center = (start + end) // 2
        w_start = window_center - max_distance
        w_end = window_center + max_distance

        matches = []
        for g in self.genes_by_chrom[chrom]:
            # Overlap check with the 200kb search window
            if max(g['start'], w_start) <= min(g['end'], w_end):
                tss = g['start'] if g['strand'] == '+' else g['end']
                dist = abs(window_center - tss)
                matches.append((dist, g['tpm']))

        # Sort by distance
        matches.sort(key=lambda x: x[0])
        
        # Take top N
        neighbor_exprs = [m[1] for m in matches[:num_neighbors]]
        
        # Padding
        if len(neighbor_exprs) < num_neighbors:
            mean_val = self.rna_df.mean(axis=0).values.astype(np.float32)
            # Pad with 0.0 or global mean? User said "if not many are expressing then it must be closed"
            # so padding with 0.0 is probably more biologically accurate for "no genes present".
            pad_size = num_neighbors - len(neighbor_exprs)
            neighbor_exprs.extend([np.zeros_like(mean_val) for _ in range(pad_size)])
            
        # Standardize return type as a flat vector
        return np.concatenate(neighbor_exprs).astype(np.float32)


    # ----------------------------------------------------- data generation
    def generate_positive_examples(self):
        """Extract a single centered 1000bp window per peak → target = log2(signal + 1)."""
        print(f"Generating positive examples ({self.seq_length}bp windows, centered) …")
        sequences, expressions, targets, info = [], [], [], []
        
        # Diagnostic counters
        n_total = len(self.peaks_df)
        n_missing_chrom = 0
        n_boundary_fail = 0
        n_too_many_n = 0
        n_io_error = 0

        # Pre-calculate chromosome naming style
        fasta_keys = set(self.fasta.keys())
        has_chr_prefix = any(k.startswith('chr') for k in fasta_keys if len(k) < 6)

        for _, row in tqdm(self.peaks_df.iterrows(),
                           total=n_total, desc='  peaks'):
            chrom = str(row['chrom'])
            # Normalize chrom names (e.g., 1 -> chr1)
            if has_chr_prefix and not chrom.startswith('chr'):
                chrom = 'chr' + chrom
            elif not has_chr_prefix and chrom.startswith('chr'):
                chrom = chrom[3:]

            if chrom not in fasta_keys:
                n_missing_chrom += 1
                continue

            pk_start, pk_end = int(row['start']), int(row['end'])
            raw_signal = float(row['signal'])
            target_val = float(np.log2(raw_signal + 1))

            center = (pk_start + pk_end) // 2
            s = center - (self.seq_length // 2)
            e = s + self.seq_length

            # Boundary correction
            if s < 0:
                s = 0
                e = self.seq_length
            
            clen = len(self.fasta[chrom])
            if e > clen:
                e = clen
                s = max(0, e - self.seq_length)

            # Verification of window integrity
            if e - s != self.seq_length:
                n_boundary_fail += 1
                continue

            try:
                # Extract sequence
                seq = str(self.fasta[chrom][s:e]).upper()
                if len(seq) != self.seq_length:
                    n_boundary_fail += 1
                    continue
                
                # N-mask filter (80% non-N required)
                if seq.count('N') / len(seq) > 0.2:
                    n_too_many_n += 1
                    continue
                    
                expr = self.get_expression(chrom, s, e, 
                                          num_neighbors=self.num_neighbors, 
                                          max_distance=self.neighbor_window)
                sequences.append(seq)
                expressions.append(expr)
                targets.append(target_val)
                info.append({'chrom': chrom, 'start': s, 'end': e,
                             'gene_id': 'peak'})
            except Exception:
                n_io_error += 1
                continue

        print(f"  ✓ {len(sequences)} positive examples extracted")
        if len(sequences) < n_total * 0.1:
            print(f"  ⚠ Alert: High Data Loss Detected!")
            print(f"    - Missing Chroms:  {n_missing_chrom}")
            print(f"    - Boundary Issues: {n_boundary_fail}")
            print(f"    - Too many 'N's:   {n_too_many_n}")
            print(f"    - I/O Errors:      {n_io_error}")
        
        return sequences, expressions, targets, info

    def generate_negative_examples(self, n_negatives, exclude_distance=200):
        """Random non-peak windows → target = 0.0."""
        print(f"Generating {n_negatives} negative examples …")
        sequences, expressions, targets, info = [], [], [], []

        # Build exclusion zones
        peak_ranges = {}
        for _, row in self.peaks_df.iterrows():
            chrom = row['chrom']
            s = max(0, int(row['start']) - exclude_distance)
            e = int(row['end']) + exclude_distance
            peak_ranges.setdefault(chrom, []).append((s, e))

        for chrom in peak_ranges:
            peak_ranges[chrom] = self._merge_intervals(peak_ranges[chrom])

        chroms = [c for c in self.fasta.keys() if c in peak_ranges]
        if not chroms:
            chroms = list(self.fasta.keys())

        attempts, max_attempts = 0, n_negatives * 20
        pbar = tqdm(total=n_negatives, desc='  negatives')

        while len(sequences) < n_negatives and attempts < max_attempts:
            attempts += 1
            chrom = np.random.choice(chroms)
            clen = len(self.fasta[chrom])
            s = np.random.randint(0, max(1, clen - self.seq_length))
            e = s + self.seq_length
            if e > clen:
                continue

            # Check exclusion
            if chrom in peak_ranges:
                hit = False
                for ps, pe in peak_ranges[chrom]:
                    if s < pe and e > ps:
                        hit = True
                        break
                if hit:
                    continue

            seq = self.extract_sequence(chrom, s, e)
            if seq is None:
                continue

            expr = self.get_expression(chrom, s, e, 
                                       num_neighbors=self.num_neighbors, 
                                       max_distance=self.neighbor_window)
            sequences.append(seq)
            expressions.append(expr)
            targets.append(0.0)
            info.append({'chrom': chrom, 'start': s, 'end': e,
                         'gene_id': 'non_peak'})
            pbar.update(1)

        pbar.close()
        print(f"  ✓ {len(sequences)} negative examples ({attempts} attempts)")
        return sequences, expressions, targets, info

    @staticmethod
    def _merge_intervals(intervals):
        if not intervals:
            return []
        intervals = sorted(intervals)
        merged = [intervals[0]]
        for cur in intervals[1:]:
            if cur[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], cur[1]))
            else:
                merged.append(cur)
        return merged

    # --------------------------------------------------------- full pipeline
    def prepare_data(self, test_chroms=('chr8', 'chr9'), val_frac=0.15):
        """Complete pipeline: positives + negatives → chromosome-split."""
        print("\n" + "=" * 60)
        print("DATA PREPARATION PIPELINE (regression)")
        print("=" * 60)

        pos_seq, pos_expr, pos_tgt, pos_info = \
            self.generate_positive_examples()

        n_neg = len(pos_seq)
        neg_seq, neg_expr, neg_tgt, neg_info = \
            self.generate_negative_examples(n_neg)

        all_seq = pos_seq + neg_seq
        all_expr = np.array(pos_expr + neg_expr, dtype=np.float32)
        all_tgt = np.array(pos_tgt + neg_tgt, dtype=np.float32)
        
        # Convert list of dicts to arrays for vectorized filtering
        all_chroms = np.array([d['chrom'] for d in pos_info + neg_info])
        all_starts = np.array([d['start'] for d in pos_info + neg_info])
        all_ends = np.array([d['end'] for d in pos_info + neg_info])
        all_gene_ids = np.array([d.get('gene_id', 'unknown') for d in pos_info + neg_info])

        return self._finalize_dataset(all_seq, all_expr, all_tgt, 
                                      all_chroms, all_starts, all_ends, all_gene_ids,
                                      test_chroms, val_frac)

    def _finalize_dataset(self, all_seq, all_expr, all_tgt, 
                          all_chroms, all_starts, all_ends, all_gene_ids,
                          test_chroms, val_frac):
        """Common logic to split and format the dataset."""
        test_mask = np.isin(all_chroms, test_chroms)
        train_mask = ~test_mask

        def subset(mask):
            idx = np.where(mask)[0]
            s = [all_seq[i] for i in idx]
            e = all_expr[idx]
            t = all_tgt[idx]
            # Build info list only for the final subsets (much smaller)
            i_list = []
            for i in idx:
                i_list.append({'chrom': all_chroms[i], 'start': all_starts[i], 
                               'end': all_ends[i], 'gene_id': all_gene_ids[i]})
            return s, e, t, i_list

        te_s, te_e, te_t, te_i = subset(test_mask)
        full_tr_s, full_tr_e, full_tr_t, full_tr_i = subset(train_mask)

        # Split train into train/val
        n = len(full_tr_s)
        n_val = max(1, int(n * val_frac))
        idx = np.random.permutation(n)
        val_idx = set(idx[:n_val].tolist())

        tr_s, tr_e, tr_t, tr_i = [], [], [], []
        val_s, val_e, val_t, val_i = [], [], [], []

        for i in range(n):
            if i in val_idx:
                val_s.append(full_tr_s[i]); val_e.append(full_tr_e[i]); val_t.append(full_tr_t[i]); val_i.append(full_tr_i[i])
            else:
                tr_s.append(full_tr_s[i]); tr_e.append(full_tr_e[i]); tr_t.append(full_tr_t[i]); tr_i.append(full_tr_i[i])

        print(f"\nSplits:  train={len(tr_s)}  val={len(val_s)}  test={len(te_s)}")
        print("=" * 60 + "\n")

        return {
            'train': (tr_s, tr_e, tr_t, tr_i),
            'val':   (val_s, val_e, val_t, val_i),
            'test':  (te_s, te_e, te_t, te_i),
        }
