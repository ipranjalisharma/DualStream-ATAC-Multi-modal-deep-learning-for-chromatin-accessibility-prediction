#  DuATTAC (ATAC Signal Predictor): Multi-Gene Context Architecture

Welcome to the ultimate guide for the ATAC Signal Predictor project. This manual provides a 360-degree view of the system, from hardware setup and biological theory to mathematical proofs and neural architecture.

---

##   System Installation & Dependencies

To ensure maximum performance and reproducibility, follow the detailed setup below.

### Hardware & OS Requirements
- **OS**: Linux (tested on Ubuntu 20.04/22.04).
- **GPU (Recommended)**: NVIDIA GPU with 8GB+ VRAM (for batch sizes like 1024).
- **RAM**: 16GB+ (primarily for genome indexing during the first run).
- **Disk Space**: ~10GB for reference files + cache.

### Step 1: Environment Setup
We recommend using **Miniconda** to manage your bioinformatics environment.

```bash
# Create a dedicated environment
conda create -n atac_env python=3.10 -y
conda activate atac_env
```

### Step 2: Core Dependencies Installation
Install the heavy-weight libraries required for genomics and deep learning.

```bash
# 1. Install PyTorch (GPU Support)
pip install -r requirements.txt
```

###  Dependency Deep-Dive: Why These Libraries?
| Library | Purpose in this Model |
| :--- | :--- |
| **`pyfaidx`** | **Genome Random Access**: Allows the model to pull any 1000bp DNA window from the giant 3GB `hg38.fa` file in microseconds without loading the whole file into RAM. |
| **`mappy`** | **Alignment Verification**: Used in `predict.py` to check if a custom sequence exists in the reference genome, ensuring your predictions are biologically valid. |
| **`pandas`** | **Metadata Orchestration**: Handles the complex mapping between BED peaks, RNA TSVs, and GTF gene coordinates. |

---

##   Biological Motivation: The Regulatory Code

### Chromatin Accessibility
In the nucleus of every human cell, 2 meters of DNA is packed into a tiny space. To control gene expression, the cell "opens" certain parts of this DNA to allow proteins called **Transcription Factors (TFs)** to bind. This state is known as **Chromatin Accessibility**.

### The Task: Continuous Signal Regression
While most models treat accessibility as a binary state (Open vs Closed), biological reality is a continuous gradient. This model uses **ATAC-seq (Assay for Transposase-Accessible Chromatin using sequencing)** signal intensity as its ground truth.
- **Input**: A 1000bp window of DNA + multi-gene RNA expression context.
- **Output**: A continuous scalar value representing the **log₂ fold-enrichment** of accessibility.

By regression-modeling the signal, we can distinguish between "Strong Enhancers" and "Weak Regulatory Regions," which is critical for understanding complex disease mutations.

---

##   Data Architecture: The Raw Materials

The pipeline is designed to handle standard bioinformatics formats with zero manual preprocessing.

### A. ATAC-seq (narrowPeak)
We use the **signalValue (Column 7)** from narrowPeak files.
- **Format**: `chrom | start | end | peak_id | score | strand | signalValue | pValue | qValue | peakCenter`
- **Logic**: We extract the midpoint of these peaks to ensure the model sees the peak of the accessibility curve.

### B. RNA-seq (Expression TSV)
Cell-type-specific mRNA counts.
- **Rows**: Gene Names (e.g., *MYC*, *GAPDH*).
- **Columns**: Expression values (TPM or FPKM).

### C. Reference Genome (FASTA)
The model currently uses the **hg38 (GRCh38)** human reference. It utilizes `pyfaidx` for fast, random-access DNA extraction, allowing us to fetch any 1000bp window in microseconds.
- **Download**: [UCSC hg38.fa.gz](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz)
- **Annotation**: [GENCODE v45 (ALL)](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.annotation.gtf.gz)

---

##   RNA Neighborhood: Top-N Multi-Gene Context

A major innovation in this model is a **multi-dimensional context vector**.

### The Spatial Problem
Regulatory elements like enhancers are influenced by the transcriptional activity of several nearby genes. A single weighted sum loses the information about *which* or *how many* genes are active.

### The Context Vector Solution
The expression of the **Top-5 nearest genes** within a ±100kb window is extracted as a feature vector <i>V<sub>expr</sub></i>:

```text
V_expr = [ TPM(g1), TPM(g2), TPM(g3), TPM(g4), TPM(g5) ]
```

Where *g_i* satisfies:
1. *dist*(window, *g_i*) <= 100kb
2. *dist*(window, *g_i*) <= *dist*(window, *g_i+1*) (Sorted by proximity)

**Padding**: Isolated regions with fewer than 5 genes are padded with **0.0**, signaling low regulatory density to the model.

---

##   Neural Architecture Deep Dive: ResNet-SE 

The architecture is a **Dual-Stream Neural Network** composed of approximately **224,000** trainable parameters.

### A. The DNA Stream (1D ResNet-SE)
Processes the 1000bp DNA sequence to identify regulatory motifs.
- **Input Layer**: Accepts a 5 x 1000 one-hot encoded matrix (A, C, G, T, N).
- **Five Residual blocks**:
    - **Residual Path**: y = F(x) + x. Each block learns the "residual" change, allowing deep gradients to flow without vanishing.
    - **Squeeze-and-Excitation (SE) Attention**: Reduces the spatial map to a channel vector, then excites only the motif-relevant channels.
- **Global Pooling**: Reduces the spatial dimension to a flat feature vector.

### B. The RNA Stream (Neighborhood context)
- **MLP Encoder**: Passes the **5-dimensional context vector** through multiple Dense layers to map it into the same feature space as the DNA motifs.

---

##   Mathematical Foundations

### A. Signal Compression & Normalization

```text
y = log2(SignalValue + 1)
```

This squashes the massive range of fold-enrichment (0 to 1000+) into a manageable range (0 to 10).

### B. Target Standardization (Z-Score)
For optimized training, the targets are normalized:

```text
Z = (y - μ) / σ
```

During inference, the model automatically performs the inverse transform.

### D. Balanced Training Strategy
The model is trained on a strictly balanced dataset (1:1 ratio):
- **Positive Examples**: Genomic regions centered on established ATAC-seq peaks.
- **Negative Examples**: Randomly sampled non-peak regions from the same cell type (excluding all known peak intervals), ensuring only high-confidence "closed" regions are used as negatives.

### C. Huber Loss Piecewise Analysis
We use Huber Loss to balance precision and stability, defined as:

```text
If |y - ŷ| <= δ:
    Loss = 0.5 * (y - ŷ)²

Otherwise:
    Loss = δ * (|y - ŷ| - 0.5 * δ)
```

Where δ = 1.0 (default). This provides:
- **Quadratic behavior** for small errors (MSE-like precision).
- **Linear behavior** for large errors (MAE-like robustness to outliers).

---

##   High-Speed Vectorized Pipeline

### MD5-Based Caching
We hash `sorted(cell_names) + seed`. If you change a cell name or seed, the model rebuilds the data. Cache hits take ~0.15s.

### Memory Mapping (mmap)
We save caching files as **Uncompressed NumPy (.npz)**. Uncompressed data with `mmap_mode='r'` allows the OS to read data straight into RAM instantly.

---

##   Metric Interpretation & Biology

- **Pearson Correlation ($r$)**: Measures trend accuracy. **0.80+** is typical for stable models.
- **R² Score**: Measures absolute accuracy. Expect **0.55 - 0.75** (achieved **0.63** on held-out cellular conditions).

### Early Stopping & Convergence
Training is governed by an Early Stopping callback (`patience=10`, `min_delta=1e-4`). The model typically converges around epoch 35-45.

---

##   Output Metadata & File Structure

### The `norm_stats.npz` File
**REQUIRED FOR PREDICTION.** Stores the $\mu$ and $\sigma$ from training so `predict.py` can return real signal values.

---

##   Troubleshooting & FAQ

- **Memory Error**: Reduce `--batch-size` to `512` or `256`.
- **Cache Mismatch**: Run `rm cache/*.npz` before re-training.
- **Missing Stats**: `predict.py` requires the `norm_stats.npz` file.

---

##   Scenario-Based Execution Guide

**Scenario A: Universal Full-Scale Model Training**
```bash
python train.py --conditions all --epochs 100 --batch-size 1024 --output-dir final_v3_production
```

**Scenario B: Rapid Hyperparameter Tuning**
```bash
python train.py --conditions 0,1 --epochs 10 --batch-size 512 --lr 0.0001 --output_dir tuning_test
```

**Scenario C: Generalization Study (Cross-Cell-Type)**
Evaluate the model on an entirely new condition not present in the training pool.
```bash
python test.py --model-path final_production/checkpoints/best_model.pth --conditions 5 --test-frac-cond 0.3
```
---

##   Master Utility Reference: All CLI Flags

### `train.py` (Model Optimization)
| Argument | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--conditions` | `str` | *(REQ)* | Condition indices or `"all"`. |
| `--data-dir` | `str` | `config.PATHS['data_dir']` | Root data directory. |
| `--fasta` | `str` | `config.PATHS['fasta']` | Reference genome FASTA. |
| `--gtf` | `str` | `config.PATHS['gtf']` | GTF annotation file. |
| `--output-dir` | `str` | `output` | Folder for checkpoints/logs. |
| `--cache-dir` | `str` | `config.PATHS['cache_dir']` | Folder for `.npz` caches. |
| `--cpu` | `flag` | `False` | Disables GPU acceleration. |
| `--batch-size` | `int` | `32` | Number of samples per step. |
| `--epochs` | `int` | `100` | Max training cycles. |
| `--learning-rate` | `float` | `0.001` | Adam learning rate. |

###  `test.py` (Evaluation & Metrics)
| Argument | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--model-path` | `str` | *(REQ)* | Path to trained model (`.pth`). |
| `--conditions` | `str` | *(REQ)* | Cell type indices to evaluate. |
| `--output-dir` | `str` | `test_results` | Folder for metrics and exports. |
| `--batch-size` | `int` | `64` | Inference batch size. |

###  `predict.py` (Signal Inference)
| Argument | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--sequence` | `str` | *(XOR)* | Single DNA sequence string. |
| `--fasta-input` | `str` | *(XOR)* | FASTA file of regions to score. |
| `--model-path` | `str` | *(REQ)* | Path to trained model checkpoint. |
| `--rna-tsv` | `str` | `None` | Expression TSV for neighborhood lookup. |
| `--output` | `str` | `None` | Path to save scores (TSV). |

---

##  Performance on Human cell lines (over 400 conditions)

<img width="3600" height="2100" alt="all_distribution_v_style" src="https://github.com/user-attachments/assets/2a126425-5373-4677-9224-a686a82c25fd" />
<img width="1200" height="900" alt="roc_curve" src="https://github.com/user-attachments/assets/378a0151-0b56-44ba-91c7-a12db01a9738" />
<img width="1200" height="900" alt="pr_curve" src="https://github.com/user-attachments/assets/f55892c2-b65c-40a9-8596-7b9af60b6f8e" />
<img width="1200" height="900" alt="confusion_matrix" src="https://github.com/user-attachments/assets/5fb839c3-f646-499c-8900-1ea7f02df560" />

**END OF MANUAL**
