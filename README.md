# WinnowNet

WinnowNet is a deep-learning PSM filtering workflow for proteomic stable isotope probing data. This repository currently provides a production `pure_cnn_pct` prediction wrapper for Sipros5-style raw inputs and training utilities for the current 7-channel CNN feature schema.

## Setup

WinnowNet is developed and tested on Linux. Create the micromamba environment used by the scripts:

```bash
micromamba create -n winnownet python=3.12 pytorch=2.5.1 numpy pandas scikit-learn einops matplotlib
micromamba activate winnownet
```

All examples below use:

```bash
micromamba run -n winnownet <command>
```

## Input Files

Use [Sipros5](https://github.com/thepanlab/Sipros5) to generate the raw WinnowNet input files:

- `*_filtered_psms.tsv`
- `*.FT1`
- `*.FT2`

For each run, keep the files in the same directory with matching stems. For example:

```text
sample_01_filtered_psms.tsv
sample_01.FT1
sample_01.FT2
```

The TSV must include PSM identifiers (`PSMId` or `SpecId`) and peptide sequences. SIP abundance columns such as `MS1IsotopicAbundances` and `MS2IsotopicAbundances` are used when present.

## Prediction

Run production prediction directly on the Sipros5 output directory or on a comma-separated list of `*_filtered_psms.tsv` files. `script/winnownet.py` extracts features, scores PSMs, applies the production threshold logic, and writes accepted target PSMs.

The production wrapper uses 256 matched peaks per PSM.

```bash
micromamba run -n winnownet python script/winnownet.py \
  --target data/sample_01 \
  --decoy data/sample_02 \
  -o results/accepted_target_psms.tsv \
  --decoy-output results/accepted_decoy_psms.tsv
```

In these examples, `data/sample_01` is the target sample and `data/sample_02` is the unlabeled control/decoy sample.

The default model is:

```text
model/pure_cnn_all_pct.pt
```

Override it with `--model`:

```bash
micromamba run -n winnownet python script/winnownet.py \
  --target data/sample_01 \
  -o results/accepted_target_psms.tsv \
  --model model/pure_cnn_all_pct.pt
```

Useful prediction options:

- `--target`: Target `*_filtered_psms.tsv` file, directory, glob, or comma-separated list.
- `--decoy`: Optional unlabeled control/decoy input. When provided, the score threshold is estimated at 1% target-decoy FDR. If no unlabeled control samples are provided, WinnowNet automatically uses the pretrained model's score-threshold calibration.
- `--decoy-output`: Optional accepted decoy audit TSV; requires `--decoy`.
- `--config`: SIP configuration file. Default: `script/SIP.cfg`.
- `--sip-atom-abundance`: Optional natural isotope abundance override for no-decoy prediction, for example `C13=1.07`.
- `--device`: `auto`, `cpu`, `cuda`, or `cuda:N`. Default: `auto`.
- `--jobs`, `--cores`, `--threads-per-job`: Control parallel file scoring and CPU use.
- `--batch-size`: CNN scoring batch size. Default: `1024`.

## Training The Current CNN Model

The current production model architecture is `pure_cnn_pct`. It expects 7-channel CNN features with schema `cnn_7ch_v1` and 256 matched peaks per PSM; old 3-channel CNN checkpoints and features are not compatible.

First generate CNN feature pickles from Sipros5 outputs. Directory/list mode automatically pairs each `*_filtered_psms.tsv` file with matching `.FT1` and `.FT2` files and writes one `.pkl` beside each TSV.

```bash
micromamba run -n winnownet python script/SpectraFeatures.py \
  -i data/sample_01,data/sample_02 \
  -f cnn \
  -c script/SIP.cfg \
  -t 8 \
  -j 4 \
  --max-peaks 256
```

For a single run:

```bash
micromamba run -n winnownet python script/SpectraFeatures.py \
  -i data/sample_01_filtered_psms.tsv \
  -1 data/sample_01.FT1 \
  -2 data/sample_01.FT2 \
  -o data/sample_01.pkl \
  -f cnn \
  -c script/SIP.cfg \
  -t 8 \
  --max-peaks 256
```

Then train `pure_cnn_pct` with target and decoy feature pickles. The `--target-pct` and `--decoy-pct` values must align with the target and decoy inputs.

```bash
micromamba run -n winnownet python script/WinnowNet_CNN.py \
  --target data/sample_01 \
  --decoy data/sample_02 \
  --target-pct 5 \
  --decoy-pct 1 \
  -m model/pure_cnn_all_pct.pt \
  --model-arch pure_cnn_pct \
  --epochs 50 \
  --learning-rate 2e-4 \
  --train-batch-size 1024 \
  --eval-batch-size 1024 \
  --class-weight none \
  --pct-loss-weight 2.0 \
  --exclude-protein-prefix Con_
```

Training writes the best model to the path passed with `-m` and per-epoch checkpoints to a sibling directory named after the model, for example `pure_cnn_all_pct_checkpoints/`.

Optional filters:

- `--target-exclude`: Per-target-input strict lower bound for `MS2IsotopicAbundances`; use `0` to disable for an input.
- `--decoy-exclude`: Per-decoy-input strict upper bound for `MS2IsotopicAbundances`; use `0` to disable for an input.
- `-p`: Load a compatible pretrained 7-channel checkpoint before training.

For a grid search over `pure_cnn_pct` hyperparameters:

```bash
micromamba run -n winnownet python test/grid_search_pure_cnn_pct.py
```

## References

1. Feng, S., Zhang, B., Wang, H., Xiong, Y., Tian, A., Yuan, X., Pan, C., and Guo, X. [Enhancing peptide identification in metaproteomics through curriculum learning in deep learning](https://www.nature.com/articles/s41467-025-63977-z). *Nature Communications* 16, 8934 (2025). https://doi.org/10.1038/s41467-025-63977-z
2. Xiong, Y., Mueller, R. S., Feng, S., Guo, X., and Pan, C. [Proteomic stable isotope probing with an upgraded Sipros algorithm for improved identification and quantification of isotopically labeled proteins](https://link.springer.com/article/10.1186/s40168-024-01866-1). *Microbiome* 12, 148 (2024). https://doi.org/10.1186/s40168-024-01866-1
3. [Sipros5](https://github.com/thepanlab/Sipros5) is used to generate the `*_filtered_psms.tsv`, `.FT1`, and `.FT2` inputs consumed by WinnowNet.
