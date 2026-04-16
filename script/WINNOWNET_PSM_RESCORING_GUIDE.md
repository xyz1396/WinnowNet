# WinnowNet PSM Rescoring Guide

## Purpose

This guide tracks the current PSM rescoring workflow in this repository. The production path is now:

1. Start from Sipros5-style raw inputs:
   - `*_filtered_psms.tsv`
   - matching `*.FT1`
   - matching `*.FT2`
2. Predict accepted PSMs with `script/winnownet.py`.
3. Train the current CNN checkpoint with `script/SpectraFeatures.py` and `script/WinnowNet_CNN.py` when retraining is needed.

The old pickle-first prediction scripts are not the production path. Use `script/winnownet.py` for current inference.

## Production Prediction

Current entry point:

```text
script/winnownet.py
```

The wrapper:

- accepts target `*_filtered_psms.tsv` files, directories, globs, or comma-separated lists
- optionally accepts unlabeled control/decoy inputs with `--decoy`
- auto-pairs each TSV with the matching `.FT1` and `.FT2` files by stem
- runs `script/sipros_theoretical_spectra` internally
- extracts the 7-channel CNN features internally
- scores with a `pure_cnn_pct` checkpoint
- writes accepted target PSMs to `--output`
- optionally writes accepted decoy/control PSMs to `--decoy-output`

Production constants in `script/winnownet.py`:

- default model: `model/pure_cnn_all_pct.pt`
- default config: `script/SIP.cfg`
- model architecture: `pure_cnn_pct`
- feature schema: `cnn_7ch_v1`
- input channels: `7`
- matched peaks per PSM: `256`
- default batch size: `1024`
- default match tolerance: `10` ppm
- default MS1 window: `10.0` m/z
- default target protein prefixes excluded before prediction: `Decoy_,Con_`

Example with target plus unlabeled control/decoy:

```bash
micromamba run -n winnownet python script/winnownet.py \
  --target data/sample_01 \
  --decoy data/sample_02 \
  -o results/accepted_target_psms.tsv \
  --decoy-output results/accepted_decoy_psms.tsv
```

In this example, `data/sample_01` is the target sample and `data/sample_02` is the unlabeled control/decoy sample.

Example without unlabeled control/decoy:

```bash
micromamba run -n winnownet python script/winnownet.py \
  --target data/sample_01 \
  -o results/accepted_target_psms.tsv
```

Threshold behavior:

- If `--decoy` is provided, `winnownet.py` estimates the threshold from target/control scores at 1% target-decoy FDR.
- If no unlabeled control samples are provided, it uses the pretrained checkpoint's score-threshold calibration. It chooses the closest stored percent-label threshold from checkpoint metadata using the target median `MS2IsotopicAbundances` above natural abundance, then falls back to `best_decision_threshold` when no percent-specific threshold is available.

Important prediction options:

- `--target`: required target input file, directory, glob, or comma-separated list.
- `--decoy`: optional unlabeled control/decoy input file, directory, glob, or comma-separated list.
- `--decoy-output`: optional accepted decoy/control output TSV; requires `--decoy`.
- `--model`: `pure_cnn_pct` checkpoint path. Default: `model/pure_cnn_all_pct.pt`.
- `--config`: SIP config file. Default: `script/SIP.cfg`.
- `--sip-atom-abundance`: optional no-decoy natural isotope abundance override, for example `C13=1.07`.
- `--target-exclude-protein-prefixes`: comma-separated prefixes excluded from target prediction when all proteins in a row match a listed prefix.
- `--device`: `auto`, `cpu`, `cuda`, or `cuda:N`.
- `--jobs`: maximum file jobs to run in parallel.
- `--cores`: maximum total CPU cores to use.
- `--threads-per-job` or `--cores-per-job`: CPU cores used inside each file job.
- `--batch-size`: CNN scoring batch size.
- `--ms1-window`: MS1 isolation window.
- `--ppm`: peak match tolerance in ppm.

## Input Naming

For each run, keep the Sipros5 outputs together with matching stems:

```text
data/sample_01_filtered_psms.tsv
data/sample_01.FT1
data/sample_01.FT2
```

For directory inputs, `winnownet.py` recursively finds files ending in `_filtered_psms.tsv`.

For TSV inputs, the paired raw files are resolved as:

```text
<stem>.FT1
<stem>.FT2
```

For example, `data/sample_01_filtered_psms.tsv` resolves to `data/sample_01.FT1` and `data/sample_01.FT2`.

## Output Columns

`winnownet.py` preserves the original TSV columns where possible and writes rescored accepted rows sorted by score. It updates or appends:

- `score`
- `q-value` or `qvalue`
- `Label`

When `MS2IsotopicAbundances` exists, it inserts:

```text
MS2isotopicAbundanceEvolopeMedian
```

`posterior_error_prob` is dropped from the output if present.

## Current CNN Feature Tensor

The current CNN model input is one tensor per PSM:

```text
[xFeatures]
```

`xFeatures` has shape:

```text
[7, 256]
```

The 7 channels are:

```text
[expmz, delta_mz, mono_mz, experimental_intensity, theoretical_intensity, SIPatomNumber, EnrichRatio]
```

Feature construction:

- matches MS1 peaks to theoretical precursor peaks
- trims theoretical fragment peaks to the observed MS2 m/z range
- matches MS2 peaks to trimmed theoretical fragment peaks
- preserves precursor and b/y isotope-envelope metadata
- selects the top 256 matched peaks by raw experimental intensity
- sorts the retained matched peaks by experimental m/z (`expmz`) before tensor output
- pads with zero-valued peak rows if fewer than 256 matched peaks are retained
- normalizes real and theoretical intensities by precursor, fragment, or isotope-envelope grouping
- repeats one `EnrichRatio` value across all matched peaks in the same isotope envelope

## Production Model Structure

The production model architecture is `pure_cnn_pct`. It is implemented as `PureCNNPct` in `script/winnownet.py` for inference and as `PureCNN_pct` in `script/WinnowNet_CNN.py` for training.

Input:

```text
[batch, 7, 256]
```

The 256 peak positions are the retained top-intensity matched peaks in increasing `expmz` order.
Therefore, local `Conv1d(kernel_size=3)` neighborhoods are neighboring retained peaks by experimental m/z.

Convolutional feature extractor:

```text
Conv1d(7, 32, kernel_size=3, padding=1)
BatchNorm1d(32)
ReLU
Conv1d(32, 64, kernel_size=3, padding=1)
BatchNorm1d(64)
ReLU
Conv1d(64, 128, kernel_size=3, padding=1)
BatchNorm1d(128)
ReLU
```

The convolution stack preserves the 256-peak width and produces:

```text
[batch, 128, 256]
```

Global pooling:

```text
x_max = max over peak dimension    -> [batch, 128]
x_mean = mean over peak dimension  -> [batch, 128]
concat(x_max, x_mean)              -> [batch, 256]
```

Shared dense layer:

```text
Linear(256, 128)
BatchNorm1d(128)
ReLU
Dropout(p=0.3)
```

Output heads:

```text
classifier:    Linear(128, 2)  -> target/decoy logits
pct_regressor: Linear(128, 1)  -> predicted log1p(percent label)
```

Forward output:

```text
(classifier_logits, pct_regression_output)
```

Prediction uses `softmax(classifier_logits)[:, 1]` as the target score. The percent-regression output is transformed with `expm1` and clamped to zero or above when reported internally.

## Training Loss

For `pure_cnn_pct`, the training loss combines target/decoy classification with percent-label regression:

```text
loss = CrossEntropyLoss(classifier_logits, target_label)
     + pct_loss_weight * SmoothL1Loss(pred_log_pct, log1p(pct_label))
```

Classification target:

```text
target_label = 1 for target PSMs
target_label = 0 for decoy/control PSMs
```

Percent regression target:

```text
expected_log_pct = log1p(pct_label)
```

The percent regressor predicts this log-scale value directly. During prediction, the model output is converted back to percent space with `expm1`.

Loss settings in current code:

- classification loss: `nn.CrossEntropyLoss`
- optional class weighting: `--class-weight balanced` computes weights from training target/decoy counts
- percent loss for `pure_cnn_pct`: `nn.SmoothL1Loss`
- default `--pct-loss-weight`: `0.5`
- bundled production checkpoint metadata uses `pct_loss_weight=2.0`

`pure_cnn_pct` requires pct labels for every batch item; missing pct labels raise an error during loss computation.

## Training Feature Pickles

Use `script/SpectraFeatures.py` only when generating `.pkl` feature files for training. Production prediction does not require prebuilt pickles because `script/winnownet.py` extracts features internally.

Current production training should generate 7-channel CNN features with 256 peaks:

```bash
micromamba run -n winnownet python script/SpectraFeatures.py \
  -i data/sample_01,data/sample_02 \
  -f cnn \
  -c script/SIP.cfg \
  -t 8 \
  -j 4 \
  --max-peaks 256
```

Single-file training feature generation:

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

Note: `script/SpectraFeatures.py` still has a legacy default of 128 peaks. Pass `--max-peaks 256` when generating training features for the current production `pure_cnn_pct` model.

Feature pickle schema:

- top-level dict contains `__meta__`
- `__meta__` stores source metadata such as columns, source file, mode, `feature_schema`, `input_channels`, and `max_peaks`
- each row entry stores `psm_id`, `model_input`, `label`, `label_raw`, `label_confidence`, `row_index`, and `row_values`
- CNN `model_input` is `[xFeatures]`

## Current CNN Training

Current production architecture:

```text
pure_cnn_pct
```

The trainer is:

```text
script/WinnowNet_CNN.py
```

`pure_cnn_pct` requires split-mode training with target and decoy/control feature pickles plus percent labels:

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

Training constants in current code:

- default epochs: `50`
- default train batch size: `1024`
- default eval batch size: `1024`
- default learning rate: `3e-4`
- supported CNN architectures: `tnet`, `pure_cnn`, `pure_cnn_pct`
- current production architecture: `pure_cnn_pct`
- feature schema: `cnn_7ch_v1`
- input channels: `7`
- default class weight mode: `none`
- default percent regression loss weight: `0.5`
- current bundled production checkpoint metadata uses `pct_loss_weight=2.0`

Optional training controls:

- `-p`: load a compatible pretrained 7-channel checkpoint before training.
- `--target-exclude`: per-target-input strict lower bound for `MS2IsotopicAbundances`; use `0` to disable for an input.
- `--decoy-exclude`: per-decoy-input strict upper bound for `MS2IsotopicAbundances`; use `0` to disable for an input.
- `--exclude-protein-prefix`: drop PSMs when all proteins start with one of the provided prefixes.
- `--class-weight`: `none` or `balanced`.
- `--pct-loss-weight`: SmoothL1 loss weight for log1p percent-label regression.

Training output:

- the best model is written to the path passed with `-m`
- per-epoch checkpoints are written to a sibling directory named `<model_stem>_checkpoints`

## Grid Search

The current grid-search launcher for production CNN training is:

```text
test/grid_search_pure_cnn_pct.py
```

Run:

```bash
micromamba run -n winnownet python test/grid_search_pure_cnn_pct.py
```

The launcher calls `script/WinnowNet_CNN.py` with `--model-arch pure_cnn_pct` and writes per-trial outputs under `data/grid_search_pure_cnn_pct` unless overridden.

## Legacy Notes

The attention model and old standalone prediction scripts still exist in the repository, but this guide is scoped to the current production `pure_cnn_pct` flow. Prefer `script/winnownet.py` for inference and `script/WinnowNet_CNN.py --model-arch pure_cnn_pct` for current CNN training.
