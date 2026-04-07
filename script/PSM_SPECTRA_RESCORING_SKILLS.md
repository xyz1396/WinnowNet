# WinnowNet PSM + Spectra Rescoring Skills Guide

## Scope
This document describes the current behavior of the rescoring pipeline in this repo after recent updates:
- `sipros_theoretical_spectra` integration
- new `theory.txt` block format
- FT1/FT2 parsing changes
- CNN/ATT feature generation and training usage

## Main pipeline
1. Generate spectrum feature pickle with:
- `script/SpectraFeatures.py` (single implementation used for training and inference)

2. Train rescoring model:
- Attention/Transformer: `script/WinnowNet_Att.py`
- CNN (PointNet-style): `script/WinnowNet_CNN.py`

3. Predict rescoring scores:
- Attention: `script/Prediction.py`
- CNN: `script/Prediction_CNN.py`

4. Downstream filtering/assembly:
- `script/filtering.py`, `script/filtering_shuffle.py`, `script/sipros_peptides_assembling.py`

## Dataset convention used now
Training scripts support either:
- embedded-label `.pkl` inputs passed with `-i`
- explicit split-mode inputs passed with `--target` and `--decoy`

Each input directory typically contains matched `.tsv` and `.pkl` pairs.
`script/WinnowNet_Att.py` and `script/WinnowNet_CNN.py` split records into train/val/test after loading.

## Feature-generation inputs
`script/SpectraFeatures.py` consumes:
- PSM table (`-i`, TSV)
- FT1 (`-1`)
- FT2 (`-2`)
- config file for `sipros_theoretical_spectra` (`-c`)

## Current CLI options (feature generation)
- `-i`: tab-delimited PSM file
- `-1`: FT1 file
- `-2`: FT2 file
- `-o`: output pickle file
- `-t`: number of threads
- `-f`: mode (`cnn` or `att`, default `att`)
- `-c`: Sipros config
- `-w`: MS1 isolation window size in m/z for ATT MS1 filtering (default `10.0`)
- `-d`: peak match tolerance `diffDa` in m/z (default `0.01`, HRMS-oriented)
- `-n`: CNN only, top N peaks per fragment envelope (default `3`)

## Experimental spectra parsing (`FTtoDict`)
`FTtoDict` is used for both FT1 and FT2.

Per scan, stored fields:
- `peaks`: list of `[mz, intensity]`, sorted by `mz`
- `parent_scan`: from `D ParentScanNumber` when present (used to map MS2->MS1)
- `isolation_center_mz__charge`
- `isolation_center_mz`
- `precursor_candidates` from `Z` optional pairs

FT2 `Z` parsing:
- first pair: selected charge and selected-charge * isolation-center-mz
- remaining tokens: optional `(candidate_charge, candidate_mz)` pairs (deduplicated)

Peak charge handling in numeric lines:
- read charge from column 6 when present
- if peak charge `> 1`, convert to monocharge m/z:
  - `mz1 = z * (mz - PROTON_MASS) + PROTON_MASS`
- stored peak still becomes `[mz1, intensity]` (no charge column kept)

## New theoretical file format (`theoryToDict`)
Now parsed from `sipros_theoretical_spectra` block format:
- `> <PSM id>`
- line1 precursor m/z values
- line2 precursor intensities
- line3 fragment m/z values
- line4 fragment intensities
- line5 fragment ion kinds (`b`/`y`)
- line6 fragment ion positions (1-based)

Comments starting with `#` are ignored.

Stored per PSM key:
- `precursor`: `[[mz, intensity], ...]`
- `fragment`: `[[mz, intensity], ...]`
- `all`: combined precursor+fragment
- `precursor_abundant_mz`: most intense precursor peak m/z
- `fragment_kinds`: list of `b`/`y`
- `fragment_positions`: list of positions

Note:
- `precursor_mono_mz` is no longer read from theory file.

## PSM tabular features (hand-crafted)
Built in `feature_dict()` from `read_tsv()` + `scan.get_features()`:
- `qvalue`
- `qn`
- `qnl`
- `mono_mass` (from peptide sequence):
  - `sum(AA_dict[aa]) + N_TERMINUS + C_TERMINUS`
- `mass_error` (from TSV `massErrors`)
- `isotopic_mass_window_shift` (from TSV `isotopicMassWindowShifts`)
- `peptide_length`
- `num_missed_cleavages`
- precursor charge one-hot `[z1, z2, z3plus]`

Important:
- `pep.theory_mass` is set from `mono_mass` and is not overwritten by theory precursor mass anymore.

## CNN feature construction (`IonExtract`)
Used when `-f cnn`.

Matching process:
1. Match MS1 peaks against theoretical precursor peaks.
2. Trim theoretical fragment peaks to the observed experimental MS2 m/z range.
3. Match MS2 peaks against the trimmed theoretical fragment peaks.
4. Fragment peaks are first filtered by envelope metadata:
- group by `(fragment_kind, fragment_position)`
- keep top `n` by theoretical intensity in each group (`-n`, default `3`)

Each match yields one triplet feature:
- `[delta_mz, experimental_intensity, theoretical_intensity]`

Post-processing:
- pad/truncate to `pairmaxlength=500`
- normalize precursor matches within the precursor group
- normalize fragment matches within each `(fragment_kind, fragment_position)` group
  for experimental and theoretical channels
- transpose to shape `[3, pairmaxlength]`

Output entry in feature pickle (CNN path):
- `[xFeatures, X_add_feature]`

Current CNN trainer usage (`script/WinnowNet_CNN.py`):
- uses only `D_features[j][0]` (`xFeatures`)
- ignores `X_add_feature`

## Attention feature construction (`IonExtract_Att`)
Used when `-f att`.

Process:
1. Filter MS1 peaks by user-defined isolation window `-w`, centered on most abundant theoretical precursor m/z.
2. Build `exp_all = filtered_ms1 + ms2`.
3. Trim theoretical fragment peaks to the observed experimental MS2 m/z range.
4. Keep precursor and fragment groups separate during normalization.
5. Normalize precursor intensities as one precursor group.
6. Normalize theoretical fragment intensities within each
   `(fragment_kind, fragment_position)` group.
7. Normalize experimental MS2 peaks as one fragment bucket.
8. Concatenate normalized precursor and fragment peaks into the final arrays.

Output entry in feature pickle (ATT path):
- `[Xexp, Xtheory]`

Current ATT trainer usage (`script/WinnowNet_Att.py`):
- directly consumes `[Xexp, Xtheory]`
- does not consume PSM tabular `X_add_feature`

## What is parsed but currently not used in model input
- `precursor_abundant_mz` (used for ATT MS1 window center only)
- `fragment_kinds` / `fragment_positions`:
  - used in CNN preprocessing for top-`n` envelope filtering
  - not directly fed as model tensors

## Output pickle schema summary
- `cnn` mode: `dict[PSMId] = [xFeatures, X_add_feature]`
- `att` mode: `dict[PSMId] = [Xexp, Xtheory]`

For non-CNN mode, keys are reordered to match `D_feature` key order before save.

## Example commands

### Generate ATT features
```bash
python script/SpectraFeatures.py \
  -i /path/target/sample_filtered_psms.tsv \
  -1 /path/target/sample.FT1 \
  -2 /path/target/sample.FT2 \
  -o /path/target/sample.pkl \
  -f att -t 8 -c script/SIP.cfg -d 0.01 -w 10
```

### Generate CNN features with envelope top-5
```bash
python script/SpectraFeatures.py \
  -i /path/target/sample_filtered_psms.tsv \
  -1 /path/target/sample.FT1 \
  -2 /path/target/sample.FT2 \
  -o /path/target/sample.pkl \
  -f cnn -t 8 -c script/SIP.cfg -d 0.01 -n 5
```

### Train ATT model using explicit target/decoy inputs
```bash
python script/WinnowNet_Att.py --target /path/target --decoy /path/decoy -m model/att.pt -p ""
```

### Train CNN model using explicit target/decoy inputs
```bash
python script/WinnowNet_CNN.py --target /path/target --decoy /path/decoy -m model/cnn.pt -p ""
```

## Known caveats
- `script/Prediction_CNN.py` imports `pad_control` from `script/WinnowNet_CNN.py`, but `script/WinnowNet_CNN.py` does not define `pad_control`.
- `script/Prediction_CNN.py` calls `model(data1, data2)` while `script/WinnowNet_CNN.Net.forward()` expects one tensor.
- If CNN prediction is needed, fix `script/Prediction_CNN.py` interface to match current CNN model.
