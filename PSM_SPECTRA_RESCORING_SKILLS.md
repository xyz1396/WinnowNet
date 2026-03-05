# WinnowNet PSM + Spectra Rescoring Skills Guide

## Scope
This document describes the current behavior of the rescoring pipeline in this repo after recent updates:
- `sipros_theoretical_spectra` integration
- new `theory.txt` block format
- FT1/FT2 parsing changes
- CNN/ATT feature generation and training usage

## Main pipeline
1. Generate spectrum feature pickle with:
- `SpectraFeatures_training.py` (training feature generation)
- `SpectraFeatures.py` (same logic, inference/production usage)

2. Train rescoring model:
- Attention/Transformer: `WinnowNet_Att.py`
- CNN (PointNet-style): `WinnowNet_CNN.py`

3. Predict rescoring scores:
- Attention: `Prediction.py`
- CNN: `Prediction_CNN.py`

4. Downstream filtering/assembly:
- `filtering.py`, `filtering_shuffle.py`, `sipros_peptides_assembling.py`

## Dataset convention used now
Training scripts support dataset folders with:
- `pct2`: positive PSMs
- `pct1`: negative PSMs

Each folder typically contains matched `.tsv` and `.pkl` pairs.
`WinnowNet_Att.py` and `WinnowNet_CNN.py` auto-pair files by stem and split into train/val/test.

## Feature-generation inputs
`SpectraFeatures_training.py` / `SpectraFeatures.py` consume:
- PSM table (`-i`, TSV)
- FT1 (`-1`, MS1)
- FT2 (`-2` or legacy `-s`, MS2)
- config file for `sipros_theoretical_spectra` (`-c`)

If `-1` is not provided but FT2 path ends with `.FT2`, the script auto-tries sibling `.FT1`.

## Current CLI options (feature generation)
- `-i`: tab-delimited PSM file
- `-1`: FT1 file
- `-2`: FT2 file
- `-s`: FT2 legacy alias
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
2. Match MS2 peaks against theoretical fragment peaks.
3. Fragment peaks are first filtered by envelope metadata:
- group by `(fragment_kind, fragment_position)`
- keep top `n` by theoretical intensity in each group (`-n`, default `3`)
4. If no matches found, fallback match against filtered `theory_all`.

Each match yields one triplet feature:
- `[delta_mz, experimental_intensity, theoretical_intensity]`

Post-processing:
- pad/truncate to `pairmaxlength=500`
- z-score normalize intensity channels (columns 2 and 3 of triplet)
- transpose to shape `[3, pairmaxlength]`

Output entry in feature pickle (CNN path):
- `[xFeatures, X_add_feature]`

Current CNN trainer usage (`WinnowNet_CNN.py`):
- uses only `D_features[j][0]` (`xFeatures`)
- ignores `X_add_feature`

## Attention feature construction (`IonExtract_Att`)
Used when `-f att`.

Process:
1. Filter MS1 peaks by user-defined isolation window `-w`, centered on most abundant theoretical precursor m/z.
2. Build `exp_all = filtered_ms1 + ms2`.
3. Use theoretical `all` as `theory_all`.
4. Standardize intensity separately for experimental and theoretical sets.

Output entry in feature pickle (ATT path):
- `[Xexp, Xtheory]`

Current ATT trainer usage (`WinnowNet_Att.py`):
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
python SpectraFeatures_training.py \
  -i /path/pct2/sample_filtered_psms.tsv \
  -1 /path/pct2/sample.FT1 \
  -2 /path/pct2/sample.FT2 \
  -o /path/pct2/sample.pkl \
  -f att -t 8 -c SiprosConfig.cfg -d 0.01 -w 10
```

### Generate CNN features with envelope top-5
```bash
python SpectraFeatures_training.py \
  -i /path/pct2/sample_filtered_psms.tsv \
  -1 /path/pct2/sample.FT1 \
  -2 /path/pct2/sample.FT2 \
  -o /path/pct2/sample.pkl \
  -f cnn -t 8 -c SiprosConfig.cfg -d 0.01 -n 5
```

### Train ATT model using pct1/pct2 folders
```bash
python WinnowNet_Att.py -i /path/sip_example -m model/att.pt -p ""
```

### Train CNN model using pct1/pct2 folders
```bash
python WinnowNet_CNN.py -i /path/sip_example -m model/cnn.pt -p ""
```

## Known caveats
- `Prediction_CNN.py` imports `pad_control` from `WinnowNet_CNN`, but `WinnowNet_CNN.py` does not define `pad_control`.
- `Prediction_CNN.py` calls `model(data1, data2)` while `WinnowNet_CNN.Net.forward()` expects one tensor.
- If CNN prediction is needed, fix `Prediction_CNN.py` interface to match current CNN model.

