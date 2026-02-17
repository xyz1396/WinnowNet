# WinnowNet PSM + Spectra Rescoring Skills Guide

## Scope
This guide summarizes the current code in all Python files in this repository, with focus on:
- PSM feature inputs
- spectra feature inputs
- CNN and Transformer frameworks used for PSM rescoring
- small demos for adding/removing features and changing precursor m/z parsing style

## Python files reviewed
- `Assembling_all.py`
- `PSM_feature.py`
- `Prediction.py`
- `Prediction_CNN.py`
- `SpectraFeatures.py`
- `SpectraFeatures_training.py`
- `WinnowNet_Att.py`
- `WinnowNet_CNN.py`
- `filtering.py`
- `filtering_shuffle.py`
- `parseconfig.py`
- `sipros_peptides_assembling.py`
- `sipros_post_module.py`
- `components/__init__.py`
- `components/encoders.py`
- `components/feedforward.py`
- `components/mixins.py`
- `components/transformers.py`

## Real Python file paths
- `Assembling_all.py`
- `PSM_feature.py`
- `Prediction.py`
- `Prediction_CNN.py`
- `SpectraFeatures.py`
- `SpectraFeatures_training.py`
- `WinnowNet_Att.py`
- `WinnowNet_CNN.py`
- `filtering.py`
- `filtering_shuffle.py`
- `parseconfig.py`
- `sipros_peptides_assembling.py`
- `sipros_post_module.py`
- `components/__init__.py`
- `components/encoders.py`
- `components/feedforward.py`
- `components/mixins.py`
- `components/transformers.py`

## End-to-end rescoring flow
1. Generate features from TSV + MS2 using `SpectraFeatures.py` (inference) or `SpectraFeatures_training.py` (training).
2. Run rescoring model:
- Transformer: `Prediction.py` with `WinnowNet_Att.DualPeakClassifier`
- CNN: `Prediction_CNN.py` with `WinnowNet_CNN.Net`
3. Filter by FDR using `filtering.py` / `filtering_shuffle.py`.
4. Optional protein-level assembly with `sipros_peptides_assembling.py`.

## PSM feature input (hand-crafted tabular features)
Implemented in `feature_dict()`:
- `qvalue`
- `qn`
- `qnl`
- `theory_mass + H`
- `delta_mass`
- `abs(delta_mass)`
- `peptide_length`
- `num_missed_cleavages`
- charge one-hot: `[z1, z2, z3plus]`

Code locations:
- `SpectraFeatures.py:197`
- `SpectraFeatures_training.py:197`

Notes:
- `qvalue`, peptide string, scan, charge are parsed in `read_tsv()`.
- peptide theoretical mass is computed from `AA_dict` plus termini.

## Spectra feature input

### MS2 parsing
`expToDict()` reads MS2 blocks:
- `S` line initializes scan + precursor mass
- `Z` line updates precursor mass/charge using `(mass - PROTON_MASS)` and charge
- numeric lines are fragment peaks `[m/z, intensity]`

Code location:
- `SpectraFeatures.py:91`
- `SpectraFeatures_training.py:91`

### Theoretical ion parsing
`theoryToDict()` reads Sipros-generated theoretical ions from text output.

Code location:
- `SpectraFeatures.py:124`
- `SpectraFeatures_training.py:124`

### Sipros binary I/O for theoretical b/y-like fragments
This is the exact contract used by the feature scripts to generate theoretical fragment spectra from PSM IDs.

#### 1) Inputs prepared from PSM candidates (`read_tsv`)
`read_tsv()` writes three aligned temporary files (same row order):
- `idx.txt`: PSM IDs (string key, e.g. `data_33_1_2`)
- `charge.txt`: precursor charge per PSM (e.g. `1`, `2`, `3`)
- `peptide.txt`: peptide sequence for that PSM (internal peptide form, typically stripped from `K.PEPTIDE.R` to `PEPTIDE`)

Where this is written:
- `SpectraFeatures.py:149`
- `SpectraFeatures_training.py:149`

#### 2) Binary call
The scripts invoke:
```bash
./Sipros_OpenMP -i1 idx.txt -i2 charge.txt -i3 peptide.txt -i4 <theoretical_file>
```
- `-i1`: PSM IDs
- `-i2`: charges
- `-i3`: peptide strings
- `-i4`: output path for theoretical fragments (named from MS2 file, `*.ms2 -> *.txt`)

Where this happens:
- `SpectraFeatures.py:309`
- `SpectraFeatures_training.py:309`

#### 3) Output text format expected by Python
`theoryToDict()` assumes the generated file is organized in fixed-size blocks:
- every 7th line (`line_id % 7 == 0`) is a block header key
- the next 6 lines contain space-separated numeric tokens in alternating pairs:
  `mz1 intensity1 mz2 intensity2 ...`
- all pairs from those 6 lines are collected as `[[mz, intensity], ...]`

In parser logic:
- header line is saved as dictionary key (`key = line`)
- numeric pairs are parsed by even/odd token index and appended to one list
- each key is stored as `theory_dic[key] = sorted(scan)`

This means Python expects one theoretical-fragment record per PSM key with a 7-line frame.

Where parsed:
- `SpectraFeatures.py:124`
- `SpectraFeatures_training.py:124`

#### 4) Read-back mapping into downstream feature extraction
After loading `D_theory = theoryToDict(...)`, each theoretical record is joined with:
- experimental scan peaks from `D_exp[...]`
- per-PSM tabular features from `D_feature[...]`

Join key logic used during extraction:
- iterate `for key in D_theory`
- scan id is recovered with `key.split('_')[-3]`
- retrieval pattern:
  - `D_exp[key.split('_')[-3]]`
  - `D_theory[key]`
  - `D_feature[key]`

Then:
- `IonExtract(...)` (CNN mode) builds matched triplets
- `IonExtract_Att(...)` (attention mode) keeps experimental/theoretical peak tables

Finally, for inference script consistency, output dict is ordered by PSM feature keys:
- `return_dict = {k: return_dict[k] for k in D_feature.keys()}`
  (`SpectraFeatures.py:336`)

### Matching and tensor construction

#### CNN mode (`IonExtract`)
- For each theoretical ion, find experimental peaks within `diffDa` (default 1 Da).
- Build triplets: `[delta_mz, exp_intensity, theory_intensity]`.
- Sort/pad to `pairmaxlength=500`.
- Standardize intensity channels.
- Output currently stored as `[xFeatures, X_add_feature]`.

Code location:
- `SpectraFeatures.py:222`
- `SpectraFeatures_training.py:222`

#### Attention mode (`IonExtract_Att`)
- Standardize intensity channel of experimental and theoretical spectra separately.
- Output stored as `[Xexp, Xtheory]`.

Code location:
- `SpectraFeatures.py:242`
- `SpectraFeatures_training.py:242`

## Rescoring network frameworks

## Transformer framework (active in `Prediction.py`)
Defined in `WinnowNet_Att.py`:
- `MS2Encoder`:
  - `PeakEncoder` embeds `(m/z, intensity)` into `dim_model`
  - stacked `nn.TransformerEncoder` (`n_layers`, `n_heads`)
  - padding mask from zero rows
- `DualPeakClassifier`:
  - two independent encoders: one for experimental peak set, one for theoretical peak set
  - mean-pool each encoder output
  - concatenate pooled vectors
  - linear classifier to 2 logits

Key locations:
- `WinnowNet_Att.py:114`
- `WinnowNet_Att.py:148`
- `components/encoders.py:62`

Default inference config in `Prediction.py`:
- `dim_model=256`
- `n_heads=4`
- `dim_feedforward=512`
- `n_layers=4`
- `dropout=0.3`
- `max_len=200`

Code location:
- `Prediction.py:108`

## CNN framework (PointNet-style)
Defined in `WinnowNet_CNN.py`:
- input tensor shape built as 3 channels per matched ion (`delta_mz`, exp intensity, theory intensity)
- `T_Net(k=3)` learns input transform matrix
- `Conv1d(3->64->128->1024)` + global max pool
- second feature transform `T_Net(k=64)`
- MLP head `1024->512->256->2`

Key locations:
- `WinnowNet_CNN.py:102`
- `WinnowNet_CNN.py:140`
- `WinnowNet_CNN.py:178`

## Important implementation notes before feature edits
1. CNN inference script and CNN training model interface are inconsistent:
- `Prediction_CNN.py` calls `model(data1, data2)` (`Prediction_CNN.py:64`)
- but `WinnowNet_CNN.Net.forward()` expects one input (`WinnowNet_CNN.py:189`)

2. `Prediction_CNN.py` imports `pad_control` from `WinnowNet_CNN`, but that function is not defined there.

3. In CNN training, loaded feature uses only `D_features[j][0]` (`WinnowNet_CNN.py:49`), so tabular `X_add_feature` is not consumed by the current CNN model.

4. In Attention mode, `IonExtract_Att()` returns only `[Xexp, Xtheory]`, so tabular `X_add_feature` is not consumed by current Transformer model.

If you want new hand-crafted PSM features to affect rescoring, you must wire them into model input explicitly.

## Small demo A: add a new PSM feature
Example: add `is_oxidized` (1 if peptide contains `~`, else 0).

### Step 1: add to feature builder
Edit both files:
- `SpectraFeatures.py`
- `SpectraFeatures_training.py`

Patch pattern in `feature_dict()`:
```python
is_oxidized = 1 if '~' in pep.identified_pep else 0
D_feature[pep.PSMId] = [
    pep.qvalue, pep.qn, pep.qnl, pep.theory_mass + H,
    pep.delta_mass, abs(pep.delta_mass), pep.peplen,
    pep.num_missed_cleavages, is_oxidized,
]
```
Then keep/adjust charge one-hot append.

### Step 2 (required if model should use it): fuse tabular features into model
Current models ignore `X_add_feature`. One practical approach:
- keep spectra encoder output as-is
- add small MLP for tabular vector
- concatenate with spectra embedding before final classifier

For `DualPeakClassifier` (concept):
```python
self.meta = nn.Sequential(
    nn.Linear(meta_dim, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
)
self.classifier = nn.Linear(2 * dim_model + 64, num_classes)

meta_out = self.meta(meta_features)
joint = torch.cat([rep1, rep2, meta_out], dim=-1)
logits = self.classifier(joint)
```

### Step 3: update dataset output
Return `(xspectra1, xspectra2, meta_features, y, weight)` in training and `(xspectra1, xspectra2, meta_features)` in prediction.

## Small demo B: delete an existing feature
Example: remove `num_missed_cleavages`.

1. Remove it from `feature_dict()` in both feature scripts.
2. If your model consumes tabular features, reduce `meta_dim` accordingly.
3. Regenerate all `.pkl` feature files; old pickles will not match new schema.
4. Retrain or at least fine-tune; feature dimensionality changed.

## Small demo C: change precursor m/z style in MS2 parsing
Current behavior prefers `Z` information when available (`expToDict()`).

If your MS2 format requires always using `S` line precursor m/z, change `expToDict()` logic to ignore `Z` replacement.

Example edit (apply to both feature scripts):
```python
# in expToDict()
if ms[0] == 'S':
    ms_scan = str(int(ms.split()[1]))
    precursor_mz = float(ms.split()[-1])
    msdict[ms_scan] = [f"{precursor_mz}_0"]
elif ms[0] == 'Z':
    # keep charge only; do not overwrite precursor mass from S line
    charge = int(ms.split()[1])
    precursor_mz = float(msdict[ms_scan][0].split('_')[0])
    msdict[ms_scan][0] = f"{precursor_mz}_{charge}"
```

Alternative style (if file stores neutral mass and you want m/z):
- convert using `mz = (neutral_mass + z * PROTON_MASS) / z`
- store `mz_charge` consistently
- update downstream `read_tsv()` mass calculations accordingly

## Validation checklist after any feature/schema change
1. Regenerate `.pkl` features using one mode (`att` or `cnn`) and inspect one sample key.
2. Confirm `Dataset.__getitem__` returns tensors with expected shapes.
3. Run one forward pass and verify no dimension errors.
4. Run a short train/inference sanity test before full run.
5. Re-run filtering to ensure rescored output length matches input PSM count.

## Quick commands
Transformer inference:
```bash
python SpectraFeatures.py -i input.tsv -s input.ms2 -o spectra.pkl -t 20 -f att
python Prediction.py -i spectra.pkl -m marine_att.pt -o rescore.out.txt
```

CNN inference (after interface mismatch is fixed):
```bash
python SpectraFeatures.py -i input.tsv -s input.ms2 -o spectra.pkl -t 20 -f cnn
python Prediction_CNN.py -i spectra.pkl -m cnn_pytorch.pt -o rescore.out.txt
```
