# WinnowNet
This algorithm was implemented and tested on Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-84-generic, x86_64).
## Note: 
This repository contains the development version of WinnowNet. For the code used to reproduce the experiments in the paper, please refer to the following repository: https://github.com/Biocomputing-Research-Group/WinnowNet4Review

## Overview
WinnowNet is designed for advanced processing of mass spectrometry data with two core methods: a CNN-based approach and a self-attention-based approach. The repository includes scripts for feature extraction, model training, prediction (inference), and evaluation. A toy example is included to help users get started.

## Table of Contents
- [WinnowNet](#winnownet)
  - [Note:](#note)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Setup and installation](#setup-and-installation)
  - [Requirements](#requirements)
  - [Download Required Files](#download-required-files)
  - [Input pre-processing](#input-pre-processing)
  - [Training WinnowNet Models](#training-winnownet-models)
  - [Requirements](#requirements-1)
    - [Datasets](#datasets)
    - [Self-Attention-Based WinnowNet](#self-attention-based-winnownet)
      - [Phase 1: Training on Easy Tasks (Synthetic Data)](#phase-1-training-on-easy-tasks-synthetic-data)
      - [Phase 2: Training on Difficult Tasks (Real Data)](#phase-2-training-on-difficult-tasks-real-data)
    - [CNN-Based WinnowNet](#cnn-based-winnownet)
      - [Phase 1: Training on Easy Tasks (Synthetic Data)](#phase-1-training-on-easy-tasks-synthetic-data-1)
      - [Phase 2: Training on Difficult Tasks (Real Data)](#phase-2-training-on-difficult-tasks-real-data-1)
    - [Notes](#notes)
  - [Inference](#inference)
    - [PSM Rescoring](#psm-rescoring)
      - [Self-Attention-Based WinnowNet](#self-attention-based-winnownet-1)
      - [CNN-Based WinnowNet](#cnn-based-winnownet-1)
  - [Evaluation](#evaluation)
    - [FDR Control at the PSM/Peptide Levels](#fdr-control-at-the-psmpeptide-levels)
  - [Contact and Support](#contact-and-support)

## Setup and installation
Create and activate the `winnownet` environment:
```bash
micromamba create -n winnownet python=3.12 pytorch=2.5.1 numpy pandas scikit-learn einops matplotlib
micromamba activate winnownet
```
## Requirements
* **Operation system**: Linux
* **GPU Memory**
  * **Inference Mode**: 2 GB (adjust batch size if necessary)
  * **Training Mode**: 2 GB (adjust batch size if necessary)

## Download Required Files
* Pre-trained model can be downloaded via:
  * **CNN-based WinnowNet**: [cnn_pytorch.pt](https://figshare.com/articles/dataset/Models/25513531)
  * **Self-Attention-based WinnowNet**: [marine_att.pt](https://figshare.com/articles/dataset/Models/25513531)
* A toy example is provided in this repository.
* **Sample Input Datasets**
[Mass spectra data for Benchmark datasets](https://figshare.com/articles/dataset/Datasets/25511770)
Other raw files benchmark datasets can be downloaded via:
[PXD007587](https://www.ebi.ac.uk/pride/archive/projects/PXD007587), [PXD006118](https://www.ebi.ac.uk/pride/archive/projects/PXD006118), [PXD013386](https://www.ebi.ac.uk/pride/archive/projects/PXD006118), [PXD023217](https://www.ebi.ac.uk/pride/archive/projects/PXD023217), [PXD035759](https://www.ebi.ac.uk/pride/archive/projects/PXD035759)

## Input pre-processing

Extract fragment ion matching features along with 11 additional features derived from both theoretical and experimental spectra. The PSM (peptide-spectrum match) candidate information should be provided in a tab-delimited file (e.g., a TSV file output from Percolator).
```bash
micromamba run -n winnownet python script/SpectraFeatures.py -i <tsv_file> -1 <ft1_file> -2 <ft2_file> -o spectra.pkl -t 48 -f cnn
```
* Replace `<tsv_file>` with the path to your PSM candidates file.
* Replace `<ft1_file>` with the path to your FT1 file.
* Replace `<ft2_file>` with the path to your FT2 file.
* The `-t 48` option sets the number of threads (adjust this value as needed).
* Use `-f cnn` when preparing input for the CNN-based architecture or `-f att` for the self-attention-based model.

## Training WinnowNet Models

This folder contains scripts, datasets, and instructions for training two variants of the WinnowNet deep learning model: a self-attention-based model and a CNN-based model. Training is carried out in two phases to enable curriculum learning from synthetic (easy) to real-world metaproteomic (difficult) datasets.

## Requirements

- Python 3.7+
- PyTorch
- NumPy, Pandas, scikit-learn

### Datasets

- **Prosit_train.zip** (Phase 1 training set):   https://figshare.com/articles/dataset/Datasets/25511770?file=55257041
- **marine1_train.zip** (Phase 2 training set): https://figshare.com/articles/dataset/Datasets/25511770?file=55257035

---

### Self-Attention-Based WinnowNet

#### Phase 1: Training on Easy Tasks (Synthetic Data)

```bash
micromamba run -n winnownet python script/SpectraFeatures.py -i filename.tsv -1 filename.FT1 -2 filename.FT2 -o spectra_feature.pkl -t 20 -f att
micromamba run -n winnownet python script/WinnowNet_Att.py -i spectra_feature_directory -m prosit_att.pt
```

**Explanation of options:**
- `-i`: Directory of `.pkl` feature files. Labels are read from each pickle, which now stores the original TSV/PIN row plus parsed `Label` / `q-value`.
- `-1`: Corresponding FT1 file.
- `-2`: Corresponding FT2 file (filename should match TSV).
- `-o`: Output file to store extracted features as a `pkl` file.
- `-t`: Number of threads for parallel processing.
- `-f`: Feature type (`att` for self-attention model).
- `-m`: Filename to save the trained model.
- Optional split-mode training is also supported with `-target target.pkl[,more.pkl] -decoy decoy.pkl[,more.pkl]`; embedded labels in those pickles are ignored in that mode.

#### Phase 2: Training on Difficult Tasks (Real Data)

```bash
micromamba run -n winnownet python script/SpectraFeatures.py -i filename.tsv -1 filename.FT1 -2 filename.FT2 -o spectra_feature.pkl -t 20 -f att
micromamba run -n winnownet python script/WinnowNet_Att.py -i spectra_feature_directory -m marine_att.pt -p prosit_att.pt
```

- `-p`: Pre-trained model from Phase 1.
- A for-loop is needed to convert all `tsv` files to `pkl` files.

**Pre-trained model:** marine_att.pt,  https://figshare.com/articles/dataset/Models/25513531

---

### CNN-Based WinnowNet

#### Phase 1: Training on Easy Tasks (Synthetic Data)

```bash
micromamba run -n winnownet python script/SpectraFeatures.py -i filename.tsv -1 filename.FT1 -2 filename.FT2 -o spectra_feature.pkl -t 20 -f cnn
micromamba run -n winnownet python script/WinnowNet_CNN.py -i spectra_feature_directory -m prosit_cnn.pt
```

#### Phase 2: Training on Difficult Tasks (Real Data)

```bash
micromamba run -n winnownet python script/SpectraFeatures.py -i filename.tsv -1 filename.FT1 -2 filename.FT2 -o spectra_feature.pkl -t 20 -f cnn
micromamba run -n winnownet python script/WinnowNet_CNN.py -i spectra_feature_directory -m cnn_pytorch.pt -p prosit_cnn.pt
```

**Pre-trained model:** cnn_pytorch.pt, https://figshare.com/articles/dataset/Models/25513531

---

### Notes

- All input FT1/FT2/TSV files must be preprocessed properly.
- Models trained in Phase 1 are reused to initialize weights in Phase 2.
- Training with GPU is recommended for performance.

## Inference
### PSM Rescoring
#### Self-Attention-Based WinnowNet
To generate input representations for PSM candidates and perform re-scoring using the self-attention model, run:
```bash
micromamba run -n winnownet python script/SpectraFeatures.py -i tsv_file -1 file.FT1 -2 file.FT2 -o spectra.pkl -t 48 -f att 
micromamba run -n winnownet python script/Prediction.py -i spectra.pkl -o rescore.out.tsv -m att_pytorch.pt  

```
#### CNN-Based WinnowNet
To generate input representations for PSM candidates and perform re-scoring using the CNN model, run:
```bash
micromamba run -n winnownet python script/SpectraFeatures.py -i filename.tsv -1 filename.FT1 -2 filename.FT2 -o spectra.pkl -t 48 -f cnn
micromamba run -n winnownet python script/Prediction_CNN.py -i spectra.pkl -o rescore.out.tsv -m cnn_pytorch.pt 

```
**Explanation of options:**
- `-i`: Input feature pickle produced by `script/SpectraFeatures.py`
- `-1`: Corresponding FT1 file.
- `-2`: Corresponding FT2 file (filename should match TSV).
- `-o`: Output file to store extracted features as a `pkl` file.
- `-t`: Number of threads for parallel processing.
- `-f`: Feature type (`att` for self-attention model, `cnn`for CNN model).
- `-m`: Filename to save the trained model.
- Prediction output is now a rescored TSV containing all original TSV/PIN columns plus updated `score`, `q-value`, and `Label` fields.

## Evaluation
### FDR Control at the PSM/Peptide Levels
Filter the re-scored PSM candidates to control the false discovery rate (FDR) at both the PSM and peptide levels (targeted at 1% FDR). You will need both the original PSM file and the re-scoring results.
```bash
micromamba run -n winnownet python script/filtering.py -i rescore.out.txt -p tsv_file -o filtered -d Rev_ -f 0.01
```
**Explanation of options:**
- `-i`: Rescoring file from WinnowNet
- `-p`: Input tab-delimited file with PSMs
- `-o`: filtered results' prefix
- `-d`: Decoy prefix used for target-decoy strategy. Default: Rev_
- `-f`: False Discovery Rate. Default: 0.01
- A for-loop is needed to convert all `tsv` files to `pkl` files.

* The filtered output files include updated PSM information (new predicted scores, spectrum IDs, identified peptides, and corresponding proteins).
* Assembling filtered identified peptides into proteins
* This script is needed to run at the working directory inlucding filtered results at PSM and Peptide levels.
```bash
micromamba run -n winnownet python script/sipros_peptides_assembling.py
```
When assembling filtered, identified peptides into proteins, the overall protein-level FDR depends on the quality of the filtered peptide list. An initial peptide-level FDR (for example, 1%) may lead to a protein-level FDR that is higher than desired. In such cases, you need to re-filter the peptides using a stricter (i.e., lower) FDR threshold until you achieve a 1% protein-level FDR. 

## Contact and Support
For further assistance, please consult the GitHub repository or reach out to the project maintainers.
