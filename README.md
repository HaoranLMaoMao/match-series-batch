# Match-Series-Batch v1.1.2

pymatchseries is necessary.Also aequires scikit-learn to be installed for NLPCA.
make life easier

## Advantage
match-series-batch is a batch non-rigid alignment tool for microscope moving image (.dm4 format) sequences, integrating optional denoising, stage averaged plot export, automated logging and CLI calls, maint:

    •    High throughput: batch processing of multiple sample folders at once
    •    Flexible denoising: optional NLMeans, NLPCA or full skip
    •    Automatic catalogue: generates a separate, time-stamped working catalogue for each calculation, preserving all intermediate results
    •    Multiple outputs: TIFF, HSPY formats for previewing and research archiving.
    •    No human intervention: completely unattended from loading to saving

## Features
- Batch non-rigid registration for multiple image series
- Supports flexible denoising: NLMeans, NLPCA, or none
- Saves each registered frame as TIFF
- Saves full registered stack as HSPY
- Exports stage average images as TIFF and HSPY
- Customizable NLPCA parameters: patch size, number of clusters, number of components
- Automatically generates a unique working directory for each run
- Detailed logging of processing steps
- Command-line interface

##Installation
You can install `match-series-batch` via pip:
```bash
pip install match-series-batch
```

## Usage
After installation, you can use the command line in Terminal :
``` bash
match-series-batch [OPTIONS]
```

Example:
```bash
match-series-batch --input ./input --output ./output --denoising nlpca --nlpca_patch_size 7 --nlpca_n_clusters 10 --nlpca_n_components 8
```

or in Jupyter notebook, can use:
```
import match_series_batch
!match-series-batch[OPTIONS]
```
# Command-line Arguments Description

- `--input`
  
  Specify the root folder containing sample subfolders.  
  Each subfolder will be treated as a separate dataset for alignment.  
  (Default: set in `config.py`)

- `--output`
  
  Specify the root folder where the aligned results will be saved.  
  (Default: set in `config.py`)

- `--lambda`
  
  Set the regularization parameter for non-rigid deformation.  
  A higher value leads to smoother deformation but potentially less precise alignment.  
  A lower value allows more local deformation for better registration.  
  (Default: 20)

- `--prefix`
  
  Set the prefix for naming output files, including `.tiff`, `.dm4`, and `.hspy` files.  
  (Default: `Aligned_`)

- `--dtype`
  
  Choose the output image data type:  
  `uint8` (0–255 grayscale) or `uint16` (0–65535 grayscale).  
  Useful for preserving dynamic range.  
  (Default: `uint8`)

- `--denoising`
  
 "nlmeans" , "nlpca" or "none"
 As set in config.py, Denoising method applied before saving images
  (Default: `nlpca`)


Notes

    •    Input folder must contain subfolders (one for each sample), each with .dm4 images.
    •    Output will include .tiff, a full aligned stack .hspy, and stage-average images.
    •    Full processing logs are recorded automatically.


##Notice for Macbook Users
If your laptop CPU is a Macbook M1/M2, MatchSeries will only work with X86_64,
Run these scripts in a terminal to generate an X86 environment for M-series processors:
```bash
# Go to the main directory
cd ~

# Printing information
echo "Downloading Miniforge3 (x86_64)..."

# Download Miniforge3 x86_64
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh

# Install Miniforge3 x86_64
echo "Install Miniforge3 (x86_64)..."
bash Miniforge3-MacOSX-x86_64.sh -b -p $HOME/miniforge_x86_64

# initialize Conda
echo "🔧 initialize Conda (x86_64)..."
source ~/miniforge_x86_64/bin/activate

# Switch to x86_64 architecture to run
arch -x86_64 /usr/bin/env bash <<'EOF'
    echo "being created match-x86 env..."
    source ~/miniforge_x86_64/bin/activate
    conda create -y -n match-x86 python=3.10
    conda activate match-x86

    echo "Install match-series, pyMatchSeries, hyperspy..."
    conda install -y -c conda-forge match-series
    pip install pyMatchSeries hyperspy

    echo "Installation is complete！"
```

In the Mac's own Terminal, run:
```bash
arch -x86_64 /usr/bin/env bash

source ~/miniforge_x86_64/bin/activate
conda activate match-x86
```


