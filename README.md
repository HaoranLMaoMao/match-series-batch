# Match-Series-Batch

Batch processing tool for non-rigid alignment of image stacks using pyMatchSeries.


You can install `match-series-batch` via pip:
```bash
pip install match-series-batch


After installation, you can use the command line tool:
match-series-batch --input /path/to/your/input_root --output /path/to/your/output_root --lambda 20 --prefix Aligned_ --dtype uint8


Example:
match-series-batch --input /data/STEM_series --output /data/STEM_aligned --lambda 15 --prefix Corrected_ --dtype uint16


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
