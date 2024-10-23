# CPL-Diff
This package contains deep learning models and related scripts to run CPL-Diff.  
CPL-Diff is a diffusion model for generating peptides of fixed length. CPL-Diff employs a Transformer architecture and uses attention masks to control the length of the generated peptide sequences, and it can specify the generation of peptides with a single function.

## Installation

1. Clone the package and install CPL-Diff.
```
git clone https://github.com/luozhenjie1997/CPL-Diff.git
cd CPL-Diff
pip install .
```

2. Create conda environment using `environment.yml` file.
```
conda env create -f environment.yml
```

## Train CPL-Diff

The training code for CPL-Diff is available at scripts/train_denoiser.ipynb. The CPL-Diff hyperparameter configuration is available at config.ini. The peptide sequence generation code is available at scripts/sample_randomLen.ipynb.
