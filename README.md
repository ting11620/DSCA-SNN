## DSCA-SNN Code

This repository provides a release for verifying the main DSCA-SNN.

## 1. Files and folders

- `SNN_layers/` contains the DSCA-LIF neuronal dynamics and the multi-branch DSCA layer.
- `SHD/` contains SHD preprocessing and training code.
- `S-MNIST/` contains the S-MNIST training code.
- `PS-MNIST/` contains the PS-MNIST training code.

## 2. Datasets

1. **SHD**  
   The Spiking Heidelberg Digits (SHD) dataset can be downloaded from:  
   https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/

2. **S-MNIST and PS-MNIST**  
   Both are based on MNIST and are loaded through `torchvision.datasets.MNIST`:  
   https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html

## 3. Prerequisites

- Python >= 3.9
- PyTorch == 1.12.1
- torchvision == 0.13.1
- NumPy
- PyTables (`tables`)

```bash
pip install -r requirements.txt
