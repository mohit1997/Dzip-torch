# Dzip
## improved general-purpose lossless compression based on novel neural network modeling
#### Arxiv: https://arxiv.org/abs/1911.03572
## Description
DZip is a general lossless compressor for sequential data which uses NN-based modelling combined with arithmetic coding. We refer to the NN-based model as the "combined model", as it is composed of a bootstrap model and a supporter model. The bootstrap model is trained prior to compression on the data to be compressed, and the resulting model parameters (weights) are stored as part of the compressed output (after being losslessly compressed with BSC). The combined model is adaptively trained (bootstrap model parameters are fixed) while compressing the data, and hence its parameters do not need to be stored as part of the compressed output.

## Requirements
0. GPU
1. Python3 (<= 3.6.8)
2. Numpy
3. Sklearn
4. Pytorch (gpu/cpu) 1.4


### Download and install dependencies
Download:
```bash
git clone https://github.com/mohit1997/Dzip-torch.git
```
To set up virtual environment and dependencies (on Linux):
```bash
cd DZip
python3 -m venv torch
source torch/bin/activate
bash install.sh
```
