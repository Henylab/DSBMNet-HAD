# DSBMNet-HAD
Implementation scripts of DSBMNet for hyperspectral anomaly detection. The manuscript is currently under peer review, and more details will be released soon.

## Setup

### Requirements

Our experiments are implemented in:

- Python 3.12.9
- PyTorch 2.3.0
- torchvision 0.18.0
- numpy 1.26.0
- scipy 1.14.1

## Prepare Dataset

Put the dataset(.mat [data, map, E]) into ./dataset

## Training and Testing

Running main_HAD.py

-If you want to train and inference on your own dataset, add the dataset name and adjust parameters such as the learning rate and masked center block size.

## Acknowledgement

The codes are based on [UADNet](https://github.com/lwdQAQ/TGRS2024_UADNet) and [DIFF Transformer](https://github.com/microsoft/unilm/tree/master/Diff-Transformer). Thanks for their awesome work.

