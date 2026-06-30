# DC-LoRA

**DC-LoRA: Dual-Constrained Low-Rank Adaptation for Continual Source-Free Semantic Segmentation**

DC-LoRA is a continual source-free adaptation framework for semantic segmentation. It combines targeted low-rank adaptation with EMA-teacher pseudo-labeling and temporally consistent pseudo-label refresh. The method updates only selected LoRA adapters and batch-normalization affine terms, then merges the learned updates into a single deployable checkpoint after each target domain.

## Checkpoints

The pretrained and adapted checkpoints are available at the following link:

```text
CHECKPOINT_LINK_HERE
```

The released checkpoints include source and adapted models used for the GTA → Cityscapes → IDD → Mapillary continual adaptation protocol.


## Datasets

The original datasets are not included in this repository. They can be downloaded from their official sources:

```text
Cityscapes:
https://www.cityscapes-dataset.com/

IDD:
https://idd.insaan.iiit.ac.in/

Mapillary Vistas:
https://www.mapillary.com/dataset/vistas

GTA5 / Playing for Data:
https://download.visinf.tu-darmstadt.de/data/from_games/
```

After downloading, the datasets should be organized (to 7-superclass) according to the loaders provided in the `datasets/` folder.

## Requirements

The code was tested with Python 3.9, PyTorch 2.1.0, CUDA 11.8, and cuDNN 8.7.

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Acknowledgement

This repository is adapted from the open-source DeepLabV3Plus-Pytorch implementation:

```text
https://github.com/VainF/DeepLabV3Plus-Pytorch
```

We thank the original authors for providing the DeepLabV3/DeepLabV3+ training and inference framework. This repository extends it with DC-LoRA for continual source-free semantic segmentation, including layer-targeted LoRA adaptation, EMA-teacher pseudo-labeling, temporally consistent pseudo-label refresh, checkpoint merging, and 7-superclass continual adaptation support.

