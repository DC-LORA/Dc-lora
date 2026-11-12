# DC-LoRA

**DC-LoRA: Dual-Constrained Low-Rank Adaptation for Continual Source-Free Semantic Segmentation**

DC-LoRA combines targeted low-rank adaptation with EMA-teacher pseudo-labeling and conservative pseudo-label refresh.

## Setup

```text
python -m pip install -r requirements.txt scikit-learn visdom
```

Stable-ST also requires `opencv-python-headless` and `scikit-image`.

Download the seven-class GTA checkpoint that matches the model and use its path in `--ckpt`. [Pretrained and adapted checkpoints](https://drive.google.com/drive/folders/1nSN9s2AoaFojca579igrom5CbmHK5wUl?usp=sharing)

Use the preparation scripts in [`datasets/`](datasets/) to map masks to the seven superclasses. Under `datasets/data/{cityscapes,idd,mapillary}`, place images in `leftImg8bit/{train,val}` and masks in `gtFine/{train,val}`. Use paired names such as `sample_leftImg8bit.png` and `sample_gtFine_labelIds.png`. [Cityscapes](https://www.cityscapes-dataset.com/) · [IDD](https://idd.insaan.iiit.ac.in/) · [Mapillary Vistas](https://www.mapillary.com/dataset/vistas) · [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/)

## DC-LoRA

Run from the repository root. Start with the GTA checkpoint and train one domain at a time in the order Cityscapes → IDD → Mapillary.

```text
python main.py --dataset cityscapes --data_root datasets/data --model deeplabv3plus_resnet101 --gpu_id 0 --ckpt checkpoints/gta_deeplabv3.pth --self_training --use_lora --active_adapter cityscapes --new_adapter_per_domain --pseudo_label_dir pseudo_labels/cityscapes
```

For IDD and Mapillary, point `--data_root` to a parent containing the prepared target as `cityscapes` and use a new adapter name and pseudo-label directory. Copy each stage's `best_*.pth` to a domain-specific filename before starting the next run. Pass that file to `--ckpt` so the next stage starts from the merged EMA weights selected on the preceding target's validation split.

Evaluate the selected Mapillary checkpoint on each target by changing `--input` to its validation image folder. The example uses `dc_lora_mapillary.pth` as the saved checkpoint name.

```text
python predict.py --dataset cityscapes --model deeplabv3plus_resnet101 --gpu_id 0 --input datasets/data/cityscapes/leftImg8bit/val --ckpt checkpoints/dc_lora_mapillary.pth
```

Add `--save_val_results_to predictions/cityscapes` to save color masks.

## C-SFDA, DT-ST, and Stable-ST

Start each method from the DeepLabV2 GTA checkpoint. The [baseline runner](baselines_faithful_implementation.py) accepts `--method csfda`, `--method dtst`, or `--method stable-st`.

```text
python baselines_faithful_implementation.py --method csfda --dataset cityscapes --data-root datasets/data --gpu_id 0 --ckpt checkpoints/gta_deeplabv2.pth
```

Run the next stage with `--dataset idd` or `--dataset mapillary` and pass the preceding stage's `checkpoints/baselines/<method>/<dataset>/best.pth` to `--ckpt`. Evaluate the selected Mapillary checkpoint separately on each target using `--eval`.

```text
python baselines_faithful_implementation.py --method csfda --dataset cityscapes --eval --data-root datasets/data --gpu_id 0 --ckpt checkpoints/baselines/csfda/mapillary/best.pth
```

## Acknowledgement

This repository is adapted from [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch). We thank the original authors for the training and inference framework. The baseline runner adapts [C-SFDA](https://openaccess.thecvf.com/content/CVPR2023/html/Karim_C-SFDA_A_Curriculum_Learning_Aided_Self-Training_Framework_for_Efficient_Source_CVPR_2023_paper.html), [DT-ST](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Towards_Better_Stability_and_Adaptability_Improve_Online_Self-Training_for_Model_CVPR_2023_paper.html), and [Stable-ST](https://doi.org/10.1007/s11263-026-02929-6) to the continual adaptation protocol in this repository. The baselines use the [DeepLabV2](https://doi.org/10.1109/TPAMI.2017.2699184) architecture.
