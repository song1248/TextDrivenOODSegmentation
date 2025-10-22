# TDOS: Leveraging Text-Driven Semantic Variation for Robust OOD Segmentation

This repository accompanies the project **“Leveraging Text-Driven Semantic Variation for Robust OOD Segmentation”**, which introduces a text-aware framework for out-of-distribution (OOD) semantic segmentation in autonomous driving scenarios. The codebase extends the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) ecosystem with CLIP-based visual and textual reasoning to detect unseen road anomalies more reliably.

## Overview

- **Text-Driven OOD Segmentation** – Couples a CLIP vision transformer or ResNet backbone with a CLIP text encoder and a Mask2Former-style decoder to align visual features with flexible textual prompts.
- **Distance-Based OOD Prompts** – Generates prompts at varying semantic distances (WordNet-driven in the paper) from in-distribution (ID) classes to carve out clearer ID/OOD decision boundaries.
- **OOD Semantic Augmentation** – Uses self-attention feature perturbations to synthesize diverse OOD prototypes without inserting external objects.
- **Vision/Text Regularization** – Freezes pretrained vision-language experts while learning prompt parameters and decoder heads; includes auxiliary identity head for stability.
- **State-of-the-art OOD Segmentation** – Demonstrated on Fishyscapes, Segment-Me-If-You-Can, and Road Anomaly benchmarks, outperforming pixel- and object-level baselines.


## Repository Layout

- `configs/`: Experiment configurations; `configs/tdos/*.py` cover CLIP ViT-B and ResNet backbones across multiple datasets.
- `models/`: Custom segmentor (`TextDrivenOODSeg`) plus CLIP backbones and utilities.
- `mmseg/`: Forked MMSegmentation modules (datasets, runners, layers) required by the new model.
- `pretrained/`: Expected location for CLIP checkpoints (e.g., `CLIP-ViT-B-16.pt`, `RN50.pt`).
- `tools/`: Launcher utilities adapted from MMSegmentation (dataset converters, distributed runners, etc.).
- `train.py`: Entry point for single-node training that wires together config, datasets, and logging.
- `dist_train.sh` : Helper scripts for multi-GPU training via PyTorch distributed.


## Environment Setup

1. **Create a Python environment** (CUDA 11.x + PyTorch ≥ 2.0 recommended):
   ```bash
   conda create -n tdos python=3.9
   conda activate tdos
   ```
2. **Install PyTorch** built for your CUDA toolkit:
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Install project dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Place CLIP checkpoints** under `pretrained/`:
   - `CLIP-ViT-B-16.pt` for ViT backbones.
   - `RN50.pt` for ResNet backbones.
   Use filenames that match the `pretrained` entries in the chosen config.

## Dataset Preparation

The configs assume Cityscapes as the in-distribution dataset, with several OOD benchmarks for evaluation:

- **Fishyscapes (Lost & Found / Static)**  
- **Segment-Me-If-You-Can (Anomaly Track / Obstacle Track)**  
- **Road Anomaly**  

Update the dataset roots inside the base configs before training:

- `configs/_base_/datasets/city2city-512.py`
- `configs/_base_/datasets/city2fishy_*.py`
- `configs/_base_/datasets/city2road_anomaly*.py`

Each config defines a `data_root` pointing to the author’s environment (e.g., `/home/jovyan/...`). Replace these with your local paths for `img_dir` and `ann_dir`.

## Training

### Single-node / Single-GPU

```bash
python train.py configs/tdos/text_driven_ood_seg_vit-b_1e-5_20k-c2c-512.py \
    --work-dir ./work_dirs/tdos_vit_b_cityscapes
```

- Logs and checkpoints are saved in `work_dirs/<config_name>/`.
- Use `--load-from` to warm-start from a checkpoint, or `--resume-from` to continue training.
- `--finetune` switches the model to eval mode before training the decoder/prompt heads only.

### Multi-GPU (Distributed)

```bash
bash dist_train.sh configs/tdos/text_driven_ood_seg_vit-b_1e-5_20k-c2fishy_LnF-512.py 4 \
    --work-dir ./work_dirs/tdos_vit_b_fishyscapes
```

`dist_train.sh` forwards any extra arguments to `train.py`. Ensure that `CUDA_VISIBLE_DEVICES` is set before launching if you need a subset of GPUs.

### Hyperparameters

- Optimizer: `AdamW` with per-module LR multipliers (backbone/text encoder frozen by default).
- Prompt length: controlled via `context_length` in the config and dynamically extended in `TextDrivenOODSeg`.
- `learnable_cls_num` (default 50) determines the number of negative prompt slots appended to the decoder queries.

Adjust these in the config or the segmentor as required for new datasets.

## Evaluation & Inference

Evaluate checkpoints using the matching config:

```bash
bash dist_test.sh configs/tdos/text_driven_ood_seg_vit-b_1e-5_20k-c2road_anomaly-512.py \
    PRETRAINED_CHECKPOINT.pth 4 --eval mIoU
```

To visualize predictions, adapt `tools/test.py` with `--show-dir` or add custom hooks inside `train.py`.

## Citation

If you build upon this work, please cite the project:

```
@inproceedings{song2025tdos,
  title     = {Leveraging Text-Driven Semantic Variation for Robust OOD Segmentation},
  author    = {Seungheon Song and Jaekoo Lee},
  booktitle = {Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2025}
}
```

## Acknowledgements

Built on top of [CLIP](https://github.com/openai/CLIP) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Please refer to their licenses and cite them as appropriate for your research.
