# DINOv2 vs DINOv3: Wafer Map Defect Classification

Comparing Meta's DINOv2 and DINOv3 vision foundation models as frozen feature extractors for semiconductor wafer bin map (WBM) defect classification using the WM-811K dataset.

## Overview

Semiconductor fabs need to classify spatial defect patterns on wafer bin maps to diagnose manufacturing issues. This project evaluates whether newer self-supervised vision models (DINOv3) produce better representations than their predecessors (DINOv2) for this task — without any fine-tuning of the backbone.

## What the Notebook Tests

- **Linear probe accuracy** — logistic regression on frozen embeddings from both models
- **Few-shot learning curves** — accuracy vs. number of labeled examples per class (5 to 1000), with error bars across 5 random trials
- **Embedding space quality** — UMAP/t-SNE projections, silhouette scores, k-NN accuracy
- **Per-class F1 breakdown** — highlights performance on rare defect classes (Donut, Near-full)
- **Attention map comparison** — where each model focuses on 6 different defect types, with difference maps
- **Radar chart** — holistic view across all quality metrics

## Dataset

**WM-811K** — 811,457 wafer bin maps from real semiconductor production, ~172K with expert-assigned labels across 9 defect classes:

`Center | Donut | Edge-Loc | Edge-Ring | Loc | Near-full | None | Random | Scratch`

Download from [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) and place `LSWMD.pkl` in the project directory.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- GPU with 16+ GB VRAM (24 GB recommended)

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers timm scikit-learn
pip install matplotlib seaborn pandas numpy
pip install umap-learn kagglehub
```

## Project Structure

```
├── dinov2_vs_dinov3_wafer_comparison.ipynb   # Main comparison notebook
├── README.md
└── .gitignore
```

## Key Takeaway

The few-shot learning curves are the most practically relevant result — they show how many expert-labeled examples each model needs to reach production-quality accuracy. The model that gets there first directly translates to less labeling effort in the fab.
