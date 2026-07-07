[README.md](https://github.com/user-attachments/files/29727195/README.md)
# Cross-Lot Generalization of Vision Foundation Models for Semiconductor Wafer Defect Classification

**Intel REU — Advanced Semiconductor Packaging | Arizona State University**
**Researcher:** Donovon Mott | **Advisor:** Dr. Leslie Hwang

## Overview

Do vision foundation models generalize better than supervised baselines when deployed on unseen manufacturing lots?

This project evaluates three vision models — DINOv2, CLIP, and ResNet-50 — on the WM-811K wafer bin map dataset using **lot-based splitting** instead of random splitting. By assigning entire manufacturing lots to either train or test (zero overlap), we simulate real fab deployment where models must classify defects on production lots they've never seen.

### Key Findings

| Model | Test Accuracy (Unseen Lots) | Generalization Gap | F1 Macro |
|-------|---------------------------|--------------------|----------|
| DINOv2 ViT-L/14 | **97.18%** | 1.5pp | 0.841 |
| CLIP ViT-L/14 | 96.80% | **0.9pp** | 0.836 |
| ResNet-50 | 94.03% | 2.5pp | 0.707 |
| CLIP Zero-Shot | 0.80% | N/A | 0.010 |

- **DINOv2** achieves the highest accuracy on unseen lots
- **CLIP** has the smallest generalization gap (best cross-lot transfer)
- **ResNet-50** collapses on rare defect classes (Edge-Loc 43%, Loc 32%, Scratch 34% F1)
- **CLIP zero-shot** fails completely — wafer maps are too far from CLIP's training distribution
- Foundation models reach **85%+ accuracy with just 50 labeled examples per class**; ResNet-50 plateaus at ~60%

## Repository Structure

```
.
├── 3modelcompare.ipynb          # Main cross-lot generalization experiment
├── README.md
└── results/
    ├── crosslot_generalization_gap.png
    ├── crosslot_per_class_f1.png
    ├── crosslot_confusion_matrices.png
    ├── crosslot_fewshot_curves.png
    ├── lot_size_distribution.png
    ├── lot_split_analysis.png
    └── crosslot_results.pkl
```

## Methodology

### Dataset
- **WM-811K**: 811,457 wafer bin maps from real 300mm production
- **172,950** expert-labeled across 9 defect classes: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none
- Wafer maps converted to 224x224 RGB images

### Lot-Based Splitting
Traditional random splits allow wafers from the same lot to appear in both train and test sets, letting models memorize lot-specific patterns. We split by `lotName`:
- **8,609 training lots** (138,024 samples)
- **2,153 test lots** (34,926 samples)
- **Zero lot overlap** between train and test

### Models
All models used as **frozen feature extractors** (no fine-tuning):

| Model | Pretraining | Parameters | Embedding Dim |
|-------|-------------|------------|---------------|
| DINOv2 ViT-L/14 | Self-supervised (DINO + iBOT) | 304M | 1024 |
| CLIP ViT-L/14 | Language-supervised (contrastive image-text) | 428M | 768 |
| ResNet-50 | Supervised ImageNet | 25.6M | 2048 |

### Evaluation
- **Linear probing**: Logistic regression on frozen embeddings
- **Few-shot learning**: 5 to 1000 labeled samples per class, evaluated on unseen lots
- **Zero-shot** (CLIP only): Text prompt classification with no labeled data
- **Generalization gap**: Train accuracy − test accuracy on lot-split data

## Running the Notebook

### Requirements
- Python 3.8+
- PyTorch with CUDA
- GPU with 16GB+ VRAM (tested on Tesla T4)

### Setup
```bash
pip install torch torchvision timm open_clip_torch umap-learn scikit-learn matplotlib seaborn
```

### Dataset
Download the [WM-811K dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) from Kaggle and update `DATA_PATH` in the notebook.

### Execution
The notebook was run on Kaggle with a Tesla T4 GPU. Total runtime is approximately 8-9 hours due to feature extraction across 172K images for three models.

```
Cell  1-3:   Setup & imports (~1 min)
Cell  4-9:   Data loading & lot-based splitting (~2 min)
Cell 10-11:  Dataset creation (~1 min)
Cell 12-15:  Model loading (~5 min)
Cell 16-19:  Feature extraction (~4.5 hours)
Cell 20-21:  Linear probe evaluation (~5 min)
Cell 22-30:  Results & visualizations (~10 min)
Cell 31-32:  CLIP zero-shot (~30 min)
Cell 33-35:  Summary & export (~1 min)
```

## References

1. Oquab, M. et al. (2024). [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193). TMLR.
2. Radford, A. et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). ICML.
3. Wu, M. et al. (2014). [Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map). IEEE TSM.
4. Dosovitskiy, A. et al. (2021). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). ICLR.
5. Shen, Z. et al. (2021). [Deep Transfer Wasserstein Adversarial Network for Wafer Map Defect Recognition](https://www.sciencedirect.com/science/article/abs/pii/S0360835221005830). C&IE.
6. NVIDIA (2024). [Optimizing Semiconductor Defect Classification with Vision Foundation Models](https://developer.nvidia.com/blog/optimizing-semiconductor-defect-classification-with-generative-ai-and-vision-foundation-models/).
7. He, K. et al. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). CVPR.

## License

This project was completed as part of the Intel Research Experience for Undergraduates (REU) program at Arizona State University.
