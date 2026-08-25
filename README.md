# Retinal Blood Vessel Segmentation using U-Net

A simple, from-scratch **U-Net** implementation in TensorFlow/Keras for segmenting blood vessels in retinal fundus images. The model takes a grayscale fundus image as input and outputs a binary mask highlighting the vascular structure, which is useful for diagnosing conditions such as diabetic retinopathy, glaucoma, and hypertensive retinopathy.

## Overview

Retinal blood vessel segmentation is a foundational step in automated screening for diabetic retinopathy and other retinal diseases. Manual vessel tracing by ophthalmologists is time-consuming and prone to inter-observer variability, so a reliable automated segmentation model can support faster, more consistent diagnosis.

This project implements the classic **U-Net encoder-decoder architecture** trained end-to-end on grayscale fundus images to produce pixel-wise vessel probability maps. The pipeline covers data loading, preprocessing, model building, training with multiple batch-size configurations, and qualitative/quantitative evaluation.

## Dataset

This project uses the [Retina Blood Vessel dataset](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel) on Kaggle, which is a preprocessed version of the DRIVE dataset.

| Split | Images | Masks | Resolution used |
|---|---|---|---|
| Train | 80 | 80 | 256 × 256 (grayscale) |
| Test | 20 | 20 | 256 × 256 (grayscale) |

Each fundus image has a corresponding binary ground-truth mask outlining the retinal vasculature.

## Model Architecture

A standard U-Net with 4 downsampling and 4 upsampling stages, skip connections between corresponding encoder/decoder levels, and a sigmoid output layer for binary segmentation.

- **Input:** 256 × 256 × 1 (single-channel grayscale fundus image)
- **Encoder:** 4 blocks of `[Conv2D → Conv2D → MaxPooling2D]`, filters doubling at each stage (64 → 128 → 256 → 512), with dropout (0.5) before the final pooling stage
- **Bottleneck:** 1024-filter double convolution with dropout (0.5)
- **Decoder:** 4 blocks of `[UpSampling2D → Conv2D → Concatenate (skip connection) → Conv2D → Conv2D]`, filters halving at each stage (512 → 256 → 128 → 64)
- **Output layer:** `Conv2D(1, 1x1, activation='sigmoid')` producing a per-pixel vessel probability map
- **Total parameters:** ~31.0M (all trainable)

```
Input (256,256,1)
 ├─ Conv64 → Conv64 ──────────────────────────────────┐
 │     ↓ MaxPool                                       │ skip
 ├─ Conv128 → Conv128 ─────────────────────────────┐   │
 │     ↓ MaxPool                                    │  │
 ├─ Conv256 → Conv256 ───────────────────────────┐  │  │
 │     ↓ MaxPool                                  │  │  │
 ├─ Conv512 → Conv512 → Dropout(0.5) ──────────┐  │  │  │
 │     ↓ MaxPool                                │  │  │  │
 └─ Conv1024 → Conv1024 → Dropout(0.5)  (bottleneck)
       ↓ UpSample → Conv512 → Concat ───────────┘  │  │  │
       → Conv512 → Conv512
       ↓ UpSample → Conv256 → Concat ──────────────┘  │  │
       → Conv256 → Conv256
       ↓ UpSample → Conv128 → Concat ─────────────────┘  │
       → Conv128 → Conv128
       ↓ UpSample → Conv64 → Concat ────────────────────┘
       → Conv64 → Conv64 → Conv1 (1x1, sigmoid)
Output (256,256,1)
```

## Training Setup

- **Framework:** TensorFlow / Keras
- **Optimizer:** Adam
- **Loss function:** Binary cross-entropy
- **Metrics:** Accuracy, custom Dice coefficient, custom IoU (Jaccard index)
- **Epochs:** 175 per configuration
- **Batch sizes tested:** 8, 4, 2, 1 (best checkpoint saved on validation Dice coefficient via `ModelCheckpoint`)
- **Preprocessing:** grayscale conversion, resize to 256×256, pixel normalization to [0, 1], channel dimension expansion

## Results

Sample Output:




Best validation performance was achieved with a batch size of 2:

| Batch size | Best train Dice | Best val Dice | Best train IoU | Best val IoU |
|---|---|---|---|---|
| 8 | ~0.645 | ~0.651 | ~0.476 | ~0.483 |
| 4 | ~0.686 | ~0.657 | ~0.502 | ~0.489 |
| 2 | ~0.756 | **~0.688** | ~0.607 | **~0.524** |
| 1 | ~0.757 | ~0.688 | ~0.608 | ~0.525 |

> Note: exact values vary slightly by run seed; see the training logs in the notebook for full per-epoch metrics.

## Getting Started

### Prerequisites

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn kagglehub
```

### Usage

1. Download the dataset via `kagglehub` (handled automatically in the notebook) or manually from [Kaggle](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel).
2. Open `UNet_Retinal_Segmentation.ipynb` in Jupyter/Colab.
3. Run all cells to preprocess the data, build the U-Net, train the model, and visualize predictions.
4. The best-performing model weights are saved to `best_model.keras`.

## Evaluation Metrics

- **Dice coefficient:** measures overlap between predicted and ground-truth masks, weighted toward the size of the intersection.
- **IoU (Intersection over Union):** measures overlap relative to the union of predicted and ground-truth vessel pixels.

Both metrics are implemented as custom TensorFlow functions and tracked during training for both the training and validation sets.

## Future Improvements

- Add data augmentation (rotation, flipping, elastic deformation) to improve generalization given the small dataset size (80 training images).
- Experiment with a combined Dice + BCE loss to better handle class imbalance between vessel and background pixels.
- Try deeper backbones or attention-gated U-Net variants (Attention U-Net, U-Net++) for improved thin-vessel recall.
- Apply CLAHE contrast enhancement as a preprocessing step, common in retinal image analysis.

## References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
- Dataset: [Retina Blood Vessel — Kaggle](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel) (derived from the DRIVE dataset).
