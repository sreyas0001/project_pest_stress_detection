
# AI-Based Pest Control System using Multispectral Imagery

This project implements an AI system to detect crop stress and potential pest infestation using multispectral drone imagery. It uses a U-Net architecture to segment stressed areas and suggests whether pest control is required based on a configurable threshold.

## Features
- **Multispectral Support**: Handles 5-band imagery (Blue, Green, Red, Red Edge, NIR).
- **Vegetation Indices**: Calculates NDVI, NDRE, GNDVI used for analysis.
- **Deep Learning**: U-Net model implemented in PyTorch for semantic segmentation.
- **Metrics**: Tracks IoU, F1-Score, Precision, and Recall.
- **Visualization**: Generates overlay maps showing healthy vs. stressed crops.
- **Decision Support**: Automates the decision for pest control application.

## Project Structure
```
├── data/               # Directory for input images and masks
│   ├── train/          # (Optional) Training split
│   └── val/            # (Optional) Validation split
├── src/                # Source code
│   ├── dataset.py      # Dataset loading and patching
│   ├── model.py        # U-Net architecture
│   ├── preprocessing.py# Image handling and indices
│   ├── train.py        # Training script
│   └── inference.py    # Inference and visualization script
├── notebooks/          # Experimentation notebooks
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

1.  **Clone the repository** (if applicable).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

The system expects multispectral images (e.g., GeoTIFF) and binary masks (e.g., PNG/TIF).

1.  **Images**: Place 5-band multispectral images in `data/images` (or `data/train/images`).
    -   Bands should be in order: Blue, Green, Red, Red Edge, NIR.
    -   If your order differs, update `BAND_MAP` in `src/preprocessing.py`.
2.  **Masks**: Place binary segmentation masks (0=Healthy, 1=Stressed) in `data/masks`.
    -   Filenames should match the images (extensions can differ).
    -   Example: `field_01.tif` -> `field_01.png`.

## Usage

### Training

To train the model on your dataset:

```bash
python src/train.py --data_dir data --epochs 50 --batch_size 4 --lr 1e-4
```

Arguments:
-   `--data_dir`: Path to dataset directory containing `images` and `masks` (or `train/val` subdirs).
-   `--patch_size`: Size of patches to train on (default 256).

### Inference

To analyze a new field image:

```bash
python src/inference.py --image_path path/to/image.tif --output_dir results
```

Arguments:
-   `--image_path`: Path to the multispectral image.
-   `--model_path`: Path to the trained `.pth` model (default `best_model.pth`).
-   `--threshold`: Stressed area ratio threshold to trigger pest control (default 0.10).

## Model Architecture

The model is a standard U-Net with:
-   **Encoder**: Double convolution blocks with max pooling.
-   **Decoder**: Upsampling (bilinear) followed by concatenation and convolution.
-   **Input**: 5 channels (B, G, R, RE, NIR).
-   **Output**: 1 channel (Probability of stress).

## Evaluation

The training script logs the following metrics:
-   **IoU (Intersection over Union)**: Detects overlap accuracy.
-   **F1-Score**: Harmonic mean of precision and recall.
-   **Precision**: Accuracy of positive predictions.
-   **Recall**: Ability to find all positive instances.

## Visualization

The `inference.py` script produces a composite image containing:
1.  **RGB View**: False-color RGB from multispectral bands.
2.  **NDVI**: Heatmap of Normalized Difference Vegetation Index.
3.  **Prediction**: Overlay of detected stress areas in Red on top of the field.
