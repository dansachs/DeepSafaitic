# DeepSafaitic: Neural Epigraphy for Ancient Desert Inscriptions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

A proof-of-concept machine learning pipeline for automatically reading and interpreting Safaitic rock inscriptions using deep learning and computer vision techniques.

## Overview

Safaitic is an ancient North Arabian script carved into desert rocks across modern-day Jordan, Syria, and Saudi Arabia. This project develops an end-to-end system that can:

1. **Detect** glyph locations in raw stone photographs
2. **Classify** individual Safaitic characters using Siamese neural networks
3. **Track** reading order and text direction (ductus) through interactive labeling
4. **Transcribe** complete inscriptions from images to text

## What Makes This Unique

### 1. **Siamese Network Architecture for Epigraphic Recognition**
Unlike standard OCR systems designed for printed text, this project uses Siamese networks trained on a small dataset of ideal glyphs. The model learns to recognize characters by comparing them to reference images, making it robust to:
- Extreme variations in stone texture and weathering
- 360° rotation (Safaitic can be written in any direction)
- Broken or fragmented characters
- Variable lighting conditions in field photography

### 2. **Two-Stage Detection and Classification Pipeline**
The system separates detection from classification:
- **Stage 1**: Computer vision techniques (adaptive thresholding, contour detection) locate potential glyphs
- **Stage 2**: Deep learning model classifies each detected region
This modular approach allows for iterative improvement of each component independently.

### 3. **Interactive Ductus Tracking**
Safaitic inscriptions have no fixed reading direction. Our interactive labeler allows researchers to:
- Manually correct detection errors
- Track the "ductus" (path) of the inscription through spline visualization
- Handle multiple text lines on the same stone
- Export structured data with reading order and directional angles

### 4. **Aggressive Data Augmentation for Small Datasets**
With only 28 unique Safaitic characters available, the system uses extensive augmentation:
- 360° rotation (SafeRotate)
- Elastic distortion
- Stone texture noise simulation
- Line thickness variation (erosion)
- Random occlusions
- This generates thousands of training pairs from a minimal dataset

### 5. **Real-World Field Photography Support**
The pipeline handles:
- High-resolution stone photographs from archaeological databases
- Variable image quality and lighting
- Ruler filtering (removes measurement tools from detection)
- Small glyph detection (e.g., "ayn" dots)
- Integration with SQLite databases for ground truth

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/DeepSafaitic.git
cd DeepSafaitic

# Install dependencies
pip install -r requirements.txt

# Install as a package (optional)
pip install -e .
```

## Project Structure

```
DeepSafaitic/
├── src/
│   └── deepsafaitic/          # Core package
│       ├── __init__.py
│       ├── model.py          # Siamese network architecture
│       └── dataset.py         # Dataset with augmentation
├── scripts/
│   ├── colab_train.py        # Training script (Google Colab)
│   ├── scan_detection_only.py # Glyph detection pipeline
│   ├── interactive_labeler.py # Interactive labeling tool
│   ├── utils/                 # Utility scripts
│   └── legacy/                # Deprecated scripts
├── notebooks/
│   └── safaitic_training.ipynb  # Jupyter training notebook
├── tests/                     # Test files
├── data/                      # Data directories (gitignored)
│   ├── cleaned_glyphs/       # Training glyph images
│   └── stone_images/         # Stone photographs
├── outputs/                   # Output directories (gitignored)
│   ├── detection_results/    # Detection outputs
│   └── labeled_results/      # Labeled data exports
├── docs/                      # Documentation
│   └── images/               # Documentation images
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

### Training (Google Colab)

1. Upload `scripts/colab_train.py` and `src/deepsafaitic/` to Google Colab
2. Mount Google Drive
3. Run the training cell

See `docs/` for detailed training guides.

### Detection

```bash
python scripts/scan_detection_only.py data/stone_images/stone_16820.jpg
```

### Interactive Labeling

```bash
python scripts/interactive_labeler.py data/stone_images/stone_16820.jpg
```

## Key Components

### Siamese Network (`src/deepsafaitic/model.py`)
- Pre-trained ResNet18 backbone
- 512-dimensional embedding space
- L2-normalized embeddings
- Contrastive loss (margin=2.0)
- Euclidean distance for similarity

### Detection Pipeline (`scripts/scan_detection_only.py`)
- Adaptive thresholding for binary mask creation
- Contour detection with aspect ratio filtering
- Ruler detection and removal
- Configurable minimum area thresholds
- Outputs timestamped results with ground truth from database

### Interactive Labeler (`scripts/interactive_labeler.py`)
- Click-and-drag box creation
- Automatic numbering and path prediction
- Manual reordering with ↑↓ keys
- Spline visualization of reading path
- Split view (original + binary mask)
- Zoom, pan, and rotation controls
- Export to CSV/JSON with angles and path IDs

## Training Pipeline Features

- **Checkpoint resuming**: Resume from saved checkpoints
- **Periodic checkpointing**: Save every 10 epochs
- **Early stopping**: Prevent overfitting
- **Gradient clipping**: Stabilize training
- **GPU memory monitoring**: Optimize resource usage
- **Timestamped models**: Track model versions
- **Reduced validation frequency**: Speed up training

## Dataset

The training dataset consists of:
- 28 unique Safaitic characters
- Each character has an "ideal" and "square" variant
- Ground truth transliterations from archaeological database
- Full stone images with known transcriptions

## Documentation

See the `docs/` directory for detailed guides:
- Setup instructions
- Training guides
- Usage examples
- Testing procedures

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Future Work

- [ ] Fine-tune model on real stone glyphs (extracted from labeled data)
- [ ] Implement text direction prediction
- [ ] Add character recognition confidence scores
- [ ] Develop end-to-end transcription pipeline
- [ ] Expand to other ancient scripts

## Citation

If you use this code in your research, please cite:

```bibtex
@software{deepsafaitic2024,
  title={DeepSafaitic: Neural Epigraphy for Ancient Desert Inscriptions},
  author={DeepSafaitic Contributors},
  year={2024},
  url={https://github.com/YOUR_USERNAME/DeepSafaitic}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Safaitic inscription data from OCIANA (Online Corpus of the Inscriptions of Ancient North Arabia)
- Archaeological database integration for ground truth
- Built with [PyTorch](https://pytorch.org/), [Albumentations](https://albumentations.ai/), and [OpenCV](https://opencv.org/)

## Contact

For questions or issues, please open an issue on GitHub.
