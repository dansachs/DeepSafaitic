# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-30

### Added
- Initial release of DeepSafaitic
- Siamese network architecture for Safaitic glyph recognition
- Training pipeline optimized for Google Colab
- Glyph detection pipeline using computer vision
- Interactive labeling tool with ductus tracking
- Aggressive data augmentation pipeline
- Support for 28 Safaitic characters
- Database integration for ground truth
- Checkpoint resuming and early stopping
- Timestamped model versioning

### Features
- Two-stage detection and classification pipeline
- Interactive spline visualization for reading paths
- Support for multiple text lines
- Ruler filtering and small glyph detection
- Export to CSV/JSON with angles and path IDs
- Zoom, pan, and rotation controls in labeler

## [Unreleased]

### Planned
- Fine-tuning on real stone glyphs
- Text direction prediction
- Character recognition confidence scores
- End-to-end transcription pipeline
- Support for additional ancient scripts

