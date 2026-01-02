"""
DeepSafaitic: Neural Epigraphy for Ancient Desert Inscriptions

A machine learning pipeline for reading and interpreting Safaitic rock inscriptions.
"""

__version__ = "0.1.0"
__author__ = "DeepSafaitic Contributors"

from .model import SafaiticSiameseNet, ContrastiveLoss, euclidean_distance
from .dataset import SafaiticSiameseDataset

__all__ = [
    'SafaiticSiameseNet',
    'ContrastiveLoss',
    'euclidean_distance',
    'SafaiticSiameseDataset',
]

