"""
Core training and evaluation logic for the lung disease classification model.
"""

from .training import train
from .evaluation import evaluate_model

__all__ = ["train", "evaluate_model"]

