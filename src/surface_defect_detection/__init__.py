"""Surface defect detection research package."""

from .config import resolve_dataset_root, default_split_names
from .dataset import audit_dataset, validate_dataset

__all__ = [
    "resolve_dataset_root",
    "default_split_names",
    "audit_dataset",
    "validate_dataset",
]
