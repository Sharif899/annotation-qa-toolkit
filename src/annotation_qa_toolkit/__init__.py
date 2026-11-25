"""annotation-qa-toolkit — validate image/video labelling work.

Offline, deterministic QA for annotation datasets: format readers,
structural checks, annotator agreement and quality reports.
"""

from .agreement import box_agreement, cohen_kappa, per_annotator_stats
from .checks import run_all_checks
from .model import Annotation, BBox, Dataset, ImageRecord
from .readers import load
from .report import build_report

__all__ = [
    "Annotation",
    "BBox",
    "Dataset",
    "ImageRecord",
    "box_agreement",
    "build_report",
    "cohen_kappa",
    "load",
    "per_annotator_stats",
    "run_all_checks",
]

__version__ = "0.1.0"
