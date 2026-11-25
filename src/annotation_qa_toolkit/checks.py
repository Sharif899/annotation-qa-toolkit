"""QA checks.

Each check is a named function returning ``Finding`` objects — a
severity (``error`` / ``warning`` / ``info``), the check name, the
image id, and a human-readable message. Nothing here mutates the
dataset; a QA report must describe problems, never silently fix them.

Severity guide:
* ``error`` — the annotation is unusable as it stands (out of bounds,
  degenerate box, orphan annotation, unknown class).
* ``warning`` — probably wrong and worth a reviewer's time (duplicate
  box, heavy overlap, tiny box, missing annotations, near-duplicate
  class names, rare class).
* ``info`` — context worth knowing (class imbalance summary).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .model import Dataset

SEVERITIES = ("error", "warning", "info")


@dataclass(frozen=True)
class Finding:
    severity: str
    check: str
    image_id: str
    message: str

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "check": self.check,
            "image_id": self.image_id,
            "message": self.message,
        }


def check_empty_annotations(dataset: Dataset) -> list[Finding]:
    """Images in the dataset that carry no annotations at all."""
    return [
        Finding(
            "warning", "empty_annotations", img.id,
            f"Image '{img.file_name}' has no annotations.",
        )
        for img in dataset.images_with_no_annotations()
    ]


def check_orphan_annotations(dataset: Dataset) -> list[Finding]:
    """Annotations whose image_id has no image record in the dataset."""
    known = {img.id for img in dataset.images}
    return [
        Finding(
            "error", "orphan_annotation", ann.image_id,
            f"Annotation '{ann.category}' refers to unknown image '{ann.image_id}'.",
        )
        for ann in dataset.annotations
        if ann.image_id not in known
    ]


def check_out_of_bounds(dataset: Dataset) -> list[Finding]:
    """Boxes that extend beyond the image edges."""
    findings: list[Finding] = []
    for ann in dataset.annotations:
        img = dataset.image_by_id(ann.image_id)
        if img is None or img.width == 0 or img.height == 0:
            continue
        box = ann.bbox
        if box.x_min < 0 or box.y_min < 0 or box.x_max > img.width or box.y_max > img.height:
            findings.append(
                Finding(
                    "error", "out_of_bounds", ann.image_id,
                    f"'{ann.category}' box ({box.x_min:.0f}, {box.y_min:.0f}, "
                    f"{box.x_max:.0f}, {box.y_max:.0f}) exceeds image "
                    f"{img.width}x{img.height}.",
                )
            )
    return findings


def check_degenerate_boxes(dataset: Dataset) -> list[Finding]:
    """Boxes with zero area (zero width/height or inverted corners)."""
    return [
        Finding(
            "error", "degenerate_box", ann.image_id,
            f"'{ann.category}' box ({ann.bbox.x_min:.0f}, {ann.bbox.y_min:.0f}, "
            f"{ann.bbox.x_max:.0f}, {ann.bbox.y_max:.0f}) has no area.",
        )
        for ann in dataset.annotations
        if ann.bbox.is_degenerate
    ]


def check_tiny_boxes(dataset: Dataset, min_side: float = 8.0) -> list[Finding]:
    """Boxes whose width or height is below ``min_side`` pixels.

    Very small boxes are usually a mis-click or a box drawn around a
    speck. The default (8 px) is a starting point, not a rule — pass a
    different ``min_side`` for small-object datasets.
    """
    return [
        Finding(
            "warning", "tiny_box", ann.image_id,
            f"'{ann.category}' box is only {ann.bbox.width:.0f}x{ann.bbox.height:.0f} px "
            f"(minimum side {min_side:.0f} px).",
        )
        for ann in dataset.annotations
        if not ann.bbox.is_degenerate
        and (ann.bbox.width < min_side or ann.bbox.height < min_side)
    ]


def check_duplicate_boxes(dataset: Dataset, iou_threshold: float = 0.95) -> list[Finding]:
    """Same-class boxes on the same image that are near-identical."""
    findings: list[Finding] = []
    for img in dataset.images:
        anns = dataset.annotations_for(img.id)
        for i, first in enumerate(anns):
            for second in anns[i + 1:]:
                if first.category != second.category:
                    continue
                iou = first.bbox.iou(second.bbox)
                if iou >= iou_threshold:
                    findings.append(
                        Finding(
                            "warning", "duplicate_box", img.id,
                            f"Two '{first.category}' boxes overlap almost completely "
                            f"(IoU {iou:.2f}) — likely drawn twice.",
                        )
                    )
    return findings


def check_heavy_overlap(dataset: Dataset, iou_threshold: float = 0.5) -> list[Finding]:
    """Same-class boxes that overlap heavily without being duplicates.

    Two boxes of the same class with IoU >= 0.5 usually mean one
    object was boxed twice with different extents, or two objects
    were merged into overlapping boxes.
    """
    findings: list[Finding] = []
    for img in dataset.images:
        anns = dataset.annotations_for(img.id)
        for i, first in enumerate(anns):
            for second in anns[i + 1:]:
                if first.category != second.category:
                    continue
                iou = first.bbox.iou(second.bbox)
                if iou_threshold <= iou < 0.95:
                    findings.append(
                        Finding(
                            "warning", "heavy_overlap", img.id,
                            f"Two '{first.category}' boxes overlap heavily (IoU {iou:.2f}).",
                        )
                    )
    return findings


def check_unknown_classes(dataset: Dataset) -> list[Finding]:
    """Annotations whose class is not in the dataset's class list."""
    known = set(dataset.categories)
    return [
        Finding(
            "error", "unknown_class", ann.image_id,
            f"Annotation uses class '{ann.category}', which is not in the class list.",
        )
        for ann in dataset.annotations
        if ann.category not in known
    ]


def _normalise_class(name: str) -> str:
    """Lower-case, strip, collapse spaces, drop a trailing plural 's'."""
    name = re.sub(r"\s+", " ", name.strip().lower())
    if len(name) > 1 and name.endswith("s") and not name.endswith("ss"):
        name = name[:-1]
    return name


def check_near_duplicate_classes(dataset: Dataset) -> list[Finding]:
    """Class names that normalise to the same thing ('Car' vs 'cars').

    Returns one finding per colliding group, listing the variants.
    """
    groups: dict[str, set[str]] = {}
    for name in dataset.categories:
        groups.setdefault(_normalise_class(name), set()).add(name)
    findings: list[Finding] = []
    for normalised, variants in sorted(groups.items()):
        if len(variants) > 1:
            findings.append(
                Finding(
                    "warning", "near_duplicate_class", "",
                    f"Classes {sorted(variants)} look like the same class "
                    f"(normalised to '{normalised}') — pick one spelling.",
                )
            )
    # Also catch annotation classes that collide with a listed class.
    listed = {_normalise_class(c): c for c in dataset.categories}
    for ann in dataset.annotations:
        norm = _normalise_class(ann.category)
        if norm in listed and listed[norm] != ann.category:
            findings.append(
                Finding(
                    "warning", "near_duplicate_class", ann.image_id,
                    f"Annotation class '{ann.category}' matches listed class "
                    f"'{listed[norm]}' after normalisation.",
                )
            )
    return findings


def check_class_imbalance(
    dataset: Dataset, rare_share: float = 0.02
) -> list[Finding]:
    """Classes that make up less than ``rare_share`` of annotations.

    A class with a handful of examples cannot be learned or evaluated
    properly. Returns one ``warning`` per rare class; a balanced
    dataset produces no findings. (The full per-class counts are
    available from ``Dataset.category_counts`` and from ``annoqa load``.)
    """
    counts = dataset.category_counts()
    total = sum(counts.values())
    findings: list[Finding] = []
    if total == 0:
        return findings
    for cls, n in sorted(counts.items()):
        if n / total < rare_share:
            findings.append(
                Finding(
                    "warning", "rare_class", "",
                    f"Class '{cls}' has only {n} of {total} annotations "
                    f"({100 * n / total:.1f}%) — below the {100 * rare_share:.0f}% "
                    f"rare-class threshold.",
                )
            )
    return findings


CHECKS = (
    check_empty_annotations,
    check_orphan_annotations,
    check_out_of_bounds,
    check_degenerate_boxes,
    check_tiny_boxes,
    check_duplicate_boxes,
    check_heavy_overlap,
    check_unknown_classes,
    check_near_duplicate_classes,
    check_class_imbalance,
)


def run_all_checks(
    dataset: Dataset, min_side: float = 8.0, rare_share: float = 0.02
) -> list[Finding]:
    """Run every check and return findings in a deterministic order."""
    findings: list[Finding] = []
    for check in CHECKS:
        if check in (check_tiny_boxes,):
            findings.extend(check(dataset, min_side=min_side))
        elif check is check_class_imbalance:
            findings.extend(check(dataset, rare_share=rare_share))
        else:
            findings.extend(check(dataset))
    findings.sort(key=lambda f: (f.severity, f.check, f.image_id, f.message))
    return findings
