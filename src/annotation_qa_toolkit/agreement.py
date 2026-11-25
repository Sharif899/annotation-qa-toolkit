"""Annotator agreement.

Two views of whether two annotators agree:

* **Cohen's kappa on class presence** — for each image and each
  class, did both annotators mark the class present, both absent, or
  disagree? Kappa corrects for how often they would agree by chance.
  It is a statement about *which classes* each annotator saw, not
  about where they drew the boxes.
* **IoU box agreement** — match boxes between the two annotators
  greedily by best IoU (threshold 0.5, same class only) and report
  matched / missed / extra boxes. This is a heuristic: matching by
  best IoU can pair boxes that belong to different objects when
  objects are crowded, so treat the numbers as a review signal, not
  ground truth.

``per_annotator_stats`` summarises each annotator's volume and class
coverage, which is how a QA lead spots an annotator who is labelling
far fewer boxes — or far more — than their peers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import Annotation, Dataset


@dataclass(frozen=True)
class KappaResult:
    kappa: float
    observed_agreement: float
    expected_agreement: float
    n_decisions: int
    disagreements: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kappa": round(self.kappa, 3),
            "observed_agreement": round(self.observed_agreement, 3),
            "expected_agreement": round(self.expected_agreement, 3),
            "n_decisions": self.n_decisions,
            "disagreements": self.disagreements,
        }


def _class_presence(dataset: Dataset) -> dict[str, set[str]]:
    """Map image id -> set of classes present on that image."""
    presence: dict[str, set[str]] = {}
    for ann in dataset.annotations:
        presence.setdefault(ann.image_id, set()).add(ann.category)
    for img in dataset.images:
        presence.setdefault(img.id, set())
    return presence


def cohen_kappa(first: Dataset, second: Dataset) -> KappaResult:
    """Cohen's kappa on per-image class presence between two datasets.

    Each (image, class) pair is one decision: present or absent.
    ``first`` and ``second`` should cover the same images; classes
    present in either dataset are considered.
    """
    image_ids = sorted({img.id for img in first.images} | {img.id for img in second.images})
    classes = sorted({a.category for a in first.annotations} | {a.category for a in second.annotations})
    first_presence = _class_presence(first)
    second_presence = _class_presence(second)
    # Confusion counts: [both present, first only, second only, both absent]
    both_yes = first_only = second_only = both_no = 0
    disagreements: list[dict] = []
    for image_id in image_ids:
        for cls in classes:
            a = cls in first_presence.get(image_id, set())
            b = cls in second_presence.get(image_id, set())
            if a and b:
                both_yes += 1
            elif a and not b:
                first_only += 1
                disagreements.append({"image_id": image_id, "category": cls, "first": True, "second": False})
            elif b and not a:
                second_only += 1
                disagreements.append({"image_id": image_id, "category": cls, "first": False, "second": True})
            else:
                both_no += 1
    n = both_yes + first_only + second_only + both_no
    if n == 0:
        return KappaResult(kappa=0.0, observed_agreement=0.0, expected_agreement=0.0, n_decisions=0)
    observed = (both_yes + both_no) / n
    p_first_yes = (both_yes + first_only) / n
    p_second_yes = (both_yes + second_only) / n
    expected = p_first_yes * p_second_yes + (1 - p_first_yes) * (1 - p_second_yes)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return KappaResult(
        kappa=round(kappa, 3),
        observed_agreement=round(observed, 3),
        expected_agreement=round(expected, 3),
        n_decisions=n,
        disagreements=disagreements,
    )


@dataclass(frozen=True)
class BoxMatch:
    image_id: str
    category: str
    iou: float


@dataclass
class BoxAgreement:
    matched: int = 0
    missed: int = 0  # in first, not in second
    extra: int = 0  # in second, not in first
    matches: list[BoxMatch] = field(default_factory=list)
    missed_boxes: list[dict] = field(default_factory=list)
    extra_boxes: list[dict] = field(default_factory=list)
    per_image: dict[str, dict] = field(default_factory=dict)

    @property
    def total_first(self) -> int:
        return self.matched + self.missed

    @property
    def total_second(self) -> int:
        return self.matched + self.extra

    @property
    def match_rate(self) -> float:
        """Share of the first annotator's boxes the second also drew."""
        return round(self.matched / self.total_first, 3) if self.total_first else 0.0

    @property
    def mean_iou(self) -> float:
        if not self.matches:
            return 0.0
        return round(sum(m.iou for m in self.matches) / len(self.matches), 3)

    def to_dict(self) -> dict:
        return {
            "matched": self.matched,
            "missed": self.missed,
            "extra": self.extra,
            "match_rate": self.match_rate,
            "mean_iou": self.mean_iou,
            "per_image": self.per_image,
            "missed_boxes": self.missed_boxes,
            "extra_boxes": self.extra_boxes,
        }


def _greedy_match(first: list[Annotation], second: list[Annotation], iou_threshold: float) -> tuple[list[BoxMatch], list[Annotation], list[Annotation]]:
    """Greedily pair same-class boxes by best IoU above the threshold."""
    pairs: list[tuple[float, int, int]] = []
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            if a.category != b.category:
                continue
            iou = a.bbox.iou(b.bbox)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda p: -p[0])
    used_first, used_second = set(), set()
    matches: list[BoxMatch] = []
    for iou, i, j in pairs:
        if i in used_first or j in used_second:
            continue
        used_first.add(i)
        used_second.add(j)
        matches.append(BoxMatch(image_id=first[i].image_id, category=first[i].category, iou=round(iou, 3)))
    missed = [a for i, a in enumerate(first) if i not in used_first]
    extra = [b for j, b in enumerate(second) if j not in used_second]
    return matches, missed, extra


def box_agreement(first: Dataset, second: Dataset, iou_threshold: float = 0.5) -> BoxAgreement:
    """IoU-based box agreement between two datasets of the same images."""
    result = BoxAgreement()
    image_ids = sorted({img.id for img in first.images} | {img.id for img in second.images})
    for image_id in image_ids:
        a_boxes = first.annotations_for(image_id)
        b_boxes = second.annotations_for(image_id)
        matches, missed, extra = _greedy_match(a_boxes, b_boxes, iou_threshold)
        result.matches.extend(matches)
        result.matched += len(matches)
        result.missed += len(missed)
        result.extra += len(extra)
        for ann in missed:
            result.missed_boxes.append({"image_id": image_id, "category": ann.category, "bbox": ann.bbox.to_dict()})
        for ann in extra:
            result.extra_boxes.append({"image_id": image_id, "category": ann.category, "bbox": ann.bbox.to_dict()})
        total_a = len(a_boxes)
        result.per_image[image_id] = {
            "first_boxes": total_a,
            "second_boxes": len(b_boxes),
            "matched": len(matches),
            "missed": len(missed),
            "extra": len(extra),
            "match_rate": round(len(matches) / total_a, 3) if total_a else 0.0,
        }
    return result


@dataclass
class AnnotatorStats:
    annotator: str
    total_annotations: int
    images_annotated: int
    avg_boxes_per_image: float
    classes_used: list[str]
    category_counts: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "annotator": self.annotator,
            "total_annotations": self.total_annotations,
            "images_annotated": self.images_annotated,
            "avg_boxes_per_image": self.avg_boxes_per_image,
            "classes_used": self.classes_used,
            "category_counts": self.category_counts,
        }


def per_annotator_stats(datasets: list[Dataset]) -> list[AnnotatorStats]:
    """One summary per dataset (each dataset is one annotator's work)."""
    stats: list[AnnotatorStats] = []
    for ds in datasets:
        annotated_images = {a.image_id for a in ds.annotations}
        counts = ds.category_counts()
        total = len(ds.annotations)
        stats.append(
            AnnotatorStats(
                annotator=ds.annotator or ds.source_format or "unknown",
                total_annotations=total,
                images_annotated=len(annotated_images),
                avg_boxes_per_image=round(total / len(ds.images), 2) if ds.images else 0.0,
                classes_used=sorted(counts),
                category_counts=dict(sorted(counts.items())),
            )
        )
    return stats
