"""Core data model.

Every reader normalises its format into this one model, so the QA
checks, the agreement analysis and the report never need to know
which tool produced the annotations.

Coordinates are absolute pixels: ``x_min``/``y_min`` is the top-left
corner, ``x_max``/``y_max`` the bottom-right corner. A box is valid
only when ``x_min < x_max`` and ``y_min < y_max``; anything else is a
finding for the QA checks, not a reason to crash the reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BBox:
    """One axis-aligned bounding box, in absolute pixels."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        w, h = self.width, self.height
        return w * h if w > 0 and h > 0 else 0.0

    @property
    def is_degenerate(self) -> bool:
        """True when the box has no area (zero width/height or inverted)."""
        return self.width <= 0 or self.height <= 0

    def iou(self, other: "BBox") -> float:
        """Intersection over union, 0.0 when the boxes do not overlap."""
        ix_min = max(self.x_min, other.x_min)
        iy_min = max(self.y_min, other.y_min)
        ix_max = min(self.x_max, other.x_max)
        iy_max = min(self.y_max, other.y_max)
        iw, ih = ix_max - ix_min, iy_max - iy_min
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }


@dataclass(frozen=True)
class ImageRecord:
    """One image in a dataset. No image bytes are stored or needed."""

    id: str
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class Annotation:
    """One labelled box on one image."""

    image_id: str
    category: str
    bbox: BBox
    annotator: str = ""
    attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "category": self.category,
            "bbox": self.bbox.to_dict(),
            "annotator": self.annotator,
            "attributes": dict(self.attributes),
        }


@dataclass
class Dataset:
    """A normalised annotation dataset, from any supported format."""

    images: list[ImageRecord] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    annotator: str = ""
    source_format: str = ""

    def image_by_id(self, image_id: str) -> ImageRecord | None:
        for image in self.images:
            if image.id == image_id:
                return image
        return None

    def annotations_for(self, image_id: str) -> list[Annotation]:
        return [a for a in self.annotations if a.image_id == image_id]

    def images_with_no_annotations(self) -> list[ImageRecord]:
        annotated = {a.image_id for a in self.annotations}
        return [img for img in self.images if img.id not in annotated]

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ann in self.annotations:
            counts[ann.category] = counts.get(ann.category, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "images": [vars(img) for img in self.images],
            "annotations": [a.to_dict() for a in self.annotations],
            "categories": list(self.categories),
            "annotator": self.annotator,
            "source_format": self.source_format,
        }
