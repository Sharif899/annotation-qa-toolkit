"""Readers: one per annotation format, all normalising to ``Dataset``.

Supported formats:

* **COCO** — ``.json`` with ``images`` / ``annotations`` / ``categories``.
* **Roboflow** — a COCO-style export; same reader, ``source_format``
  recorded as ``roboflow`` when the file carries the ``roboflow`` key
  or the caller says so.
* **YOLO** — a directory of ``.txt`` files (one per image) plus a
  ``classes.txt``; boxes are normalised centre-x/centre-y/width/height
  and need the image size, so a ``sizes.json`` mapping file name to
  ``[width, height]`` (or a ``--sizes`` equivalent) is required. The
  reader also accepts a single ``.txt`` file with ``--sizes``.
* **Pascal VOC** — ``.xml`` files (one per image) in a directory or a
  single file.
* **Labelbox** — a JSON export: a list of rows, each with ``data_row``
  (``external_id`` / ``id``) and ``projects`` holding ``labels`` with
  ``annotations`` of type ``bounding_box``.
* **SuperAnnotate** — one JSON file per image, ``instances`` with
  ``type: "bbox"`` and ``points`` ``{x1, y1, x2, y2}``; the file name
  (``imageName`` or the JSON file's own stem) identifies the image.

Every reader is strict about malformed input: a bad file raises
``AnnotationReadError`` naming the file, instead of silently dropping
annotations — a QA tool that hides problems is worse than none.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET

from .model import Annotation, BBox, Dataset, ImageRecord

FORMATS = ("coco", "roboflow", "yolo", "voc", "labelbox", "superannotate")


class AnnotationReadError(ValueError):
    """Raised when an annotation file cannot be parsed."""


def _load_json(path: str) -> object:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise AnnotationReadError(f"{path}: invalid JSON ({exc})") from exc
    except OSError as exc:
        raise AnnotationReadError(f"{path}: cannot read file ({exc})") from exc


def _bbox(x_min, y_min, x_max, y_max) -> BBox:
    return BBox(float(x_min), float(y_min), float(x_max), float(y_max))


def read_coco(path: str, annotator: str = "") -> Dataset:
    """Read a COCO JSON file. ``source_format`` is ``coco``."""
    data = _load_json(path)
    return _coco_dict(data, path, annotator=annotator, source_format="coco")


def read_roboflow(path: str, annotator: str = "") -> Dataset:
    """Read a Roboflow export (COCO-style JSON)."""
    data = _load_json(path)
    return _coco_dict(data, path, annotator=annotator, source_format="roboflow")


def _coco_dict(data: object, path: str, annotator: str, source_format: str) -> Dataset:
    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        raise AnnotationReadError(
            f"{path}: not a COCO file (expected 'images' and 'annotations')"
        )
    categories: dict[int, str] = {}
    for cat in data.get("categories", []):
        categories[cat["id"]] = str(cat["name"])
    images = [
        ImageRecord(
            id=str(img["id"]),
            file_name=str(img.get("file_name", img["id"])),
            width=int(img.get("width", 0)),
            height=int(img.get("height", 0)),
        )
        for img in data["images"]
    ]
    annotations = []
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        annotations.append(
            Annotation(
                image_id=str(ann["image_id"]),
                category=categories.get(ann["category_id"], str(ann["category_id"])),
                bbox=_bbox(x, y, x + w, y + h),
                annotator=annotator,
                attributes={k: v for k, v in ann.items() if k not in
                            ("id", "image_id", "category_id", "bbox")},
            )
        )
    return Dataset(
        images=images,
        annotations=annotations,
        categories=sorted(set(categories.values())),
        annotator=annotator,
        source_format=source_format,
    )


def read_yolo(
    path: str,
    classes: list[str] | None = None,
    sizes: dict[str, tuple[int, int]] | None = None,
    annotator: str = "",
) -> Dataset:
    """Read YOLO annotations.

    ``path`` is a directory of ``.txt`` files or a single ``.txt`` file.
    ``classes`` is the class list (from ``classes.txt``); ``sizes`` maps
    an image file name (or the txt stem) to ``(width, height)`` — YOLO
    boxes are normalised, so without sizes the reader cannot place
    them and raises ``AnnotationReadError``.
    """
    if classes is None:
        classes_path = os.path.join(path, "classes.txt") if os.path.isdir(path) else None
        if classes_path and os.path.exists(classes_path):
            with open(classes_path, encoding="utf-8") as fh:
                classes = [line.strip() for line in fh if line.strip()]
        else:
            raise AnnotationReadError(
                f"{path}: YOLO needs a class list (pass classes= or provide classes.txt)"
            )
    if sizes is None:
        sizes_path = os.path.join(path, "sizes.json") if os.path.isdir(path) else None
        if sizes_path and os.path.exists(sizes_path):
            raw = _load_json(sizes_path)
            if not isinstance(raw, dict):
                raise AnnotationReadError(f"{sizes_path}: expected an object")
            sizes = {k: (int(v[0]), int(v[1])) for k, v in raw.items()}
        else:
            raise AnnotationReadError(
                f"{path}: YOLO needs image sizes (pass sizes= or provide sizes.json)"
            )
    txt_files = (
        sorted(f for f in os.listdir(path) if f.endswith(".txt") and f != "classes.txt")
        if os.path.isdir(path)
        else [os.path.basename(path)]
    )
    base = path if os.path.isdir(path) else os.path.dirname(path)
    images: list[ImageRecord] = []
    annotations: list[Annotation] = []
    for txt in txt_files:
        stem = os.path.splitext(txt)[0]
        image_name = next(
            (name for name in (f"{stem}.jpg", f"{stem}.png", f"{stem}.jpeg", stem)
             if name in sizes),
            None,
        )
        if image_name is None:
            raise AnnotationReadError(
                f"{os.path.join(base, txt)}: no size known for image '{stem}'"
            )
        width, height = sizes[image_name]
        images.append(ImageRecord(id=stem, file_name=image_name, width=width, height=height))
        try:
            with open(os.path.join(base, txt), encoding="utf-8") as fh:
                lines = [line.strip() for line in fh if line.strip()]
        except OSError as exc:
            raise AnnotationReadError(f"{os.path.join(base, txt)}: cannot read ({exc})") from exc
        for lineno, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                raise AnnotationReadError(
                    f"{os.path.join(base, txt)}:{lineno}: expected 5 values, got {len(parts)}"
                )
            try:
                cls_idx, cx, cy, w, h = int(parts[0]), *(float(p) for p in parts[1:])
            except ValueError as exc:
                raise AnnotationReadError(
                    f"{os.path.join(base, txt)}:{lineno}: non-numeric value"
                ) from exc
            if not 0 <= cls_idx < len(classes):
                raise AnnotationReadError(
                    f"{os.path.join(base, txt)}:{lineno}: class index {cls_idx} "
                    f"out of range ({len(classes)} classes)"
                )
            annotations.append(
                Annotation(
                    image_id=stem,
                    category=classes[cls_idx],
                    bbox=_bbox(
                        (cx - w / 2) * width, (cy - h / 2) * height,
                        (cx + w / 2) * width, (cy + h / 2) * height,
                    ),
                    annotator=annotator,
                )
            )
    return Dataset(
        images=images,
        annotations=annotations,
        categories=list(classes),
        annotator=annotator,
        source_format="yolo",
    )


def read_voc(path: str, annotator: str = "") -> Dataset:
    """Read Pascal VOC XML files: a directory, or a single ``.xml``."""
    if os.path.isdir(path):
        xml_files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".xml"))
    else:
        xml_files = [path]
    if not xml_files:
        raise AnnotationReadError(f"{path}: no .xml files found")
    images: list[ImageRecord] = []
    annotations: list[Annotation] = []
    categories: set[str] = set()
    for xml_file in xml_files:
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError as exc:
            raise AnnotationReadError(f"{xml_file}: invalid XML ({exc})") from exc
        filename = root.findtext("filename") or os.path.splitext(os.path.basename(xml_file))[0]
        size_el = root.find("size")
        width = int(size_el.findtext("width", "0")) if size_el is not None else 0
        height = int(size_el.findtext("height", "0")) if size_el is not None else 0
        image_id = os.path.splitext(filename)[0]
        images.append(ImageRecord(id=image_id, file_name=filename, width=width, height=height))
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            bnd = obj.find("bndbox")
            if bnd is None:
                raise AnnotationReadError(f"{xml_file}: object '{name}' has no bndbox")
            categories.add(name)
            annotations.append(
                Annotation(
                    image_id=image_id,
                    category=name,
                    bbox=_bbox(
                        bnd.findtext("xmin"), bnd.findtext("ymin"),
                        bnd.findtext("xmax"), bnd.findtext("ymax"),
                    ),
                    annotator=annotator,
                )
            )
    return Dataset(
        images=images,
        annotations=annotations,
        categories=sorted(categories),
        annotator=annotator,
        source_format="voc",
    )


def read_labelbox(path: str, annotator: str = "") -> Dataset:
    """Read a Labelbox JSON export (a list of data rows)."""
    data = _load_json(path)
    if not isinstance(data, list):
        raise AnnotationReadError(f"{path}: Labelbox export must be a JSON list")
    images: list[ImageRecord] = []
    annotations: list[Annotation] = []
    categories: set[str] = set()
    for row in data:
        if not isinstance(row, dict) or "data_row" not in row:
            raise AnnotationReadError(f"{path}: Labelbox row missing 'data_row'")
        data_row = row["data_row"]
        image_id = str(data_row.get("id") or data_row.get("external_id") or "")
        file_name = str(data_row.get("external_id") or data_row.get("row_data") or image_id)
        width = int(data_row.get("width") or row.get("media_attributes", {}).get("width") or 0)
        height = int(data_row.get("height") or row.get("media_attributes", {}).get("height") or 0)
        images.append(ImageRecord(id=image_id, file_name=file_name, width=width, height=height))
        for project in row.get("projects", {}).values():
            for label in project.get("labels", []):
                for ann in label.get("annotations", {}).get("objects", []):
                    if ann.get("annotation_kind") != "ImageBoundingBox":
                        continue
                    box = ann["bounding_box"]
                    categories.add(str(ann["name"]))
                    annotations.append(
                        Annotation(
                            image_id=image_id,
                            category=str(ann["name"]),
                            bbox=_bbox(
                                box["left"], box["top"],
                                box["left"] + box["width"], box["top"] + box["height"],
                            ),
                            annotator=annotator,
                        )
                    )
    return Dataset(
        images=images,
        annotations=annotations,
        categories=sorted(categories),
        annotator=annotator,
        source_format="labelbox",
    )


def read_superannotate(path: str, annotator: str = "") -> Dataset:
    """Read SuperAnnotate JSON files: a directory, or a single file.

    Each file describes one image: ``instances`` with ``type: "bbox"``
    and ``points`` ``{"x1","y1","x2","y2"}``. Image size comes from
    ``metadata`` when present, else the file name is used as the id.
    """
    if os.path.isdir(path):
        json_files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json"))
    else:
        json_files = [path]
    if not json_files:
        raise AnnotationReadError(f"{path}: no .json files found")
    images: list[ImageRecord] = []
    annotations: list[Annotation] = []
    categories: set[str] = set()
    for json_file in json_files:
        data = _load_json(json_file)
        if not isinstance(data, dict) or "instances" not in data:
            raise AnnotationReadError(
                f"{json_file}: not a SuperAnnotate file (expected 'instances')"
            )
        metadata = data.get("metadata", {}) or {}
        file_name = str(metadata.get("name") or data.get("imageName")
                        or os.path.splitext(os.path.basename(json_file))[0])
        image_id = os.path.splitext(file_name)[0]
        width = int(metadata.get("width") or 0)
        height = int(metadata.get("height") or 0)
        images.append(ImageRecord(id=image_id, file_name=file_name, width=width, height=height))
        for inst in data["instances"]:
            if inst.get("type") != "bbox":
                continue
            points = inst.get("points", {})
            class_name = str(inst.get("className") or inst.get("class") or "")
            categories.add(class_name)
            annotations.append(
                Annotation(
                    image_id=image_id,
                    category=class_name,
                    bbox=_bbox(points["x1"], points["y1"], points["x2"], points["y2"]),
                    annotator=annotator,
                    attributes={"class_id": inst.get("classId")},
                )
            )
    return Dataset(
        images=images,
        annotations=annotations,
        categories=sorted(categories),
        annotator=annotator,
        source_format="superannotate",
    )


_READERS = {
    "coco": read_coco,
    "roboflow": read_roboflow,
    "yolo": read_yolo,
    "voc": read_voc,
    "labelbox": read_labelbox,
    "superannotate": read_superannotate,
}


def detect_format(path: str) -> str:
    """Best-effort format detection from path and file contents."""
    if os.path.isdir(path):
        entries = os.listdir(path)
        if any(e.endswith(".txt") for e in entries):
            return "yolo"
        if any(e.endswith(".xml") for e in entries):
            return "voc"
        if any(e.endswith(".json") for e in entries):
            return "superannotate"
        raise AnnotationReadError(f"{path}: cannot detect format from directory contents")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xml":
        return "voc"
    if ext == ".txt":
        return "yolo"
    if ext == ".json":
        data = _load_json(path)
        if isinstance(data, list):
            return "labelbox"
        if isinstance(data, dict):
            if "instances" in data:
                return "superannotate"
            if "images" in data and "annotations" in data:
                return "roboflow" if "roboflow" in data else "coco"
        raise AnnotationReadError(f"{path}: unrecognised JSON annotation format")
    raise AnnotationReadError(f"{path}: cannot detect format (unknown extension '{ext}')")


def load(path: str, fmt: str | None = None, annotator: str = "", **kwargs) -> Dataset:
    """Load an annotation file or directory, auto-detecting the format."""
    fmt = fmt or detect_format(path)
    if fmt not in _READERS:
        raise AnnotationReadError(f"unknown format '{fmt}' (expected one of {', '.join(FORMATS)})")
    return _READERS[fmt](path, annotator=annotator, **kwargs)
