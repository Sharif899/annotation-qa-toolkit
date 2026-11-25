"""Shared fixtures for the annotation QA tests."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from annotation_qa_toolkit.model import Annotation, BBox, Dataset, ImageRecord


@pytest.fixture
def box():
    return BBox(10, 20, 110, 120)


@pytest.fixture
def simple_dataset():
    """Two images, a few clean boxes — the happy path."""
    images = [
        ImageRecord(id="a", file_name="a.jpg", width=200, height=200),
        ImageRecord(id="b", file_name="b.jpg", width=200, height=200),
    ]
    annotations = [
        Annotation(image_id="a", category="car", bbox=BBox(10, 10, 60, 60)),
        Annotation(image_id="a", category="person", bbox=BBox(100, 100, 140, 180)),
        Annotation(image_id="b", category="car", bbox=BBox(20, 20, 90, 90)),
    ]
    return Dataset(images=images, annotations=annotations,
                   categories=["car", "person"], source_format="test")


@pytest.fixture
def example_a_path():
    return Path(__file__).resolve().parent.parent / "data" / "examples" / "coco_annotator_a.json"


@pytest.fixture
def example_b_path():
    return Path(__file__).resolve().parent.parent / "data" / "examples" / "coco_annotator_b.json"


def write_json(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
