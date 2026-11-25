"""Tests for annotator agreement."""

import pytest

from annotation_qa_toolkit.agreement import (
    box_agreement,
    cohen_kappa,
    per_annotator_stats,
)
from annotation_qa_toolkit.model import Annotation, BBox, Dataset, ImageRecord
from annotation_qa_toolkit.readers import load


def _ds(boxes_by_image, categories, annotator=""):
    images = [
        ImageRecord(id=i, file_name=f"{i}.jpg", width=200, height=200)
        for i in sorted(boxes_by_image)
    ]
    annotations = [
        Annotation(image_id=img, category=cat, bbox=box)
        for img, boxes in boxes_by_image.items()
        for cat, box in boxes
    ]
    return Dataset(images=images, annotations=annotations,
                   categories=categories, annotator=annotator)


def test_kappa_perfect_agreement():
    boxes = {"a": [("car", BBox(0, 0, 10, 10))], "b": [("person", BBox(0, 0, 10, 10))]}
    result = cohen_kappa(_ds(boxes, ["car", "person"]), _ds(boxes, ["car", "person"]))
    assert result.kappa == 1.0
    assert result.observed_agreement == 1.0
    assert result.n_decisions == 4  # 2 images x 2 classes


def test_kappa_no_agreement_beyond_chance():
    first = _ds({"a": [("car", BBox(0, 0, 10, 10))]}, ["car", "person"])
    second = _ds({"a": [("person", BBox(0, 0, 10, 10))]}, ["car", "person"])
    result = cohen_kappa(first, second)
    assert result.kappa < 0  # systematic disagreement is worse than chance
    assert len(result.disagreements) == 2


def test_kappa_partial_agreement():
    first = _ds({"a": [("car", BBox(0, 0, 10, 10)), ("person", BBox(0, 0, 10, 10))]},
                ["car", "person"])
    second = _ds({"a": [("car", BBox(0, 0, 10, 10))]}, ["car", "person"])
    result = cohen_kappa(first, second)
    assert result.n_decisions == 2
    assert result.observed_agreement == 0.5
    assert len(result.disagreements) == 1


def test_kappa_empty_datasets():
    empty = Dataset()
    result = cohen_kappa(empty, empty)
    assert result.kappa == 0.0 and result.n_decisions == 0


def test_box_agreement_identical():
    boxes = {"a": [("car", BBox(10, 10, 60, 60)), ("person", BBox(100, 100, 140, 180))]}
    result = box_agreement(_ds(boxes, ["car", "person"]), _ds(boxes, ["car", "person"]))
    assert result.matched == 2 and result.missed == 0 and result.extra == 0
    assert result.match_rate == 1.0 and result.mean_iou == 1.0


def test_box_agreement_missed_and_extra():
    first = _ds({"a": [("car", BBox(10, 10, 60, 60)), ("dog", BBox(100, 100, 150, 150))]},
                ["car", "dog"])
    second = _ds({"a": [("car", BBox(12, 12, 62, 62)), ("person", BBox(0, 0, 20, 20))]},
                 ["car", "person"])
    result = box_agreement(first, second)
    assert result.matched == 1 and result.missed == 1 and result.extra == 1
    assert result.match_rate == 0.5
    assert result.missed_boxes[0]["category"] == "dog"
    assert result.extra_boxes[0]["category"] == "person"


def test_box_agreement_below_threshold_is_not_a_match():
    first = _ds({"a": [("car", BBox(10, 10, 60, 60))]}, ["car"])
    second = _ds({"a": [("car", BBox(100, 100, 150, 150))]}, ["car"])
    result = box_agreement(first, second)
    assert result.matched == 0 and result.missed == 1 and result.extra == 1


def test_box_agreement_different_classes_do_not_match():
    first = _ds({"a": [("car", BBox(10, 10, 60, 60))]}, ["car"])
    second = _ds({"a": [("person", BBox(10, 10, 60, 60))]}, ["person"])
    result = box_agreement(first, second)
    assert result.matched == 0


def test_box_agreement_per_image(example_a_path, example_b_path):
    a = load(str(example_a_path))
    b = load(str(example_b_path))
    result = box_agreement(a, b)
    assert result.matched == 20
    assert result.missed == 4
    assert result.extra == 3
    assert result.match_rate == 0.833
    assert result.mean_iou == 0.86
    assert result.per_image["img_003"]["missed"] == 1  # B missed the dog
    # img_008: A's "Person" (capitalised) is a different class from B's
    # "person", so it is missed — and B's "person" plus the extra car
    # are both extra. The typo breaks box agreement, as it should.
    assert result.per_image["img_008"]["missed"] == 1
    assert result.per_image["img_008"]["extra"] == 2
    # img_005: the tiny dog box was shifted so far it no longer overlaps.
    assert result.per_image["img_005"]["missed"] == 1
    assert result.per_image["img_005"]["extra"] == 1


def test_kappa_example_data(example_a_path, example_b_path):
    a = load(str(example_a_path))
    b = load(str(example_b_path))
    result = cohen_kappa(a, b)
    assert result.kappa == 0.886
    assert result.observed_agreement == 0.95
    assert result.n_decisions == 60  # 10 images x 6 classes


def test_per_annotator_stats(example_a_path, example_b_path):
    a = load(str(example_a_path), annotator="ada")
    b = load(str(example_b_path), annotator="bola")
    stats = per_annotator_stats([a, b])
    assert stats[0].annotator == "ada"
    assert stats[0].total_annotations == 24
    assert stats[0].images_annotated == 9
    assert stats[0].avg_boxes_per_image == 2.4
    assert stats[1].total_annotations == 23
    assert "dog" in stats[0].classes_used
