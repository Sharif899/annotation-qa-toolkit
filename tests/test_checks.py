"""Tests for the QA checks."""

import pytest

from annotation_qa_toolkit.checks import (
    check_class_imbalance,
    check_degenerate_boxes,
    check_duplicate_boxes,
    check_empty_annotations,
    check_heavy_overlap,
    check_near_duplicate_classes,
    check_orphan_annotations,
    check_out_of_bounds,
    check_tiny_boxes,
    check_unknown_classes,
    run_all_checks,
)
from annotation_qa_toolkit.model import Annotation, BBox, Dataset, ImageRecord
from annotation_qa_toolkit.readers import load


def _ds(annotations, images=None, categories=None):
    images = images or [ImageRecord(id="a", file_name="a.jpg", width=200, height=200)]
    return Dataset(images=images, annotations=annotations,
                   categories=categories or ["car", "person"])


def test_empty_annotations(simple_dataset):
    assert check_empty_annotations(simple_dataset) == []
    simple_dataset.images.append(ImageRecord(id="c", file_name="c.jpg", width=100, height=100))
    findings = check_empty_annotations(simple_dataset)
    assert len(findings) == 1 and findings[0].image_id == "c"
    assert findings[0].severity == "warning"


def test_orphan_annotations():
    ds = _ds([Annotation(image_id="ghost", category="car", bbox=BBox(0, 0, 10, 10))])
    findings = check_orphan_annotations(ds)
    assert len(findings) == 1 and findings[0].severity == "error"


def test_out_of_bounds():
    ds = _ds([
        Annotation(image_id="a", category="car", bbox=BBox(150, 10, 250, 60)),  # past 200
        Annotation(image_id="a", category="car", bbox=BBox(10, 10, 50, 50)),   # fine
        Annotation(image_id="a", category="car", bbox=BBox(-5, 10, 50, 50)),   # negative
    ])
    assert len(check_out_of_bounds(ds)) == 2


def test_out_of_bounds_skips_unknown_image_size():
    img = ImageRecord(id="a", file_name="a.jpg", width=0, height=0)
    ds = _ds([Annotation(image_id="a", category="car", bbox=BBox(0, 0, 500, 500))], images=[img])
    assert check_out_of_bounds(ds) == []


def test_degenerate_boxes():
    ds = _ds([
        Annotation(image_id="a", category="car", bbox=BBox(10, 10, 10, 50)),
        Annotation(image_id="a", category="car", bbox=BBox(10, 10, 50, 10)),
        Annotation(image_id="a", category="car", bbox=BBox(50, 10, 10, 50)),
        Annotation(image_id="a", category="car", bbox=BBox(10, 10, 50, 50)),
    ])
    assert len(check_degenerate_boxes(ds)) == 3


def test_tiny_boxes_default_threshold():
    ds = _ds([
        Annotation(image_id="a", category="car", bbox=BBox(10, 10, 15, 16)),  # 5x6
        Annotation(image_id="a", category="car", bbox=BBox(10, 10, 30, 30)),  # fine
    ])
    findings = check_tiny_boxes(ds)
    assert len(findings) == 1
    assert "5x6" in findings[0].message


def test_tiny_boxes_custom_threshold():
    ds = _ds([Annotation(image_id="a", category="car", bbox=BBox(10, 10, 15, 16))])
    assert check_tiny_boxes(ds, min_side=4) == []
    assert len(check_tiny_boxes(ds, min_side=10)) == 1


def test_tiny_boxes_skips_degenerate():
    ds = _ds([Annotation(image_id="a", category="car", bbox=BBox(10, 10, 10, 10))])
    assert check_tiny_boxes(ds) == []  # degenerate is its own check


def test_duplicate_boxes():
    ds = _ds([
        Annotation(image_id="a", category="car", bbox=BBox(10, 10, 60, 60)),
        Annotation(image_id="a", category="car", bbox=BBox(10, 10, 60, 60)),
        Annotation(image_id="a", category="person", bbox=BBox(10, 10, 60, 60)),
    ])
    findings = check_duplicate_boxes(ds)
    assert len(findings) == 1  # same class only; the person box is not a duplicate
    assert "IoU 1.00" in findings[0].message


def test_duplicate_boxes_threshold():
    ds = _ds([
        Annotation(image_id="a", category="car", bbox=BBox(10, 10, 60, 60)),
        Annotation(image_id="a", category="car", bbox=BBox(12, 12, 62, 62)),  # IoU ~0.86
    ])
    assert check_duplicate_boxes(ds) == []
    assert len(check_duplicate_boxes(ds, iou_threshold=0.8)) == 1


def test_heavy_overlap_but_not_duplicate():
    ds = _ds([
        Annotation(image_id="a", category="car", bbox=BBox(10, 10, 60, 60)),
        Annotation(image_id="a", category="car", bbox=BBox(30, 30, 80, 80)),  # IoU ~0.22
    ])
    assert check_heavy_overlap(ds) == []
    ds2 = _ds([
        Annotation(image_id="a", category="car", bbox=BBox(10, 10, 60, 60)),
        Annotation(image_id="a", category="car", bbox=BBox(20, 15, 70, 65)),  # IoU ~0.62
    ])
    assert len(check_heavy_overlap(ds2)) == 1


def test_unknown_classes():
    ds = _ds([Annotation(image_id="a", category="truck", bbox=BBox(10, 10, 50, 50))])
    findings = check_unknown_classes(ds)
    assert len(findings) == 1 and findings[0].severity == "error"


def test_near_duplicate_classes_in_list():
    ds = _ds([], categories=["car", "Car", "person"])
    findings = check_near_duplicate_classes(ds)
    assert len(findings) == 1
    assert "Car" in findings[0].message and "car" in findings[0].message


def test_near_duplicate_classes_plural():
    ds = _ds([], categories=["car", "cars"])
    assert len(check_near_duplicate_classes(ds)) == 1


def test_near_duplicate_classes_annotation_vs_list():
    ds = _ds(
        [Annotation(image_id="a", category="Cars", bbox=BBox(10, 10, 50, 50))],
        categories=["car", "person"],
    )
    findings = check_near_duplicate_classes(ds)
    assert len(findings) == 1
    assert findings[0].image_id == "a"


def test_near_duplicate_classes_clean():
    ds = _ds([], categories=["car", "person", "dog"])
    assert check_near_duplicate_classes(ds) == []


def test_class_imbalance_rare():
    ds = _ds(
        [Annotation(image_id="a", category="car", bbox=BBox(10, 10, 50, 50))] * 98
        + [Annotation(image_id="a", category="dog", bbox=BBox(10, 10, 50, 50))],
        categories=["car", "dog"],
    )
    findings = check_class_imbalance(ds, rare_share=0.02)
    assert len(findings) == 1
    assert findings[0].check == "rare_class" and findings[0].severity == "warning"
    assert "'dog'" in findings[0].message


def test_class_imbalance_balanced_is_clean():
    ds = _ds(
        [Annotation(image_id="a", category="car", bbox=BBox(10, 10, 50, 50))] * 5
        + [Annotation(image_id="a", category="dog", bbox=BBox(10, 10, 50, 50))] * 5,
        categories=["car", "dog"],
    )
    assert check_class_imbalance(ds, rare_share=0.02) == []


def test_class_imbalance_empty_dataset():
    assert check_class_imbalance(_ds([])) == []


def test_run_all_checks_clean(simple_dataset):
    assert run_all_checks(simple_dataset) == []


def test_run_all_checks_example_data(example_a_path):
    ds = load(str(example_a_path))
    findings = run_all_checks(ds)
    by_check = {f.check for f in findings}
    assert by_check == {
        "out_of_bounds", "duplicate_box", "heavy_overlap", "tiny_box",
        "empty_annotations", "near_duplicate_class",
    }
    assert len(findings) == 7
    assert sum(1 for f in findings if f.severity == "error") == 1
    assert sum(1 for f in findings if f.severity == "warning") == 6
    assert sum(1 for f in findings if f.severity == "info") == 0


def test_run_all_checks_is_deterministic(example_a_path):
    ds = load(str(example_a_path))
    assert run_all_checks(ds) == run_all_checks(ds)
