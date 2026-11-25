"""Tests for the core data model."""

from annotation_qa_toolkit.model import Annotation, BBox, Dataset, ImageRecord


def test_bbox_area_and_dimensions(box):
    assert box.width == 100
    assert box.height == 100
    assert box.area == 10000


def test_bbox_degenerate():
    assert BBox(10, 10, 10, 20).is_degenerate  # zero width
    assert BBox(10, 10, 20, 10).is_degenerate  # zero height
    assert BBox(20, 10, 10, 20).is_degenerate  # inverted
    assert not BBox(10, 10, 11, 11).is_degenerate


def test_bbox_iou_identical(box):
    assert box.iou(box) == 1.0


def test_bbox_iou_no_overlap():
    assert BBox(0, 0, 10, 10).iou(BBox(20, 20, 30, 30)) == 0.0


def test_bbox_iou_partial():
    # 10x10 boxes offset by 5 in x: overlap 5x10=50, union 100+100-50=150.
    assert BBox(0, 0, 10, 10).iou(BBox(5, 0, 15, 10)) == 50 / 150


def test_bbox_iou_touching_edges_is_zero():
    assert BBox(0, 0, 10, 10).iou(BBox(10, 0, 20, 10)) == 0.0


def test_dataset_helpers(simple_dataset):
    assert len(simple_dataset.annotations_for("a")) == 2
    assert simple_dataset.image_by_id("b").file_name == "b.jpg"
    assert simple_dataset.image_by_id("missing") is None
    assert simple_dataset.images_with_no_annotations() == []
    assert simple_dataset.category_counts() == {"car": 2, "person": 1}


def test_dataset_images_with_no_annotations(simple_dataset):
    simple_dataset.images.append(ImageRecord(id="c", file_name="c.jpg", width=100, height=100))
    empty = simple_dataset.images_with_no_annotations()
    assert [img.id for img in empty] == ["c"]


def test_annotation_to_dict_round_trip(box):
    ann = Annotation(image_id="a", category="car", bbox=box, annotator="ada")
    d = ann.to_dict()
    assert d["category"] == "car"
    assert d["bbox"]["x_max"] == 110
    assert d["annotator"] == "ada"
