"""Tests for the format readers."""

import json

import pytest

from annotation_qa_toolkit.readers import (
    AnnotationReadError,
    detect_format,
    load,
    read_coco,
    read_labelbox,
    read_roboflow,
    read_superannotate,
    read_voc,
    read_yolo,
)
from tests.conftest import write_json

COCO = {
    "images": [
        {"id": 1, "file_name": "a.jpg", "width": 200, "height": 200},
        {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100},
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50]},
        {"id": 2, "image_id": 2, "category_id": 2, "bbox": [5, 5, 20, 30]},
    ],
    "categories": [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "person"},
    ],
}


def test_read_coco(tmp_path):
    path = write_json(tmp_path, "data.json", COCO)
    ds = read_coco(str(path), annotator="ada")
    assert ds.source_format == "coco"
    assert ds.annotator == "ada"
    assert len(ds.images) == 2
    assert len(ds.annotations) == 2
    assert ds.categories == ["car", "person"]
    box = ds.annotations[0].bbox
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (10, 10, 60, 60)


def test_read_coco_missing_keys(tmp_path):
    path = write_json(tmp_path, "bad.json", {"images": []})
    with pytest.raises(AnnotationReadError, match="not a COCO file"):
        read_coco(str(path))


def test_read_coco_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(AnnotationReadError, match="invalid JSON"):
        read_coco(str(path))


def test_read_coco_unknown_category_id(tmp_path):
    data = json.loads(json.dumps(COCO))
    data["annotations"][0]["category_id"] = 99
    path = write_json(tmp_path, "data.json", data)
    ds = read_coco(str(path))
    assert ds.annotations[0].category == "99"


def test_read_roboflow_marks_format(tmp_path):
    data = json.loads(json.dumps(COCO))
    data["roboflow"] = {"version": "1.0"}
    path = write_json(tmp_path, "data.json", data)
    ds = read_roboflow(str(path))
    assert ds.source_format == "roboflow"
    assert len(ds.annotations) == 2


def test_read_yolo(tmp_path):
    (tmp_path / "classes.txt").write_text("car\nperson\n", encoding="utf-8")
    (tmp_path / "sizes.json").write_text(
        json.dumps({"a.jpg": [200, 200], "b.jpg": [100, 100]}), encoding="utf-8")
    (tmp_path / "a.txt").write_text("0 0.25 0.25 0.5 0.5\n1 0.75 0.75 0.2 0.4\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("", encoding="utf-8")
    ds = read_yolo(str(tmp_path))
    assert ds.source_format == "yolo"
    assert len(ds.images) == 2
    assert len(ds.annotations) == 2
    box = ds.annotations[0].bbox
    # centre (50, 50), size 100x100 on a 200x200 image -> (0, 0, 100, 100).
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (0, 0, 100, 100)
    assert ds.annotations[1].category == "person"


def test_read_yolo_needs_classes(tmp_path):
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    with pytest.raises(AnnotationReadError, match="class list"):
        read_yolo(str(tmp_path))


def test_read_yolo_needs_sizes(tmp_path):
    (tmp_path / "classes.txt").write_text("car\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    with pytest.raises(AnnotationReadError, match="image sizes"):
        read_yolo(str(tmp_path))


def test_read_yolo_bad_line(tmp_path):
    (tmp_path / "classes.txt").write_text("car\n", encoding="utf-8")
    (tmp_path / "sizes.json").write_text(json.dumps({"a.jpg": [100, 100]}), encoding="utf-8")
    (tmp_path / "a.txt").write_text("0 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(AnnotationReadError, match="expected 5 values"):
        read_yolo(str(tmp_path))


def test_read_yolo_class_index_out_of_range(tmp_path):
    (tmp_path / "classes.txt").write_text("car\n", encoding="utf-8")
    (tmp_path / "sizes.json").write_text(json.dumps({"a.jpg": [100, 100]}), encoding="utf-8")
    (tmp_path / "a.txt").write_text("3 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    with pytest.raises(AnnotationReadError, match="out of range"):
        read_yolo(str(tmp_path))


def test_read_voc(tmp_path):
    xml = """<?xml version="1.0"?>
<annotation>
  <filename>a.jpg</filename>
  <size><width>200</width><height>150</height><depth>3</depth></size>
  <object><name>car</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>80</xmax><ymax>90</ymax></bndbox></object>
</annotation>
"""
    path = tmp_path / "a.xml"
    path.write_text(xml, encoding="utf-8")
    ds = read_voc(str(path))
    assert ds.source_format == "voc"
    assert len(ds.images) == 1 and len(ds.annotations) == 1
    assert ds.annotations[0].bbox.x_max == 80
    assert ds.images[0].width == 200


def test_read_voc_bad_xml(tmp_path):
    path = tmp_path / "bad.xml"
    path.write_text("<annotation><unclosed>", encoding="utf-8")
    with pytest.raises(AnnotationReadError, match="invalid XML"):
        read_voc(str(path))


def test_read_voc_directory(tmp_path, example_a_path):
    ds = load(str(example_a_path.parent / "voc"))
    assert ds.source_format == "voc"
    assert len(ds.images) == 2


LABELBOX = [
    {
        "data_row": {"id": "row1", "external_id": "a.jpg", "width": 200, "height": 200},
        "projects": {
            "p1": {
                "labels": [
                    {"annotations": {"objects": [
                        {"name": "car", "annotation_kind": "ImageBoundingBox",
                         "bounding_box": {"left": 10, "top": 20, "width": 50, "height": 40}},
                        {"name": "note", "annotation_kind": "Classification",
                         "bounding_box": {}},
                    ]}}
                ]
            }
        },
    }
]


def test_read_labelbox(tmp_path):
    path = write_json(tmp_path, "lb.json", LABELBOX)
    ds = read_labelbox(str(path))
    assert ds.source_format == "labelbox"
    assert len(ds.annotations) == 1  # the Classification entry is skipped
    box = ds.annotations[0].bbox
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (10, 20, 60, 60)


def test_read_labelbox_not_a_list(tmp_path):
    path = write_json(tmp_path, "lb.json", {"data_row": {}})
    with pytest.raises(AnnotationReadError, match="must be a JSON list"):
        read_labelbox(str(path))


SUPERANNOTATE = {
    "metadata": {"name": "a.jpg", "width": 200, "height": 200},
    "instances": [
        {"type": "bbox", "className": "car", "classId": 1,
         "points": {"x1": 10, "y1": 20, "x2": 80, "y2": 90}},
        {"type": "point", "className": "car", "points": {"x": 5, "y": 5}},
    ],
}


def test_read_superannotate(tmp_path):
    path = write_json(tmp_path, "a.json", SUPERANNOTATE)
    ds = read_superannotate(str(path))
    assert ds.source_format == "superannotate"
    assert len(ds.annotations) == 1  # the point instance is skipped
    assert ds.annotations[0].bbox.x_max == 80


def test_read_superannotate_not_sa_file(tmp_path):
    path = write_json(tmp_path, "a.json", {"foo": 1})
    with pytest.raises(AnnotationReadError, match="not a SuperAnnotate file"):
        read_superannotate(str(path))


def test_detect_format_coco(tmp_path):
    path = write_json(tmp_path, "data.json", COCO)
    assert detect_format(str(path)) == "coco"


def test_detect_format_roboflow(tmp_path):
    data = json.loads(json.dumps(COCO))
    data["roboflow"] = {}
    path = write_json(tmp_path, "data.json", data)
    assert detect_format(str(path)) == "roboflow"


def test_detect_format_labelbox(tmp_path):
    path = write_json(tmp_path, "lb.json", LABELBOX)
    assert detect_format(str(path)) == "labelbox"


def test_detect_format_superannotate(tmp_path):
    path = write_json(tmp_path, "a.json", SUPERANNOTATE)
    assert detect_format(str(path)) == "superannotate"


def test_detect_format_yolo_dir(tmp_path):
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    assert detect_format(str(tmp_path)) == "yolo"


def test_detect_format_unknown_extension(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(AnnotationReadError, match="cannot detect format"):
        detect_format(str(path))


def test_load_example_files(example_a_path, example_b_path):
    a = load(str(example_a_path))
    b = load(str(example_b_path))
    assert len(a.images) == 10 and len(a.annotations) == 24
    assert len(b.annotations) == 23
