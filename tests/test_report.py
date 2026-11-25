"""Tests for the quality report."""

from annotation_qa_toolkit.checks import Finding, run_all_checks
from annotation_qa_toolkit.readers import load
from annotation_qa_toolkit.report import build_report, render_text


def test_build_report_clean(simple_dataset):
    report = build_report(simple_dataset, [])
    assert report.total_images == 2
    assert report.total_annotations == 3
    assert report.findings == []
    assert report.by_severity == {}
    assert report.worst_images == []
    assert all(score == 100 for score in report.image_scores.values())


def test_build_report_scoring():
    findings = [
        Finding("error", "out_of_bounds", "a", "box out of bounds"),
        Finding("warning", "tiny_box", "a", "tiny box"),
        Finding("warning", "duplicate_box", "b", "duplicate"),
        Finding("info", "class_imbalance", "", "counts"),
    ]
    from annotation_qa_toolkit.model import Dataset, ImageRecord
    ds = Dataset(images=[
        ImageRecord(id="a", file_name="a.jpg", width=100, height=100),
        ImageRecord(id="b", file_name="b.jpg", width=100, height=100),
    ])
    report = build_report(ds, findings)
    assert report.image_scores["a"] == 80   # 100 - 15 - 5
    assert report.image_scores["b"] == 95   # 100 - 5
    assert report.worst_images[0]["image_id"] == "a"
    assert report.by_severity == {"error": 1, "warning": 2, "info": 1}
    assert report.by_check["tiny_box"] == 1


def test_build_report_score_floors_at_zero():
    findings = [Finding("error", "out_of_bounds", "a", f"error {i}") for i in range(10)]
    from annotation_qa_toolkit.model import Dataset, ImageRecord
    ds = Dataset(images=[ImageRecord(id="a", file_name="a.jpg", width=100, height=100)])
    report = build_report(ds, findings)
    assert report.image_scores["a"] == 0


def test_render_text_clean(simple_dataset):
    text = render_text(build_report(simple_dataset, []))
    assert "Images: 2" in text
    assert "no findings" in text.lower()
    assert "every image scored 100" in text


def test_render_text_example_data(example_a_path):
    ds = load(str(example_a_path))
    report = build_report(ds, run_all_checks(ds), dataset_name="annotator_a")
    text = render_text(report)
    assert "Findings: 7" in text
    assert "errors: 1" in text
    assert "img_004" in text  # worst image first
    assert "score  85" in text
    assert "duplicate_box" in text


def test_report_to_dict_round_trip(simple_dataset):
    report = build_report(simple_dataset, [])
    d = report.to_dict()
    assert d["total_images"] == 2
    assert d["image_scores"] == {"a": 100, "b": 100}
