"""Tests for the CLI. The console script is exercised through main()."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from annotation_qa_toolkit.cli import main

ROOT = Path(__file__).resolve().parent.parent
VOC_CLEAN = ROOT / "data" / "examples" / "voc" / "img_001.xml"


def run_cli(*argv):
    return main(list(argv))


def test_load_command(capsys, example_a_path):
    assert run_cli("load", str(example_a_path)) == 0
    out = capsys.readouterr().out
    assert "Images: 10" in out
    assert "Annotations: 24" in out
    assert "car" in out


def test_load_json(capsys, example_a_path):
    assert run_cli("load", str(example_a_path), "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["images"]) == 10
    assert data["source_format"] == "coco"


def test_load_missing_file(capsys):
    assert run_cli("load", "no-such-file.json") == 2
    assert "error" in capsys.readouterr().err


def test_load_stdin(capsys, example_a_path, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(example_a_path.read_text(encoding="utf-8")))
    assert run_cli("load", "-") == 0
    assert "Images: 10" in capsys.readouterr().out


def test_load_stdin_invalid_json(capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("{broken"))
    assert run_cli("load", "-") == 2


def test_load_stdin_empty(capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert run_cli("load", "-") == 2


def test_check_command_clean(capsys):
    assert run_cli("check", str(VOC_CLEAN)) == 0
    assert "No findings" in capsys.readouterr().out


def test_check_command_finds_errors(capsys, example_a_path):
    assert run_cli("check", str(example_a_path)) == 1  # errors present -> exit 1
    out = capsys.readouterr().out
    assert "out_of_bounds" in out
    assert "duplicate_box" in out


def test_check_json(capsys, example_a_path):
    assert run_cli("check", str(example_a_path), "--json") == 1
    findings = json.loads(capsys.readouterr().out)
    assert len(findings) == 7
    assert findings[0]["severity"] in ("error", "info", "warning")


def test_check_format_flag(capsys, example_a_path):
    assert run_cli("check", str(example_a_path), "--format", "coco") == 1


def test_agreement_command(capsys, example_a_path, example_b_path):
    code = run_cli("agreement", str(example_a_path), str(example_b_path),
                   "--first-annotator", "ada", "--second-annotator", "bola")
    assert code == 0
    out = capsys.readouterr().out
    assert "Cohen's kappa (class presence): 0.886" in out
    assert "matched 20" in out
    assert "ada: 24 annotations" in out


def test_agreement_json(capsys, example_a_path, example_b_path):
    assert run_cli("agreement", str(example_a_path), str(example_b_path), "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["kappa"]["kappa"] == 0.886
    assert data["box_agreement"]["matched"] == 20
    assert len(data["annotators"]) == 2


def test_report_command(capsys, example_a_path):
    assert run_cli("report", str(example_a_path)) == 1  # errors present
    out = capsys.readouterr().out
    assert "Annotation QA report" in out
    assert "Worst images" in out
    assert "img_004" in out


def test_report_json(capsys, example_a_path):
    assert run_cli("report", str(example_a_path), "--json") == 1
    data = json.loads(capsys.readouterr().out)
    assert data["total_images"] == 10
    assert data["image_scores"]["img_004"] == 85


def test_report_clean_exits_zero(capsys):
    assert run_cli("report", str(VOC_CLEAN)) == 0


def test_console_script_entrypoint(example_a_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-m", "annotation_qa_toolkit.cli", "load", str(example_a_path)],
        capture_output=True, text=True, cwd=str(ROOT), env=env,
    )
    assert result.returncode == 0
    assert "Images: 10" in result.stdout
