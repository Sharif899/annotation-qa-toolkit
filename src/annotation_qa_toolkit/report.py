"""Quality report.

Aggregates findings into a report a QA lead can act on: totals by
severity and by check, a per-image quality score, and the worst
images first.

Quality score (per image, documented so it can be argued with):

    score = 100 - 15 * errors - 5 * warnings, floored at 0

An ``error`` (out-of-bounds box, degenerate box, orphan annotation,
unknown class) costs 15 points; a ``warning`` (duplicate, tiny box,
missing annotations, …) costs 5. ``info`` findings cost nothing.
An image with no findings scores 100. The score is a triage signal,
not a grade — a 100 does not mean the boxes are correctly placed,
only that no structural problem was found.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .checks import Finding
from .model import Dataset

ERROR_COST = 15
WARNING_COST = 5


@dataclass
class QualityReport:
    dataset_name: str
    total_images: int
    total_annotations: int
    findings: list[Finding] = field(default_factory=list)
    by_severity: dict[str, int] = field(default_factory=dict)
    by_check: dict[str, int] = field(default_factory=dict)
    image_scores: dict[str, int] = field(default_factory=dict)
    worst_images: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "total_images": self.total_images,
            "total_annotations": self.total_annotations,
            "findings": [f.to_dict() for f in self.findings],
            "by_severity": self.by_severity,
            "by_check": self.by_check,
            "image_scores": self.image_scores,
            "worst_images": self.worst_images,
        }


def _score_for(findings: list[Finding]) -> int:
    score = 100
    for f in findings:
        if f.severity == "error":
            score -= ERROR_COST
        elif f.severity == "warning":
            score -= WARNING_COST
    return max(score, 0)


def build_report(dataset: Dataset, findings: list[Finding], dataset_name: str = "") -> QualityReport:
    by_image: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        if f.image_id:
            by_image[f.image_id].append(f)
    image_scores = {
        img.id: _score_for(by_image.get(img.id, [])) for img in dataset.images
    }
    worst = sorted(
        (
            {
                "image_id": img.id,
                "file_name": img.file_name,
                "score": image_scores[img.id],
                "errors": sum(1 for f in by_image.get(img.id, []) if f.severity == "error"),
                "warnings": sum(1 for f in by_image.get(img.id, []) if f.severity == "warning"),
            }
            for img in dataset.images
            if by_image.get(img.id)
        ),
        key=lambda row: (row["score"], -row["errors"], row["image_id"]),
    )
    return QualityReport(
        dataset_name=dataset_name or dataset.source_format or "dataset",
        total_images=len(dataset.images),
        total_annotations=len(dataset.annotations),
        findings=list(findings),
        by_severity=dict(Counter(f.severity for f in findings)),
        by_check=dict(Counter(f.check for f in findings)),
        image_scores=image_scores,
        worst_images=worst,
    )


def _bar(count: int, width: int = 30) -> str:
    if count == 0:
        return ""
    return "█" * min(count, width)


def render_text(report: QualityReport) -> str:
    """Human-readable report, deterministic ordering."""
    lines = [
        f"Annotation QA report — {report.dataset_name}",
        "=" * 60,
        f"Images: {report.total_images}   Annotations: {report.total_annotations}",
        f"Findings: {len(report.findings)}  "
        f"(errors: {report.by_severity.get('error', 0)}, "
        f"warnings: {report.by_severity.get('warning', 0)}, "
        f"info: {report.by_severity.get('info', 0)})",
        "",
        "Findings by check:",
    ]
    if report.by_check:
        for check, count in sorted(report.by_check.items()):
            lines.append(f"  {check:<24} {count:>3} {_bar(count)}")
    else:
        lines.append("  (none — no findings)")
    lines.append("")
    lines.append("Worst images (lowest quality score first):")
    if report.worst_images:
        for row in report.worst_images[:10]:
            lines.append(
                f"  {row['image_id']:<18} score {row['score']:>3}  "
                f"errors {row['errors']}  warnings {row['warnings']}"
            )
    else:
        lines.append("  (none — every image scored 100)")
    lines.append("")
    lines.append("Findings:")
    if report.findings:
        for f in report.findings:
            where = f"[{f.image_id}] " if f.image_id else ""
            lines.append(f"  [{f.severity.upper():7}] {f.check:<22} {where}{f.message}")
    else:
        lines.append("  (none)")
    return "\n".join(lines) + "\n"
