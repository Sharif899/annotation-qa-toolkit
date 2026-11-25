"""Command line interface.

    annoqa load <path> [--format F] [--annotator NAME] [--sizes S]
    annoqa check <path> [--format F] [--min-side N] [--rare-share R]
    annoqa agreement <first> <second> [--format F]
    annoqa report <path> [--format F] [--min-side N] [--rare-share R]

``path`` may be ``-`` to read JSON from stdin (COCO / Labelbox /
SuperAnnotate only). ``--json`` prints machine-readable output
instead of the human-readable form. ``check`` and ``report`` exit
with status 1 when any ``error``-severity finding exists, so they can
gate a labelling pipeline; ``load`` and ``agreement`` always exit 0
on success.
"""

from __future__ import annotations

import argparse
import json
import sys

from .agreement import box_agreement, cohen_kappa, per_annotator_stats
from .checks import run_all_checks
from .model import Dataset
from .readers import AnnotationReadError, load
from .report import build_report, render_text


def _load_dataset(path: str, fmt: str | None, annotator: str, sizes_path: str | None) -> Dataset:
    if path == "-":
        raw = sys.stdin.read()
        if not raw.strip():
            raise AnnotationReadError("stdin: empty input")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AnnotationReadError(f"stdin: invalid JSON ({exc})") from exc
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump(data, fh)
            tmp = fh.name
        try:
            return load(tmp, fmt=fmt, annotator=annotator)
        finally:
            os.unlink(tmp)
    kwargs: dict = {}
    if sizes_path:
        with open(sizes_path, encoding="utf-8") as fh:
            raw_sizes = json.load(fh)
        kwargs["sizes"] = {k: (int(v[0]), int(v[1])) for k, v in raw_sizes.items()}
    return load(path, fmt=fmt, annotator=annotator, **kwargs)


def _cmd_load(args: argparse.Namespace) -> int:
    ds = _load_dataset(args.path, args.format, args.annotator, args.sizes)
    if args.json:
        print(json.dumps(ds.to_dict(), indent=2, sort_keys=True))
        return 0
    counts = ds.category_counts()
    print(f"Dataset: {ds.source_format or 'unknown'}  (annotator: {ds.annotator or '—'})")
    print(f"Images: {len(ds.images)}   Annotations: {len(ds.annotations)}")
    print(f"Classes ({len(ds.categories)}): {', '.join(ds.categories) or '—'}")
    for cls, n in sorted(counts.items()):
        print(f"  {cls:<20} {n}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    ds = _load_dataset(args.path, args.format, args.annotator, args.sizes)
    findings = run_all_checks(ds, min_side=args.min_side, rare_share=args.rare_share)
    if args.json:
        print(json.dumps([f.to_dict() for f in findings], indent=2, sort_keys=True))
    else:
        if not findings:
            print("No findings — the dataset passed every check.")
        for f in findings:
            where = f"[{f.image_id}] " if f.image_id else ""
            print(f"[{f.severity.upper():7}] {f.check:<22} {where}{f.message}")
    return 1 if any(f.severity == "error" for f in findings) else 0


def _cmd_agreement(args: argparse.Namespace) -> int:
    first = _load_dataset(args.first, args.format, args.first_annotator, args.sizes)
    second = _load_dataset(args.second, args.format, args.second_annotator, args.sizes)
    kappa = cohen_kappa(first, second)
    boxes = box_agreement(first, second, iou_threshold=args.iou)
    stats = per_annotator_stats([first, second])
    payload = {
        "kappa": kappa.to_dict(),
        "box_agreement": boxes.to_dict(),
        "annotators": [s.to_dict() for s in stats],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print("Annotator agreement")
    print(f"  Cohen's kappa (class presence): {kappa.kappa} "
          f"(observed {kappa.observed_agreement}, expected {kappa.expected_agreement}, "
          f"n={kappa.n_decisions})")
    print(f"  Box agreement (IoU >= {args.iou}): matched {boxes.matched}, "
          f"missed {boxes.missed}, extra {boxes.extra}, "
          f"match rate {boxes.match_rate}, mean IoU {boxes.mean_iou}")
    for s in stats:
        print(f"  {s.annotator}: {s.total_annotations} annotations, "
              f"{s.images_annotated} images, {s.avg_boxes_per_image} boxes/image, "
              f"classes: {', '.join(s.classes_used)}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    ds = _load_dataset(args.path, args.format, args.annotator, args.sizes)
    findings = run_all_checks(ds, min_side=args.min_side, rare_share=args.rare_share)
    report = build_report(ds, findings, dataset_name=args.path)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_text(report), end="")
    return 1 if any(f.severity == "error" for f in findings) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annoqa",
        description="Annotation QA toolkit: validate labelling work across "
                    "COCO, YOLO, Pascal VOC, Labelbox, SuperAnnotate and Roboflow exports.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--format", choices=["coco", "roboflow", "yolo", "voc", "labelbox", "superannotate"],
                       help="annotation format (auto-detected when omitted)")
        p.add_argument("--annotator", default="", help="name of the annotator for this dataset")
        p.add_argument("--sizes", default=None, metavar="SIZES.JSON",
                       help="YOLO image sizes: JSON mapping image name to [width, height]")
        p.add_argument("--json", action="store_true", help="print machine-readable JSON")

    load_p = sub.add_parser("load", help="load a dataset and print a summary")
    load_p.add_argument("path", help="annotation file, directory, or - for stdin")
    add_common(load_p)
    load_p.set_defaults(func=_cmd_load)

    check_p = sub.add_parser("check", help="run every QA check and list the findings")
    check_p.add_argument("path", help="annotation file, directory, or - for stdin")
    check_p.add_argument("--min-side", type=float, default=8.0, help="minimum box side in px (default 8)")
    check_p.add_argument("--rare-share", type=float, default=0.02, help="rare-class share threshold (default 0.02)")
    add_common(check_p)
    check_p.set_defaults(func=_cmd_check)

    agree_p = sub.add_parser("agreement", help="compare two annotators' datasets")
    agree_p.add_argument("first", help="first annotator's annotations")
    agree_p.add_argument("second", help="second annotator's annotations")
    agree_p.add_argument("--first-annotator", default="annotator_a")
    agree_p.add_argument("--second-annotator", default="annotator_b")
    agree_p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for box matching (default 0.5)")
    add_common(agree_p)
    agree_p.set_defaults(func=_cmd_agreement)

    report_p = sub.add_parser("report", help="full quality report for a dataset")
    report_p.add_argument("path", help="annotation file, directory, or - for stdin")
    report_p.add_argument("--min-side", type=float, default=8.0)
    report_p.add_argument("--rare-share", type=float, default=0.02)
    add_common(report_p)
    report_p.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except AnnotationReadError as exc:
        print(f"annoqa: error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"annoqa: error: file not found: {exc.filename}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
