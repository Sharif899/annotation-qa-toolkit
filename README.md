# annotation-qa-toolkit

Offline, deterministic QA for image and video annotation datasets — the
checks a QA reviewer runs before a labelled dataset ships to training.

It reads the formats labelling tools actually export — **COCO, YOLO,
Pascal VOC, Labelbox, SuperAnnotate and Roboflow** — normalises them
into one model, then checks the work: boxes out of bounds, duplicates,
degenerate and tiny boxes, class typos, images with no annotations,
and class imbalance. It also compares two annotators (Cohen's kappa
on class presence, IoU box matching) and scores every image so the
worst ones come first.

Python 3.10+, standard library only. No API keys, no network, no
client data. The example dataset is synthetic.

## Why this exists

I spent years on the other side of this problem — training annotators
and reviewing their work at iMerit, and running evaluation pipelines
at Databricks. The expensive mistakes in a labelling project are
rarely exotic: a class spelled two ways, the same box drawn twice, a
box dragged past the image edge, one annotator quietly labelling half
as much as everyone else. Those are structural problems, and they can
be caught mechanically before a model ever sees the data. This toolkit
is the mechanical part. It does not judge whether a box is correctly
placed — only a human reviewer can do that.

## Install

```bash
pip install -e .
```

## Usage

```bash
# Summarise a dataset
annoqa load data/examples/coco_annotator_a.json

# Run every QA check (exit 1 when an error-severity finding exists)
annoqa check data/examples/coco_annotator_a.json

# Full quality report, worst images first
annoqa report data/examples/coco_annotator_a.json

# Compare two annotators
annoqa agreement data/examples/coco_annotator_a.json data/examples/coco_annotator_b.json \
    --first-annotator ada --second-annotator bola

# Any command accepts --json for machine-readable output
annoqa report data/examples/coco_annotator_a.json --json

# YOLO needs image sizes (sizes.json next to the .txt files, or --sizes)
annoqa load data/examples/yolo

# Formats are auto-detected; override with --format coco|yolo|voc|labelbox|superannotate|roboflow
# JSON can come from stdin:  cat export.json | annoqa check -
```

## Example output

Real output from the included example data (10 images, two
annotators, deliberately imperfect):

```
$ annoqa report data/examples/coco_annotator_a.json
Annotation QA report — data/examples/coco_annotator_a.json
============================================================
Images: 10   Annotations: 24
Findings: 7  (errors: 1, warnings: 6, info: 0)

Findings by check:
  duplicate_box              1 █
  empty_annotations          1 █
  heavy_overlap              1 █
  near_duplicate_class       2 ██
  out_of_bounds              1 █
  tiny_box                   1 █

Worst images (lowest quality score first):
  img_004            score  85  errors 1  warnings 0
  img_002            score  95  errors 0  warnings 1
  img_005            score  95  errors 0  warnings 1
  img_006            score  95  errors 0  warnings 1
  img_008            score  95  errors 0  warnings 1
  img_010            score  95  errors 0  warnings 1

Findings:
  [ERROR  ] out_of_bounds          [img_004] 'car' box (900, 500, 1200, 700) exceeds image 1024x768.
  [WARNING] duplicate_box          [img_002] Two 'person' boxes overlap almost completely (IoU 1.00) — likely drawn twice.
  [WARNING] empty_annotations      [img_010] Image 'shop_002.jpg' has no annotations.
  [WARNING] heavy_overlap          [img_006] Two 'bicycle' boxes overlap heavily (IoU 0.62).
  [WARNING] near_duplicate_class   Classes ['Person', 'person'] look like the same class (normalised to 'person') — pick one spelling.
  [WARNING] near_duplicate_class   [img_008] Annotation class 'Person' matches listed class 'person' after normalisation.
  [WARNING] tiny_box               [img_005] 'dog' box is only 5x6 px (minimum side 8 px).
```

```
$ annoqa agreement data/examples/coco_annotator_a.json data/examples/coco_annotator_b.json \
      --first-annotator ada --second-annotator bola
Annotator agreement
  Cohen's kappa (class presence): 0.886 (observed 0.95, expected 0.561, n=60)
  Box agreement (IoU >= 0.5): matched 20, missed 4, extra 3, match rate 0.833, mean IoU 0.86
  ada: 24 annotations, 9 images, 2.4 boxes/image, classes: Person, bicycle, car, dog, person, traffic_light
  bola: 23 annotations, 9 images, 2.3 boxes/image, classes: bicycle, car, dog, person, traffic_light
```

The agreement numbers tell the story: the two annotators mostly agree
(κ 0.886), but Bola missed the dog on `img_003`, drew an extra car on
`img_008`, and Ada's capitalised `Person` class broke box matching on
that image — the kind of small, fixable drift a QA lead wants to catch
in review, not after training.

## The checks

| Check | Severity | What it catches |
|---|---|---|
| `out_of_bounds` | error | Box extends past the image edge |
| `degenerate_box` | error | Zero-area box (zero width/height, inverted corners) |
| `orphan_annotation` | error | Annotation refers to an image that isn't in the dataset |
| `unknown_class` | error | Class not in the dataset's class list |
| `duplicate_box` | warning | Same-class boxes, IoU ≥ 0.95 — drawn twice |
| `heavy_overlap` | warning | Same-class boxes, IoU 0.5–0.95 — likely one object boxed twice |
| `tiny_box` | warning | Box side below `--min-side` px (default 8) |
| `empty_annotations` | warning | Image with no annotations at all |
| `near_duplicate_class` | warning | `Car` vs `cars` — same class under two spellings |
| `rare_class` | warning | Class below `--rare-share` of annotations (default 2%) |

Quality score per image: `100 − 15 × errors − 5 × warnings`, floored
at 0. It is a triage signal, not a grade — a score of 100 means no
structural problem was found, not that the boxes are correctly placed.

## Honest limits

- **It validates structure, not correctness.** It cannot tell whether
  a box is around the right object, or whether a class label is
  semantically right. That is the reviewer's job.
- **IoU matching is a heuristic.** Greedy best-IoU pairing can match
  boxes that belong to different objects when objects are crowded.
  Treat match rates as a review signal, not ground truth.
- **Kappa is on class presence, not box identity.** It measures
  whether both annotators saw the same classes on each image, not
  whether they drew the same boxes — use box agreement for that.
- **YOLO needs image sizes.** YOLO boxes are normalised, so a
  `sizes.json` (or `--sizes`) mapping image names to `[width, height]`
  is required; without it the reader refuses rather than guessing.
- **Thresholds are starting points.** The 8 px tiny-box minimum and
  the 2% rare-class share suit ordinary object detection; small-object
  datasets should pass their own `--min-side` and `--rare-share`.

## Layout

```
src/annotation_qa_toolkit/
  model.py       Image / Annotation / Dataset, bounding-box geometry
  readers.py     COCO, YOLO, Pascal VOC, Labelbox, SuperAnnotate, Roboflow
  checks.py      the ten QA checks
  agreement.py   Cohen's kappa, IoU box matching, per-annotator stats
  report.py      quality scores, worst-images list, text/JSON rendering
  cli.py         annoqa load | check | agreement | report
data/examples/   synthetic COCO (two annotators), YOLO and Pascal VOC sets
tests/           88 pytest tests
```

## Tests

```bash
pip install pytest
python -m pytest -q
```

## Licence

MIT.
