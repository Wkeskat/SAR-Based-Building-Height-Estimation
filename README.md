# SAR-based Building Height Retrieval

Reproduction of Sun et al. (2022), "Large-scale building height retrieval from
single SAR imagery based on bounding box regression networks", adapted for
TerraSAR-X spotlight imagery over Khon Kaen, Thailand (GISTDA).

**Current results** (test set, n=6,800, held-out): Mean IoU 0.866, Height MAE
1.13m, RMSE 1.56m, Correlation 0.900. See `docs/model_evaluation_report_v2.docx`
for the full evaluation.

---

## 1. Environment setup

```powershell
pip install -r requirements.txt --break-system-packages
```

**Read the comments inside `requirements.txt` before installing** — a couple of
versions are pinned for real reasons (Pillow 10.4.0 specifically, not latest).

**CUDA GPU is mandatory**, not optional — even for `predict.py`. The model
architecture (`faster_rcnn_giou.py`) has a hardcoded `.cuda()` call inside
`forward()` regardless of the `--cuda` flag. CPU-only inference is not possible
without further code changes.

---

## 2. Repository structure

```
train.py, test.py, predict.py    — main entry points
_init_paths.py                    — sys.path setup, required by all three
cfgs/res101.yml                   — model config
lib/                              — model architecture, dataset loader, utils
scripts/                          — data prep, split, quality audit, prediction pipeline
docs/                              — full pipeline writeup and evaluation report
data/pretrained_model/            — put resnet101_caffe.pth here (TRAINING ONLY)
```

Two directories are **not included** and must be provided separately:
- `data/pretrained_model/resnet101_caffe.pth` — only needed for training from
  scratch, not for `predict.py` (which loads a fully-trained checkpoint directly)
- Your own dataset directory (`sar_pipeline/` in this project) — `JPEGImages/`,
  `Annotations/`, `ImageSets/Main/*.txt`

---

## 3. Full pipeline, in order

### Phase 1 — Data preparation
```powershell
python scripts/01_compute_bbox_labels.py
python scripts/02_export_percrop_patches.py
```
Convert footprint polygons + SAR imagery into one crop per building (VOC
format). Uses the layover geometry relationship **`L = h·tan θ`** to compute
each building's ground-truth box from its known height — confirmed directly
against `01_compute_bbox_labels.py`'s own `compute_building_bbox()` function.
This is the ground-range (GEC) form of the relationship; `cos θ` applies to
slant-range/SLC data and is **not** correct here. An earlier version of this
codebase used `cos θ` throughout downstream evaluation/prediction code and
produced self-consistent but systematically inflated height estimates (~17%
high) until traced back to this source script and corrected everywhere.

Note also that the layover shift computed here is a genuine **2D vector** in
(East, North), determined by satellite heading and look side — not purely
horizontal. Downstream scripts approximate this using only the x-axis
component (~1.6% underestimate for this scene's geometry); acceptable given
it's an order of magnitude smaller than the formula correction above, but
worth knowing if adapting this pipeline to a scene with a very different
heading/look-side combination.

- Margin per crop: **376px** (union of footprint + layover box, plus buffer).
  This exact value matters — see the note on scale-invariance below.
- Channel packing: R=SAR amplitude, G=footprint mask (binary), B=SAR duplicate.
  Must be PNG, not JPEG — JPEG compression blurs the mask edges the model
  depends on.

### Phase 2 — Geographic split
```powershell
python scripts/03_resplit_geographic_trainvaltest.py
```
Splits by **percentile of surviving building positions**, not raw image width
(buildings only occupy 24.6–86.7% of the scene). Produces `train.txt`,
`val.txt`, `test.txt`, `buffer.txt` plus `buffer.shp` for QGIS inspection.

### Phase 3 — Quality audit
```powershell
python scripts/04_scan_degenerate_masks.py
python scripts/05_remove_degenerate_from_splits.py
```
Scans for corrupted footprint masks (found 640/58,131 = 1.1%, spatially
clustered — likely a source-data issue in one region, not random noise).
Removes them from all splits, with automatic backups.

### Phase 4 — Train
```powershell
python train.py --dataset sar_building --net res101 --bs 1 --nw 0 --cuda --s 1 --auto_resume
```
`--auto_resume` finds the latest checkpoint automatically — safe to re-run
after any interruption.

### Phase 5 — Test
```powershell
python test.py --dataset sar_building --net res101 --checksession 1 --checkepoch 10 --checkpoint 38157 --cuda
```
Runs `evaluate_detections()` — computes IoU, converts predicted boxes back to
height, reports MAE/RMSE/correlation, breaks results down by height bucket,
and flags the worst individual cases with a degenerate-mask check.

### Phase 6 — Predict on new data
```powershell
python scripts/06_export_predict_patches.py   # crop the new area (edit paths inside first)
python predict.py --net res101 --cuda --id_column objectid --metadata_pkl <path> --footprint_path <path> --output_geojson <path> --degenerate_ids_path <path>
```
`predict.py` defaults (`--checksession 1 --checkepoch 10 --checkpoint 38157`)
already point at the trained checkpoint — no path editing needed if using the
standard folder layout. Outputs a GeoJSON with height + a `quality_flag`
column, plus a second `_bbox.geojson` with the predicted box geometry itself.
Predicted boxes are clamped so they never fall inside the footprint on the
top/right/bottom edges (matching the mathematical guarantee that real labels
always satisfy — see Section 4); this never affects the reported height,
which depends only on the left edge.

`--id_column` auto-detects common conventions (`building_id`, `objectid`,
`gid`, `fid`, `globalId`) — override explicitly if your footprint file uses
something else.

**If predicting on a genuinely new geographic area** (not held-out buildings
from the same scene), expect meaningfully reduced accuracy compared to the
Phase 5 test-set numbers above — this is expected generalization behavior,
not a bug. If comparing against an external data source (e.g. a government
building registry) rather than the SAR-derived labels used in training, also
account for possible differences in how "height" is defined/measured by that
source before treating disagreement as model error.

---

## 4. Critical things that will bite you if skipped

**Use `tan θ`, not `cos θ`, anywhere height is computed from layover length.**
Confirmed against `01_compute_bbox_labels.py`, the actual label-generation
script. `cos θ` is correct for slant-range (SLC) SAR data; this pipeline uses
ground-range (GEC) imagery, which requires `tan θ`. The two formulas differ
by roughly 17% at this scene's incidence angle (~41.3°) — enough to be
self-consistent and easy to miss, but not a rounding-level difference.

**Clear caches after editing anything.** Two different kinds:
```powershell
# after editing train.txt/val.txt/test.txt/buffer.txt, or regenerating crops:
del "data\cache\*.pkl"

# after editing any .py file, if changes don't seem to take effect:
Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force
```
The data cache is keyed only by dataset name, not by split-file contents — it
will silently serve stale data otherwise.

**Crop margin must match exactly between training and prediction.** This
model has no true RoI-pooling (`pooled_feat = base_feat` — confirmed by
reading `faster_rcnn_giou.py`), so it is *not* scale-invariant. A different
margin at prediction time than training time (200px vs 376px) produces
systematic, not random, height errors.

**Use `training=False` in `roibatchLoader` for both testing and prediction,
never `training=True`.** That flag activates aspect-ratio batch reordering,
which silently misaligns predictions with building IDs — no error, just wrong
labels attached to every result. `predict.py` and `test.py` both already do
this correctly; don't "fix" it back to `True`.

**`fasterRCNN.eval()`, not `.train()`, for validation/testing/prediction —
but only once `training=False` is also set.** The two must be changed
together; this codebase's `forward()` ties loss computation and code paths to
`self.training` in ways that don't match standard PyTorch conventions.

**Always verify `pred_boxes` is divided by `im_scale`** after
`bbox_transform_inv`/`clip_boxes`. Skipping this inflates predicted heights by
roughly the resize factor (commonly ~3x in this project).

**A predicted box should always contain the footprint on 3 of its 4 sides.**
Ground-truth labels are mathematically guaranteed to satisfy `B_b ⊇ B_f`
(built as `union(footprint, shifted_footprint).envelope`). Predictions have no
such architectural guarantee — the top/right/bottom edges are clamped in
`predict.py` to enforce this for visual consistency in exported GeoJSONs. The
left edge is deliberately left unclamped, since that's the only edge the
height formula reads.

---

## 5. Known limitations

- Single-object architecture — one building per image, not a general detector
- Fixed incidence angle assumption (41.3°) when converting box to height;
  real range is 41.0–41.6° across the scene (adds ≤0.7% error, not the main
  source of uncertainty)
- One geographic region (~1% of the dataset) has known footprint data quality
  issues; degenerate masks are filtered but the region hasn't been
  re-digitized at the source
- Relative height error is higher for short buildings (~27% at <5m vs. ~13%
  at 10-40m) — absolute error stays roughly constant (~1-3m) across height
  classes; the same pixel-level noise is simply a larger fraction of a
  smaller true value. Not a sign of degraded model quality at short heights.
- Accuracy on genuinely new geographic areas (outside the training scene) is
  measurably lower than on the held-out test set from the same scene — expected
  generalization behavior worth reporting explicitly, not concealing, when
  presenting results on new-area predictions.

---

## 6. Not yet included in `scripts/`

The following diagnostic and evaluation utilities were developed alongside
this pipeline but are not yet part of this repository's `scripts/` folder:
UTM coordinate conversion, prediction-to-footprint joining, single-image
preprocessing debug, dataloader index-alignment verification, full-dataset
prediction evaluation (with and without degenerate-mask filtering), test-set
bounding-box GeoJSON export, bbox-containment verification, and test-height-
to-footprint export. Add these if reproducing the full evaluation/QGIS
workflow described in the accompanying poster and evaluation report.
