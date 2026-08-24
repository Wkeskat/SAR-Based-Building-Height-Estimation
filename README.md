# SAR-based Building Height Retrieval

Reproduction of Sun et al. (2022), "Large-scale building height retrieval from
single SAR imagery based on bounding box regression networks", adapted for
TerraSAR-X spotlight imagery over Khon Kaen, Thailand (GISTDA).

**Current results** (test set, n=6,800, held-out): Mean IoU 0.866, Height MAE
1.32m, RMSE 1.83m, Correlation 0.900. See `docs/model_evaluation_report_v2.docx`
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
scripts/                          — data prep, split, quality audit, prediction pipeline, diagnostics
docs/                             — full pipeline writeup and evaluation report
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
Convert footprint polygons + SAR imagery into one crop per building (VOC
format). Uses the layover geometry relationship `L = h·cos θ` to compute each
building's ground-truth box from its known height.

- Margin per crop: **376px** (union of footprint + layover box, plus buffer).
  This exact value matters — see the note on scale-invariance below.
- Channel packing: R=SAR amplitude, G=footprint mask (binary), B=SAR duplicate.
  Must be PNG, not JPEG — JPEG compression blurs the mask edges the model
  depends on.

### Phase 2 — Geographic split
```powershell
python scripts/resplit_geographic_trainvaltest.py
```
Splits by **percentile of surviving building positions**, not raw image width
(buildings only occupy 24.6–86.7% of the scene). Produces `train.txt`,
`val.txt`, `test.txt`, `buffer.txt` plus `buffer.shp` for QGIS inspection.

### Phase 3 — Quality audit
```powershell
python scripts/scan_degenerate_masks.py
python scripts/remove_degenerate_from_splits.py
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
python scripts/export_predict_patches.py   # crop the new area (edit paths inside first)
python predict.py --net res101 --cuda --id_column objectid --metadata_pkl <path> --footprint_path <path> --output_geojson <path> --degenerate_ids_path <path>
```
`predict.py` defaults (`--checksession 1 --checkepoch 10 --checkpoint 38157`)
already point at the trained checkpoint — no path editing needed if using the
standard folder layout. Outputs a GeoJSON with height + a `quality_flag`
column, plus a second `_bbox.geojson` with the predicted box geometry itself.

`--id_column` auto-detects common conventions (`building_id`, `objectid`,
`gid`, `fid`, `globalId`) — override explicitly if your footprint file uses
something else.

---

## 4. Critical things that will bite you if skipped

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

---

## 5. Known limitations

- Single-object architecture — one building per image, not a general detector
- Fixed incidence angle assumption (41.3°) when converting box to height;
  real range is 41.0–41.6° across the scene (adds ≤0.7% error, not the main
  source of uncertainty)
- One geographic region (~1% of the dataset) has known footprint data quality
  issues; degenerate masks are filtered but the region hasn't been
  re-digitized at the source
