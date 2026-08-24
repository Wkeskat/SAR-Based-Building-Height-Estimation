"""
predict.py (v5 - consolidated, outputs GeoJSON directly)

รันโมเดลที่เทรนแล้ว ทำนาย bounding box + คำนวณความสูงอาคาร แล้วรวมกับ
geometry เดิมเป็น GeoJSON เดียว (ไม่ต้องรัน 13_join_predictions_to_footprints.py
แยกอีกขั้นตอน - รวมไว้ในสคริปต์นี้แล้ว)

ประวัติการแก้ไข (Changelog):
  v1: ไม่มีการหาร im_scale -> ทำนายความสูงเกินจริง (~50-60m)
  v2: เพิ่มการหาร im_scale แต่ margin ตอน export ไม่ตรงกับตอนเทรน
  v3: แก้ margin (override_margin_px=376) - ยัง saturate ที่ ~250m
  v4: พบ index misalignment จาก roibatchLoader(training=True) - แก้เป็น
      training=False + fasterRCNN.eval() (ตรงกับ test.py ทุกประการ)
  v5 (ปัจจุบัน): รวม logic การ join geometry เข้ามาในไฟล์เดียว, เพิ่ม
      quality_flag, เพิ่มตัวเลือกกรอง building_id ที่รู้ว่า mask เสียหาย
      (degenerate_ids_path) ก่อนเขียนผลลัพธ์สุดท้าย - เปลี่ยน --checkpoint
      จาก path เต็มกลับมาเป็นตัวเลข session/epoch/checkpoint (เหมือน test.py)
      พร้อมตั้งค่า default ให้ตรงกับโมเดลที่เทรนไว้แล้ว (1, 10, 38157) - ใครก็
      รันได้โดยไม่ต้องรู้ path ของ checkpoint เลย ถ้าใช้โฟลเดอร์ตาม convention

วิธีรัน (ง่ายที่สุด - ใช้ default ทั้งหมด ถ้า checkpoint อยู่ที่
models/res101/sar_building/faster_rcnn_1_10_38157.pth):
  python predict.py --net res101 --cuda ^
      --metadata_pkl path\\to\\sar_building_predict\\predict_metadata.pkl ^
      --footprint_path path\\to\\footprints.geojson ^
      --output_geojson path\\to\\predicted_building_heights.geojson

วิธีรัน (ระบุ checkpoint เอง ถ้าไม่ใช่ default):
  python predict.py --net res101 --cuda ^
      --checksession 1 --checkepoch 10 --checkpoint 38157 ^
      --metadata_pkl path\\to\\sar_building_predict\\predict_metadata.pkl ^
      --footprint_path path\\to\\footprints.geojson ^
      --output_geojson path\\to\\predicted_building_heights.geojson
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging
import datetime
import pickle
import csv
import time
import numpy as np
import pandas as pd
import geopandas as gpd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "lib"))

import torch
from torch.autograd import Variable

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv

try:
  from model.faster_rcnn.vgg16 import vgg16
except ModuleNotFoundError:
  vgg16 = None
from model.faster_rcnn.resnet_giou import resnet

# แก้ list นี้ให้ตรงกับ building_id ที่ต้องการดู debug แบบละเอียด (ว่างไว้ได้
# ถ้าไม่ต้องการ debug output)
DEBUG_BUILDING_IDS = set()

# สมมติฐานความสูงสูงสุดที่เป็นไปได้จริง (ม.) - ใช้ตั้ง quality_flag เท่านั้น
MAX_REASONABLE_HEIGHT_M = 100.0


def setup_logger(output_dir):
  os.makedirs(output_dir, exist_ok=True)
  stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  log_path = os.path.join(output_dir, f"predict_log_{stamp}.txt")

  logger = logging.getLogger("predict")
  logger.setLevel(logging.INFO)
  logger.handlers.clear()

  fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
  fh = logging.FileHandler(log_path, encoding="utf-8")
  fh.setFormatter(fmt)
  logger.addHandler(fh)
  ch = logging.StreamHandler()
  ch.setFormatter(fmt)
  logger.addHandler(ch)

  logger.info(f"Log file: {log_path}")
  return logger


def parse_args():
  parser = argparse.ArgumentParser(description='ทำนายความสูงอาคาร -> GeoJSON เดียว')
  parser.add_argument('--dataset', dest='dataset', default='sar_building_predict', type=str)
  parser.add_argument('--net', dest='net', default='res101', type=str)
  parser.add_argument('--load_dir', dest='load_dir', default="models", type=str,
                       help='โฟลเดอร์หลักที่เก็บ checkpoint (default: models, ตรงกับที่ train.py ใช้)')
  parser.add_argument('--checksession', dest='checksession', default=1, type=int,
                       help='session number ของ checkpoint (default: 1, ตรงกับโมเดลที่เทรนไว้แล้ว)')
  parser.add_argument('--checkepoch', dest='checkepoch', default=10, type=int,
                       help='epoch number ของ checkpoint (default: 10, epoch สุดท้ายที่เทรนไว้)')
  parser.add_argument('--checkpoint', dest='checkpoint', default=38157, type=int,
                       help='checkpoint/iteration number (default: 38157, checkpoint สุดท้ายที่เทรนไว้) '
                            '- ไม่ต้องพิมพ์ path เต็ม แค่ตัวเลข 3 ตัวนี้พอ (เหมือน test.py)')
  parser.add_argument('--cuda', dest='cuda', action='store_true')
  parser.add_argument('--class_agnostic', dest='class_agnostic', action='store_true')
  parser.add_argument('--metadata_pkl', dest='metadata_pkl', required=True, type=str,
                       help='predict_metadata.pkl จาก 10_export_predict_patches.py')
  parser.add_argument('--footprint_path', dest='footprint_path', required=True, type=str,
                       help='footprint file (.shp/.geojson) เดียวกับที่ใช้ตอน Step 10')
  parser.add_argument('--output_geojson', dest='output_geojson', required=True, type=str,
                       help='ไฟล์ผลลัพธ์สุดท้าย (.geojson หรือ .shp)')
  parser.add_argument('--output_csv', dest='output_csv', default=None, type=str,
                       help='(ไม่บังคับ) เก็บ CSV ดิบไว้ด้วยเผื่อ debug - default: ข้าง output_geojson')
  parser.add_argument('--degenerate_ids_path', dest='degenerate_ids_path', default=None, type=str,
                       help='(ไม่บังคับ) path ไปยัง degenerate_building_ids.txt - '
                            'ถ้าระบุ จะตัด building_id เหล่านั้นออกก่อนเขียนผลลัพธ์สุดท้าย')
  parser.add_argument('--id_column', dest='id_column', default=None, type=str,
                       help='(ไม่บังคับ) ระบุตรงๆ ว่า column ไหนใน footprint file คือ '
                            'unique id (เช่น "objectid") - ถ้าไม่ระบุ จะลองหาอัตโนมัติจาก '
                            'building_id/building_i/objectid/gid/fid/globalId')
  return parser.parse_args()


def main():
  args = parse_args()

  output_dir = os.path.dirname(os.path.abspath(args.output_geojson))
  logger = setup_logger(output_dir)
  logger.info('Called with args:')
  logger.info(str(vars(args)))

  args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  cfg_from_file(os.path.join(THIS_DIR, "cfgs", f"{args.net}.yml"))
  cfg_from_list(args.set_cfgs)
  cfg.TRAIN.USE_FLIPPED = False

  # --- โหลด dataset (training=False: index ตรงลำดับ ตรงกับ image_index เป๊ะ -
  # ดู changelog v4 ด้านบนสำหรับเหตุผลที่ห้ามใช้ training=True) ---
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.dataset, False)
  logger.info('{:d} roidb entries'.format(len(roidb)))

  with open(args.metadata_pkl, "rb") as f:
    meta_data = pickle.load(f)
  metadata = meta_data["metadata"]
  pixel_spacing_m = meta_data["pixel_spacing_m"]
  # affine transform + CRS สำหรับแปลง predicted box (pixel) -> UTM polygon
  # (Step 10 เก็บไว้ให้แล้ว ไม่ต้องเปิด GeoTIFF ใหม่) - ถ้าเป็น metadata เก่า
  # ที่ยังไม่มีค่านี้ จะข้ามการเขียน bbox geojson ไป
  sar_transform_vals = meta_data.get("sar_transform")
  sar_crs_str = meta_data.get("sar_crs")
  if sar_transform_vals is None:
    logger.info("⚠️  predict_metadata.pkl ไม่มี sar_transform (เป็นไฟล์เก่า) - "
                 "จะไม่เขียน bbox geojson ให้ รันสคริปต์ Step 10 ใหม่ถ้าต้องการ")

  # --- โหลดโมเดล ---
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    raise ValueError("network is not defined")

  fasterRCNN.create_architecture()

  # สร้าง path ของ checkpoint จากตัวเลข session/epoch/checkpoint แทนที่จะให้
  # ผู้ใช้พิมพ์ path เต็มเอง (ใช้ convention เดียวกับ train.py/test.py:
  # models/{net}/sar_building/faster_rcnn_{session}_{epoch}_{checkpoint}.pth)
  input_dir = os.path.join(args.load_dir, args.net, "sar_building")
  load_name = os.path.join(input_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
  assert os.path.exists(load_name), (
      f"ไม่พบ checkpoint ที่ {load_name} - เช็คว่า --load_dir/--net/--checksession/"
      f"--checkepoch/--checkpoint ตรงกับ checkpoint ที่มีจริงหรือไม่ "
      f"(เช่น ls {input_dir} เพื่อดูชื่อไฟล์ checkpoint ที่มีอยู่จริง)"
  )

  logger.info("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
  logger.info('load model successfully!')

  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  if args.cuda:
    im_data, im_info = im_data.cuda(), im_info.cuda()
    num_boxes, gt_boxes = num_boxes.cuda(), gt_boxes.cuda()
    cfg.CUDA = True
    fasterRCNN.cuda()

  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  num_images = len(imdb.image_index)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                            imdb.num_classes, training=False, normalize=False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
  data_iter = iter(dataloader)

  results = []
  start = time.time()

  fasterRCNN.eval()  # ต้องคู่กับ training=False ด้านบนเสมอ (ดู changelog v4)

  with torch.no_grad():
    for i in range(num_images):
      data = next(data_iter)
      im_data.resize_(data[0].size()).copy_(data[0])
      im_info.resize_(data[1].size()).copy_(data[1])
      gt_boxes.resize_(data[2].size()).copy_(data[2])
      num_boxes.resize_(data[3].size()).copy_(data[3])

      rois, bbox_pred, _ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
      boxes = rois.data[:, :, 1:5]

      image_id = imdb.image_index[i]
      debug_this = image_id in DEBUG_BUILDING_IDS
      if debug_this:
        logger.info(f"--- DEBUG {image_id} ---")
        logger.info(f"  candidate box (rois): {boxes[0,0,:4].cpu().numpy()}")
        logger.info(f"  im_info (h,w,scale): {im_info.data[0].cpu().numpy()}")

      if cfg.TEST.BBOX_REG:
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
          means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
          if args.cuda:
            stds, means = stds.cuda(), means.cuda()
          box_deltas = box_deltas.view(-1, 4) * stds + means
          box_deltas = box_deltas.view(1, -1, 4)
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
        pred_boxes = boxes

      # สำคัญมาก: หารด้วย im_scale เพื่อแปลงกลับเป็นพิกัดของ crop ต้นฉบับ
      pred_boxes = pred_boxes / im_info.data[0, 2].item()
      pred_box = pred_boxes[0, 0, :4].cpu().numpy()

      if image_id not in metadata:
        logger.info(f"⚠️  ไม่พบ metadata สำหรับ {image_id} - ข้าม")
        continue

      bf_xmin = metadata[image_id]["bf_xmin_crop_rel"]
      theta_deg = metadata[image_id]["theta_deg"]
      theta_rad = np.deg2rad(theta_deg)

      L_pred_px = bf_xmin - pred_box[0]
      h_pred = (L_pred_px * pixel_spacing_m) / np.tan(theta_rad)

      if debug_this:
        logger.info(f"  bf_xmin={bf_xmin}, pred_box[0]={pred_box[0]:.3f}, "
                     f"L_pred_px={L_pred_px:.3f}, h_pred={h_pred:.3f}")

      results.append({
          "building_id": image_id,
          "predicted_height_m": round(float(h_pred), 3),
          "theta_deg_used": round(theta_deg, 3),
          # แปลง predicted box จากพิกัด crop-relative -> พิกัดเต็มภาพ (full-image
          # pixel) โดยบวก crop origin กลับเข้าไป เพื่อใช้สร้าง polygon UTM ทีหลัง
          "box_xmin_px": float(pred_box[0]) + metadata[image_id].get("crop_xmin", 0),
          "box_ymin_px": float(pred_box[1]) + metadata[image_id].get("crop_ymin", 0),
          "box_xmax_px": float(pred_box[2]) + metadata[image_id].get("crop_xmin", 0),
          "box_ymax_px": float(pred_box[3]) + metadata[image_id].get("crop_ymin", 0),
      })

      if (i + 1) % 500 == 0:
        logger.info(f"ความคืบหน้า: {i+1}/{num_images}")

  elapsed = time.time() - start
  logger.info(f"ทำนายเสร็จทั้งหมด {len(results)} อาคาร ใช้เวลา {elapsed:.1f} วินาที")

  # --- บันทึก CSV ดิบไว้เผื่อ debug (ไม่บังคับ) ---
  output_csv = args.output_csv or os.path.join(output_dir, "predictions_raw.csv")
  with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["building_id", "predicted_height_m", "theta_deg_used",
                                             "box_xmin_px", "box_ymin_px", "box_xmax_px", "box_ymax_px"])
    writer.writeheader()
    writer.writerows(results)
  logger.info(f"บันทึก CSV ดิบที่ {output_csv}")

  # --- กรอง building_id ที่รู้ว่า mask เสียหาย (ถ้าระบุ) ---
  preds_df = pd.DataFrame(results)
  if args.degenerate_ids_path and os.path.exists(args.degenerate_ids_path):
    with open(args.degenerate_ids_path) as f:
      degenerate_ids = set(line.strip() for line in f if line.strip())
    before = len(preds_df)
    preds_df = preds_df[~preds_df["building_id"].isin(degenerate_ids)]
    logger.info(f"กรอง building_id ที่รู้ว่า mask เสียหาย: ตัดออก {before - len(preds_df)} "
                f"จาก {len(degenerate_ids)} รายการที่รู้จัก")

  # --- รวมกับ geometry เดิม ---
  logger.info("กำลังรวมผลทำนายกับ geometry อาคาร...")
  gdf = gpd.read_file(args.footprint_path)

  if args.id_column is not None:
    assert args.id_column in gdf.columns, (
        f"ระบุ --id_column='{args.id_column}' แต่ไม่พบใน footprint file - "
        f"columns ที่มีจริง: {list(gdf.columns)}"
    )
    id_col = args.id_column
  else:
    id_col = None
    for candidate in ["building_id", "building_i", "objectid", "gid", "fid", "globalId"]:
      if candidate in gdf.columns:
        id_col = candidate
        break
    assert id_col is not None, (
        f"ไม่พบ column ที่ใช้เป็น unique id ได้ (ลองหา building_id/building_i/"
        f"objectid/gid/fid/globalId แล้ว) - columns ที่มีจริง: {list(gdf.columns)} "
        f"- ระบุ --id_column ตรงๆ แทน"
    )

  # แปลง id เป็น int ผ่าน float ก่อน - ข้อมูลจากหน่วยงานภายนอกมัก
  # เก็บ objectid/gid เป็น float (เช่น 41892664.0) ทำให้ int() ตรงๆ พัง
  # (ValueError: invalid literal for int() with base 10: '41892664.0')
  # float() รับ decimal point ได้ แล้ว int() ตัดเศษทิ้ง - ปลอดภัยกับทั้ง
  # กรณี "00000042" (pipeline เราเอง) และ "41892664.0" (ข้อมูลภายนอก)
  preds_df["building_id_int"] = preds_df["building_id"].astype(float).astype("int64")
  gdf["_join_key"] = gdf[id_col].astype(float).astype("int64")

  merged = gdf.merge(
      preds_df[["building_id_int", "predicted_height_m", "theta_deg_used"]],
      left_on="_join_key", right_on="building_id_int", how="left",
  )
  merged = merged.drop(columns=["_join_key", "building_id_int"])

  # --- quality_flag: เช็คว่าผลทำนายสมเหตุสมผลไหม ---
  def classify(h):
    if pd.isna(h):
      return "no_prediction"
    if h <= 0:
      return "suspect_negative"
    if h > MAX_REASONABLE_HEIGHT_M:
      return "suspect_too_tall"
    return "ok"

  merged["quality_flag"] = merged["predicted_height_m"].apply(classify)

  n_matched = merged["predicted_height_m"].notna().sum()
  logger.info(f"จับคู่ได้: {n_matched} / {len(merged)} อาคาร")

  flag_counts = merged["quality_flag"].value_counts()
  logger.info("สรุป quality_flag:")
  for flag, count in flag_counts.items():
    logger.info(f"  {flag}: {count}")

  driver = "ESRI Shapefile" if args.output_geojson.lower().endswith(".shp") else "GeoJSON"
  merged.to_file(args.output_geojson, driver=driver)
  logger.info(f"บันทึกผลลัพธ์สุดท้ายที่ {args.output_geojson}")

  # --- เขียนไฟล์แยก: predicted bounding box เป็น polygon UTM ---
  # ใช้ตรวจสอบด้วยตาใน QGIS ได้ว่าโมเดลทำนายกล่องไปตรงไหนจริงๆ (ซ้อนทับกับ
  # ภาพ SAR ได้เลย) - ต่างจากไฟล์หลักที่เป็น footprint polygon เดิม
  if sar_transform_vals is not None:
    from shapely.geometry import Polygon
    from affine import Affine

    transform = Affine(*sar_transform_vals)
    bbox_rows = []
    for r in results:
      xmin, ymin = r["box_xmin_px"], r["box_ymin_px"]
      xmax, ymax = r["box_xmax_px"], r["box_ymax_px"]
      # 4 มุมของกล่อง: pixel -> UTM ด้วย affine transform
      corners_px = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
      corners_utm = [transform * (px, py) for px, py in corners_px]
      bbox_rows.append({
          "building_id": r["building_id"],
          "predicted_height_m": r["predicted_height_m"],
          "theta_deg_used": r["theta_deg_used"],
          "geometry": Polygon(corners_utm),
      })

    bbox_gdf = gpd.GeoDataFrame(bbox_rows, crs=sar_crs_str)
    stem, ext = os.path.splitext(args.output_geojson)
    bbox_output_path = f"{stem}_bbox{ext}"
    bbox_driver = "ESRI Shapefile" if bbox_output_path.lower().endswith(".shp") else "GeoJSON"
    bbox_gdf.to_file(bbox_output_path, driver=bbox_driver)
    logger.info(f"บันทึก predicted bounding box (polygon) ที่ {bbox_output_path}")

  heights = np.array([r["predicted_height_m"] for r in results])
  logger.info(f"สรุปความสูงที่ทำนาย: mean={heights.mean():.2f}m, median={np.median(heights):.2f}m, "
              f"min={heights.min():.2f}m, max={heights.max():.2f}m")


if __name__ == '__main__':
  main()
