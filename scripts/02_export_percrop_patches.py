"""
export_percrop_patches.py

: B_f (pixel) + B_b (pixel) -> ONE crop per building + single-object VOC XML

เปลี่ยนจาก dense grid tiling (หลายอาคารต่อ patch) เป็น crop-per-building
(1 patch = 1 อาคาร) ให้ตรงกับ forward() จริงของโค้ดต้นฉบับ (faster_rcnn_giou.py):

    im1 = im_data.data[i][1]              # channel เขียว = footprint mask
    mask = (im1 - min) / ptp(im1)         # normalize [0,1] (ไม่ threshold เพราะ
                                           # เป็น binary mask อยู่แล้ว)
    ftpbox = extract_bboxes(mask)[0]      # หา box จาก mask โดยตรง ณ runtime
    ...
    RCNN_loss_bbox = Ciou_loss(pred_box, gt_boxes[:,0,:4])   # ใช้แค่ box แรกเท่านั้น

ดังนั้นแต่ละไฟล์ภาพต้องเป็น "1 อาคาร = 1 ภาพ" และต้องเป็นภาพ 3 channel จริง
(ไม่ใช่ grayscale เฉยๆ) เพราะ minibatch.py จะ replicate grayscale เป็น 3 channel
เหมือนกันหมด (mask จะหายไป) ถ้าเราเซฟเป็น grayscale ธรรมดา:

    im = imread(roidb[i]['image'])
    if len(im.shape) == 2:
        im = im[:,:,np.newaxis]
        im = np.concatenate((im,im,im), axis=2)   # <- ถ้า grayscale, 3 channel จะเหมือนกันหมด
    im = im[:,:,::-1]   # RGB->BGR (สลับแค่ channel 0 กับ 2, channel 1 ไม่ขยับ)

Channel layout ที่ต้องเซฟ (RGB, ก่อน BGR-flip):
    R (channel 0): SAR amplitude (8-bit stretched)
    G (channel 1): footprint mask (0/255 binary) <- โค้ดต้นฉบับอ่าน channel นี้
    B (channel 2): SAR amplitude อีกรอบ (duplicate เพื่อให้เปิดดูเป็นภาพปกติได้)

ต้องเซฟเป็น .png (lossless) ไม่ใช่ .jpg เพราะ JPEG compression จะทำให้ mask
channel เบลอ (ค่า 0/255 จะกลายเป็นค่ากลางๆ) ทำให้ extract_bboxes คำนวณ box ผิด

Dependencies: rasterio, geopandas, numpy, pillow, pickle
"""

from __future__ import annotations

import os
import time
import logging
import pickle
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from PIL import Image, ImageDraw
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class PerCropExportConfig:
    sar_tif_path: str
    footprints_pkl_path: str
    bbox_geojson_path: str
    output_dir: str
    margin_px: int = 0          # 0 = auto-compute from data (see main())
    percentile_clip: tuple = (2.0, 98.0)
    max_nodata_fraction: float = 0.3   # ข้าม crop ถ้า nodata เกิน 30% ของพื้นที่
    warn_nodata_fraction: float = 0.1  # เตือน (แต่ยังเขียนไฟล์) ถ้าเกิน 10%
    progress_every: int = 500          # log ความคืบหน้าทุกๆ N อาคาร


def setup_logger(output_dir: str) -> logging.Logger:
    """
    ตั้งค่า logger ให้เขียนทั้งหน้าจอ (console) และไฟล์ log
    (output_dir/export_log_<timestamp>.txt) เพื่อให้ review รายละเอียดย้อนหลังได้
    หลัง run เสร็จ (มีประโยชน์มากตอน run กับข้อมูลจริงหลักหมื่นอาคาร ที่ terminal
    output อาจ scroll หายไปหมดแล้ว)
    """
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"export_log_{stamp}.txt")

    logger = logging.getLogger("percrop_export")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # กัน handler ซ้อนถ้าเรียกซ้ำในเซสชันเดียวกัน

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logger.info(f"Log file: {log_path}")
    return logger


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


# ---------------------------------------------------------------------------
# Reuse the UTM->pixel merge logic from Step 4 (grid version) - unchanged
# ---------------------------------------------------------------------------

def load_building_boxes_pixel(cfg: PerCropExportConfig, sar_transform, sar_crs):
    """
    Returns dict: building_id -> {
        'bb': (xmin, ymin, xmax, ymax),        # B_b, pixel
        'bf': (xmin, ymin, xmax, ymax),        # B_f, pixel
        'footprint_px': [(x,y), ...],          # footprint polygon, pixel coords
        'height': float,
    }
    """
    with open(cfg.footprints_pkl_path, "rb") as f:
        fp_data = pickle.load(f)

    bf_by_id = {}
    for rec in fp_data["records"]:
        bf_by_id[rec["building_id"]] = {
            "bbox": (rec["bf_xmin"], rec["bf_ymin"], rec["bf_xmax"], rec["bf_ymax"]),
            "footprint_px": rec.get("pixel_coords"),
            "height": rec.get("height"),
        }

    gdf = gpd.read_file(cfg.bbox_geojson_path)
    if gdf.crs != sar_crs:
        gdf = gdf.to_crs(sar_crs)

    assert "building_id" in gdf.columns, (
        f"building_bbox_labels.geojson ต้องมี property 'building_id' "
        f"columns ที่มีจริง: {list(gdf.columns)}"
    )

    inv_transform = ~sar_transform
    merged = {}

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.exterior.coords) if geom.geom_type == "Polygon" else None
        if coords is None:
            raise ValueError(f"ต้องการ Polygon, ได้ {geom.geom_type}")
        pix = [inv_transform * (x, y) for x, y in coords]
        xs = [p[0] for p in pix]
        ys = [p[1] for p in pix]
        bb_pixel = (min(xs), min(ys), max(xs), max(ys))

        bid = row["building_id"]
        bf_entry = bf_by_id.get(bid)
        if bf_entry is None:
            continue  # ไม่มี B_f คู่กัน ข้ามอาคารนี้ไป

        merged[bid] = {
            "bb": bb_pixel,
            "bf": bf_entry["bbox"],
            "footprint_px": bf_entry["footprint_px"],
            "height": bf_entry["height"],
        }

    return merged


def to_uint8(band: np.ndarray, pct_low: float, pct_high: float) -> np.ndarray:
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    lo, hi = np.percentile(valid, [pct_low, pct_high])
    if hi <= lo:
        hi = lo + 1e-6
    stretched = np.clip((band - lo) / (hi - lo), 0, 1)
    return (stretched * 255).astype(np.uint8)


def write_single_object_voc_xml(xml_path, jpg_filename, w, h, bbox, building_id):
    ann = Element("annotation")
    SubElement(ann, "folder").text = "JPEGImages"
    SubElement(ann, "filename").text = jpg_filename
    size = SubElement(ann, "size")
    SubElement(size, "width").text = str(w)
    SubElement(size, "height").text = str(h)
    SubElement(size, "depth").text = "3"   # 3-channel: SAR, mask, SAR
    SubElement(ann, "segmented").text = "0"

    o = SubElement(ann, "object")
    SubElement(o, "name").text = "building"
    SubElement(o, "pose").text = "Unspecified"
    SubElement(o, "truncated").text = "0"
    SubElement(o, "difficult").text = "0"
    SubElement(o, "building_id").text = str(building_id)
    bnd = SubElement(o, "bndbox")
    xmin, ymin, xmax, ymax = bbox
    SubElement(bnd, "xmin").text = str(int(round(xmin)))
    SubElement(bnd, "ymin").text = str(int(round(ymin)))
    SubElement(bnd, "xmax").text = str(int(round(xmax)))
    SubElement(bnd, "ymax").text = str(int(round(ymax)))

    xml_str = minidom.parseString(tostring(ann)).toprettyxml(indent="  ")
    with open(xml_path, "w") as f:
        f.write(xml_str)


def export_percrop_patches(cfg: PerCropExportConfig):
    jpeg_dir = os.path.join(cfg.output_dir, "JPEGImages")
    ann_dir = os.path.join(cfg.output_dir, "Annotations")
    os.makedirs(jpeg_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    logger = setup_logger(cfg.output_dir)
    t_start = time.time()
    logger.info(f"เริ่มรัน: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with rasterio.open(cfg.sar_tif_path) as src:
        img_width, img_height = src.width, src.height
        transform, crs = src.transform, src.crs
        nodata_value = src.nodata if src.nodata is not None else 0.0
        logger.info(f"NoData value ที่ใช้เช็ค: {nodata_value} "
                    f"({'จาก metadata ของภาพ' if src.nodata is not None else 'ไม่มีใน metadata, ใช้ 0.0 เป็นค่า default'})")

        buildings = load_building_boxes_pixel(cfg, transform, crs)
        total = len(buildings)
        logger.info(f"โหลดอาคารทั้งหมด {total} หลัง")

        # margin อัตโนมัติ: ต้องครอบคลุม layover ที่ยาวที่สุดในชุดข้อมูล
        # (L = h*cos(theta) หรือ h/tan(theta) แล้วแต่ slant-range/ground-range -
        # ที่นี่ประมาณจาก bb-bf ที่ขยายจริงในข้อมูล ซึ่ง robust กว่าเพราะไม่ต้อง
        # รู้สูตร/มุมตกกระทบที่แน่นอนอีกที)
        if cfg.margin_px <= 0:
            max_extension = max(
                (e["bf"][0] - e["bb"][0]) for e in buildings.values()
            )
            margin_px = int(max_extension * 1.5) + 20  # เผื่อ 50% + context เพิ่ม 20px
            logger.info(f"Margin อัตโนมัติ: {margin_px}px (จาก layover สูงสุด {max_extension:.1f}px)")
        else:
            margin_px = cfg.margin_px

        n_written, n_skipped, n_skipped_nodata, n_warned = 0, 0, 0, 0
        nodata_warnings = []
        t_loop_start = time.time()

        for idx, (bid, entry) in enumerate(buildings.items(), start=1):
            bf = entry["bf"]
            bb = entry["bb"]

            # union ของ B_f และ B_b (กันเคสที่ B_b ไม่ครอบ B_f สนิท)
            union_xmin = min(bf[0], bb[0])
            union_ymin = min(bf[1], bb[1])
            union_xmax = max(bf[2], bb[2])
            union_ymax = max(bf[3], bb[3])

            crop_xmin = int(np.floor(union_xmin - margin_px))
            crop_ymin = int(np.floor(union_ymin - margin_px))
            crop_xmax = int(np.ceil(union_xmax + margin_px))
            crop_ymax = int(np.ceil(union_ymax + margin_px))

            # clip เข้าขอบภาพ โดยเลื่อน window แทนการตัดถ้าเป็นไปได้ ให้ box
            # ยังอยู่ในเฟรมครบ ไม่ถูก truncate
            crop_w = crop_xmax - crop_xmin
            crop_h = crop_ymax - crop_ymin

            if crop_xmin < 0:
                crop_xmax -= crop_xmin
                crop_xmin = 0
            if crop_ymin < 0:
                crop_ymax -= crop_ymin
                crop_ymin = 0
            if crop_xmax > img_width:
                shift = crop_xmax - img_width
                crop_xmin = max(0, crop_xmin - shift)
                crop_xmax = img_width
            if crop_ymax > img_height:
                shift = crop_ymax - img_height
                crop_ymin = max(0, crop_ymin - shift)
                crop_ymax = img_height

            # ถ้ายัง clip ไม่พอ (อาคารใหญ่กว่าภาพทั้งใบ - แทบเป็นไปไม่ได้) ข้ามไป
            if (crop_xmax - crop_xmin < (union_xmax - union_xmin) or
                    crop_ymax - crop_ymin < (union_ymax - union_ymin)):
                n_skipped += 1
                continue

            win_w = crop_xmax - crop_xmin
            win_h = crop_ymax - crop_ymin

            window = Window(crop_xmin, crop_ymin, win_w, win_h)
            band = src.read(1, window=window).astype(np.float32)

            # เช็ค nodata ก่อนแปลงเป็น 8-bit (แปลงแล้วจะแยกยากว่าอันไหน nodata
            # จริง เพราะ percentile stretch อาจทำให้ nodata=0 ปนกับค่ามืดจริง)
            if np.isnan(nodata_value):
                is_nodata = np.isnan(band)
            else:
                is_nodata = np.isclose(band, nodata_value)
            nodata_fraction = float(is_nodata.mean())

            if nodata_fraction > cfg.max_nodata_fraction:
                n_skipped_nodata += 1
                continue  # crop ส่วนใหญ่อยู่นอกขอบเขตภาพจริง (nodata) - ข้าม
            elif nodata_fraction > cfg.warn_nodata_fraction:
                n_warned += 1
                nodata_warnings.append((bid, round(nodata_fraction, 3)))

            sar_u8 = to_uint8(band, *cfg.percentile_clip)

            # rasterize footprint polygon -> binary mask, crop-relative coords
            mask_img = Image.new("L", (win_w, win_h), 0)
            if entry["footprint_px"]:
                rel_poly = [(x - crop_xmin, y - crop_ymin) for x, y in entry["footprint_px"]]
                ImageDraw.Draw(mask_img).polygon(rel_poly, fill=255)
            mask_u8 = np.array(mask_img, dtype=np.uint8)

            # compose RGB: R=SAR, G=mask, B=SAR (ก่อน BGR-flip ตอนเทรน channel 1
            # ยังเป็น mask เหมือนเดิมไม่ว่าจะ flip หรือไม่ เพราะ index กลางไม่ขยับ)
            rgb = np.stack([sar_u8, mask_u8, sar_u8], axis=-1)

            # bbox (B_b) เป็น crop-relative coords สำหรับ label
            bb_rel = (bb[0] - crop_xmin, bb[1] - crop_ymin,
                      bb[2] - crop_xmin, bb[3] - crop_ymin)

            patch_name = f"{bid:08d}" if isinstance(bid, int) else str(bid)
            png_filename = f"{patch_name}.png"

            Image.fromarray(rgb, mode="RGB").save(os.path.join(jpeg_dir, png_filename))
            write_single_object_voc_xml(
                os.path.join(ann_dir, f"{patch_name}.xml"),
                png_filename, win_w, win_h, bb_rel, bid,
            )
            n_written += 1

            if cfg.progress_every and idx % cfg.progress_every == 0:
                elapsed = time.time() - t_loop_start
                rate = idx / elapsed  # buildings/sec
                remaining = (total - idx) / rate if rate > 0 else float("nan")
                logger.info(
                    f"ความคืบหน้า: {idx}/{total} ({idx/total*100:.1f}%) | "
                    f"เขียนแล้ว {n_written} | ข้าม {n_skipped + n_skipped_nodata} | "
                    f"ใช้เวลาไปแล้ว {format_duration(elapsed)} | "
                    f"เหลืออีกประมาณ {format_duration(remaining)}"
                )

    t_end = time.time()
    total_elapsed = t_end - t_start

    logger.info("=" * 60)
    logger.info(f"เขียนไฟล์ทั้งหมด {n_written} อาคาร")
    logger.info(f"ข้าม (crop ไม่พอ อาคารใหญ่กว่าภาพ): {n_skipped}")
    logger.info(f"ข้าม (nodata เกิน {cfg.max_nodata_fraction*100:.0f}%): {n_skipped_nodata}")
    if n_warned:
        logger.info(f"⚠️  เตือน (nodata {cfg.warn_nodata_fraction*100:.0f}-{cfg.max_nodata_fraction*100:.0f}%, "
                     f"ยังเขียนไฟล์): {n_warned} อาคาร เช่น: {nodata_warnings[:5]}")
    logger.info(f"Output: {jpeg_dir}  /  {ann_dir}")
    logger.info(f"เวลาทั้งหมดที่ใช้: {format_duration(total_elapsed)} "
                f"(เฉลี่ย {total_elapsed/total*1000:.1f} ms/อาคาร)")
    logger.info(f"เสร็จสิ้น: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



if __name__ == "__main__":
    cfg = PerCropExportConfig(
        sar_tif_path=r"path\IMAGE_SRA.tif",  # Block=256x256 <- updated
        footprints_pkl_path=r"path\footprints_pixel.pkl",
        bbox_geojson_path=r"path\building_bbox_labels.geojson",
        output_dir=r"path\floder_pipeline",
        margin_px=0,   # 0 = auto-compute from your data's max layover extension
    )
    export_percrop_patches(cfg)
