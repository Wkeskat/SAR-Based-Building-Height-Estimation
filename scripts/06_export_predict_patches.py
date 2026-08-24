"""


สร้าง crop สำหรับ "ทำนาย" ความสูงอาคารในพื้นที่ใหม่ (ไม่รู้ความสูงจริง)
ต่างจาก 04_export_percrop_patches.py ตรงที่:
  - ไม่มี B_b (ไม่รู้ความสูง จึงคำนวณ layover label ไม่ได้ และไม่ต้องคำนวณ)
  - Margin ใช้สมมติฐาน "ความสูงสูงสุดที่เป็นไปได้" แทนการคำนวณจาก B_b จริง
  - เขียน dummy bounding box ลง XML (ไม่มีความหมาย ใช้แค่ให้ sar_building.py
    loader เดิม parse ได้โดยไม่ต้องแก้โค้ด - ตอน inference โมเดลไม่ใช้ gt_boxes
    จริงอยู่แล้ว)
  - บันทึก metadata (bf_xmin, incidence angle ต่ออาคาร) แยกไว้ต่างหาก สำหรับ
    แปลงผลทำนายกลับเป็นความสูงจริงใน predict.py

Dependencies: rasterio, geopandas, pyproj, numpy, pillow, lxml, scipy
"""

from __future__ import annotations

import os
import pickle
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from pyproj import Transformer
from lxml import etree
from scipy.interpolate import LinearNDInterpolator
from PIL import Image, ImageDraw
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from dataclasses import dataclass

SPEED_OF_LIGHT = 299_792_458.0


@dataclass
class PredictExportConfig:
    sar_tif_path: str
    footprint_shp_path: str          # shapefile/GeoJSON, ต้องมี column "building_id"
    georef_xml_path: str
    output_dir: str
    assumed_max_height_m: float = 100.0   # ใช้ก็ต่อเมื่อ override_margin_px เป็น None
    override_margin_px: int = None   # ระบุตรงๆ ให้ตรงกับ margin ที่ใช้ตอนเทรน
                                       # (สำคัญมาก! โมเดลนี้ไม่มี RoI-pooling จริง
                                       # - pooled_feat = base_feat ทั้งภาพเสมอ
                                       # (ดู faster_rcnn_giou.py forward()) จึง
                                       # ไม่ scale-invariant กับขนาด crop ถ้า margin
                                       # ตอน predict ต่างจากตอนเทรนมาก จะทำนายผิด
                                       # เพี้ยนอย่างเป็นระบบ (over-predict สม่ำเสมอ
                                       # ไม่ใช่ noise สุ่ม) - ควรตั้งค่านี้ให้เท่ากับ
                                       # margin จริงตอนเทรน (เช่น 376px ถ้าตรงกับ
                                       # log ตอนรัน 04_export_percrop_patches.py)
    pixel_spacing_m: float = 0.5
    max_nodata_fraction: float = 0.3
    percentile_clip: tuple = (2.0, 98.0)
    id_column: str = None   # ระบุตรงๆ ถ้ารู้ว่า column ไหนคือ unique id ที่ถูกต้อง
                             # (เช่น "objectid") - ถ้าไม่ระบุ จะลองหาอัตโนมัติจาก
                             # ชื่อ column ที่พบบ่อย (building_id, objectid, gid, ...)


# ---------------------------------------------------------------------------
# GEOREF.xml -> incidence angle interpolator (lon, lat) -> theta_deg
# ---------------------------------------------------------------------------

def parse_georef_incidence_interpolator(georef_xml_path: str) -> LinearNDInterpolator:
    tree = etree.parse(georef_xml_path)
    root = tree.getroot()

    lats, lons, incs = [], [], []
    for gp in root.findall(".//gridPoint"):
        lats.append(float(gp.find("lat").text))
        lons.append(float(gp.find("lon").text))
        incs.append(float(gp.find("inc").text))

    points = np.column_stack([lons, lats])
    return LinearNDInterpolator(points, np.array(incs))


# ---------------------------------------------------------------------------
# Footprint -> pixel (เหมือน Step 1 เดิม แต่ไม่ต้องมี height)
# ---------------------------------------------------------------------------

def load_footprints_pixel(cfg: PredictExportConfig, sar_transform, sar_crs):
    gdf = gpd.read_file(cfg.footprint_shp_path)

    # หา column ที่ใช้เป็น unique id ต่ออาคาร - รองรับหลาย convention:
    # "building_id"/"building_i" (จาก pipeline ของเราเอง, "building_i" คือกรณี
    # Shapefile ตัดชื่อเหลือ 10 ตัวอักษร) และ "objectid"/"gid"/"fid"/"globalId"
    # (convention ทั่วไปของข้อมูล GIS จากหน่วยงานราชการ/ESRI) - หรือระบุตรงๆ
    # ผ่าน cfg.id_column ถ้ารู้อยู่แล้วว่า column ไหนคือ unique id ที่ถูกต้อง
    if cfg.id_column is not None:
        assert cfg.id_column in gdf.columns, (
            f"ระบุ id_column='{cfg.id_column}' แต่ไม่พบใน footprint file - "
            f"columns ที่มีจริง: {list(gdf.columns)}"
        )
        id_col = cfg.id_column
    else:
        id_col = None
        for candidate in ["building_id", "building_i", "objectid", "gid", "fid", "globalId"]:
            if candidate in gdf.columns:
                id_col = candidate
                break
        assert id_col is not None, (
            f"ไม่พบ column ที่ใช้เป็น unique id ได้ (ลองหา building_id/building_i/"
            f"objectid/gid/fid/globalId แล้ว) - columns ที่มีจริง: {list(gdf.columns)} "
            f"- ระบุ id_column ตรงๆ ใน PredictExportConfig แทน"
        )
    if id_col != "building_id":
        print(f"หมายเหตุ: ใช้ column '{id_col}' แทน 'building_id' "
              f"(ชื่อถูกตัดจากข้อจำกัดของ Shapefile - ใช้ GeoJSON แทนถ้าต้องการเลี่ยงปัญหานี้)")

    if gdf.crs != sar_crs:
        gdf = gdf.to_crs(sar_crs)

    # เก็บ centroid ใน WGS84 ไว้คำนวณ incidence angle ต่ออาคาร
    to_wgs84 = Transformer.from_crs(sar_crs, "EPSG:4326", always_xy=True)

    inv_transform = ~sar_transform
    records = {}

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.exterior.coords) if geom.geom_type == "Polygon" else None
        if coords is None:
            continue  # ข้าม MultiPolygon เพื่อความง่าย (เหมือน Step 1 เดิม)

        pix = [inv_transform * (x, y) for x, y in coords]
        xs = [p[0] for p in pix]
        ys = [p[1] for p in pix]

        centroid_x, centroid_y = geom.centroid.x, geom.centroid.y
        lon, lat = to_wgs84.transform(centroid_x, centroid_y)

        records[row[id_col]] = {
            "pixel_coords": list(zip(xs, ys)),
            "bf_xmin": min(xs), "bf_ymin": min(ys),
            "bf_xmax": max(xs), "bf_ymax": max(ys),
            "lon": lon, "lat": lat,
        }

    return records


def to_uint8(band, pct_low, pct_high):
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    lo, hi = np.percentile(valid, [pct_low, pct_high])
    if hi <= lo:
        hi = lo + 1e-6
    stretched = np.clip((band - lo) / (hi - lo), 0, 1)
    return (stretched * 255).astype(np.uint8)


def write_dummy_voc_xml(xml_path, png_filename, w, h, dummy_box, building_id):
    """
    เขียน XML แบบเดียวกับ percrop เดิม แต่ bndbox เป็นแค่ placeholder (=B_f เอง)
    ไม่มีความหมายจริง เพราะตอน inference โมเดลไม่ใช้ gt_boxes อยู่แล้ว
    (roibatchLoader ตอน training=False จะ return dummy gt_boxes เองด้วยซ้ำ)
    """
    ann = Element("annotation")
    SubElement(ann, "folder").text = "JPEGImages"
    SubElement(ann, "filename").text = png_filename
    size = SubElement(ann, "size")
    SubElement(size, "width").text = str(w)
    SubElement(size, "height").text = str(h)
    SubElement(size, "depth").text = "3"
    SubElement(ann, "segmented").text = "0"

    o = SubElement(ann, "object")
    SubElement(o, "name").text = "building"
    SubElement(o, "pose").text = "Unspecified"
    SubElement(o, "truncated").text = "0"
    SubElement(o, "difficult").text = "0"
    SubElement(o, "building_id").text = str(building_id)
    bnd = SubElement(o, "bndbox")
    xmin, ymin, xmax, ymax = dummy_box
    SubElement(bnd, "xmin").text = str(int(round(xmin)))
    SubElement(bnd, "ymin").text = str(int(round(ymin)))
    SubElement(bnd, "xmax").text = str(int(round(xmax)))
    SubElement(bnd, "ymax").text = str(int(round(ymax)))

    xml_str = minidom.parseString(tostring(ann)).toprettyxml(indent="  ")
    with open(xml_path, "w") as f:
        f.write(xml_str)


def export_predict_patches(cfg: PredictExportConfig):
    jpeg_dir = os.path.join(cfg.output_dir, "JPEGImages")
    ann_dir = os.path.join(cfg.output_dir, "Annotations")
    imagesets_dir = os.path.join(cfg.output_dir, "ImageSets", "Main")
    os.makedirs(jpeg_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(imagesets_dir, exist_ok=True)

    inc_interp = parse_georef_incidence_interpolator(cfg.georef_xml_path)

    with rasterio.open(cfg.sar_tif_path) as src:
        img_width, img_height = src.width, src.height
        transform, crs = src.transform, src.crs
        nodata_value = src.nodata if src.nodata is not None else 0.0

        footprints = load_footprints_pixel(cfg, transform, crs)
        print(f"โหลดอาคารทั้งหมด {len(footprints)} หลัง")

        # margin: ใช้ override_margin_px ตรงๆ ถ้ามี (แนะนำ! ให้ตรงกับตอนเทรน
        # เป๊ะๆ) หรือคำนวณจากสมมติฐานความสูงสูงสุดถ้าไม่ได้ระบุ
        if cfg.override_margin_px is not None:
            margin_px = cfg.override_margin_px
            print(f"Margin: {margin_px}px (กำหนดตรงๆ ผ่าน override_margin_px "
                  f"- ควรตรงกับ margin ตอนเทรนโมเดลจริง)")
        else:
            theta_for_margin = 41.3
            max_L_px = (cfg.assumed_max_height_m * np.cos(np.deg2rad(theta_for_margin))) / cfg.pixel_spacing_m
            margin_px = int(max_L_px * 1.2) + 20
            print(f"Margin (จากสมมติฐานอาคารสูงสุด {cfg.assumed_max_height_m}m): {margin_px}px")
            print(f"⚠️  คำเตือน: ถ้า margin นี้ไม่ตรงกับตอนเทรนโมเดล (เช่น 376px) "
                  f"อาจทำให้ทำนายความสูงผิดเพี้ยนอย่างเป็นระบบ เพราะโมเดลนี้ไม่มี "
                  f"RoI-pooling จริง (ไม่ scale-invariant) - แนะนำใช้ override_margin_px "
                  f"แทนถ้ารู้ margin ที่ใช้ตอนเทรนแน่ชัด")

        metadata = {}  # building_id -> {bf_xmin_crop_rel, theta_deg}
        predict_ids = []
        n_written, n_skipped_nodata, n_skipped_geom = 0, 0, 0

        for bid, entry in footprints.items():
            bf_xmin, bf_ymin = entry["bf_xmin"], entry["bf_ymin"]
            bf_xmax, bf_ymax = entry["bf_xmax"], entry["bf_ymax"]

            crop_xmin = int(np.floor(bf_xmin - margin_px))
            crop_ymin = int(np.floor(bf_ymin - margin_px))
            crop_xmax = int(np.ceil(bf_xmax + margin_px))
            crop_ymax = int(np.ceil(bf_ymax + margin_px))

            if crop_xmin < 0:
                crop_xmax -= crop_xmin; crop_xmin = 0
            if crop_ymin < 0:
                crop_ymax -= crop_ymin; crop_ymin = 0
            if crop_xmax > img_width:
                shift = crop_xmax - img_width
                crop_xmin = max(0, crop_xmin - shift); crop_xmax = img_width
            if crop_ymax > img_height:
                shift = crop_ymax - img_height
                crop_ymin = max(0, crop_ymin - shift); crop_ymax = img_height

            if (crop_xmax - crop_xmin < (bf_xmax - bf_xmin) or
                    crop_ymax - crop_ymin < (bf_ymax - bf_ymin)):
                n_skipped_geom += 1
                continue

            win_w, win_h = crop_xmax - crop_xmin, crop_ymax - crop_ymin
            window = Window(crop_xmin, crop_ymin, win_w, win_h)
            band = src.read(1, window=window).astype(np.float32)

            is_nodata = np.isclose(band, nodata_value)
            if float(is_nodata.mean()) > cfg.max_nodata_fraction:
                n_skipped_nodata += 1
                continue

            sar_u8 = to_uint8(band, *cfg.percentile_clip)

            mask_img = Image.new("L", (win_w, win_h), 0)
            rel_poly = [(x - crop_xmin, y - crop_ymin) for x, y in entry["pixel_coords"]]
            ImageDraw.Draw(mask_img).polygon(rel_poly, fill=255)
            mask_u8 = np.array(mask_img, dtype=np.uint8)

            rgb = np.stack([sar_u8, mask_u8, sar_u8], axis=-1)

            bf_rel = (bf_xmin - crop_xmin, bf_ymin - crop_ymin,
                      bf_xmax - crop_xmin, bf_ymax - crop_ymin)

            theta_deg = float(inc_interp(entry["lon"], entry["lat"]))
            if np.isnan(theta_deg):
                theta_deg = theta_for_margin  # นอกขอบเขต grid - ใช้ค่ากลางแทน

            patch_name = f"{bid:08d}" if isinstance(bid, int) else str(bid)
            png_filename = f"{patch_name}.png"

            Image.fromarray(rgb, mode="RGB").save(os.path.join(jpeg_dir, png_filename))
            write_dummy_voc_xml(
                os.path.join(ann_dir, f"{patch_name}.xml"),
                png_filename, win_w, win_h, bf_rel, bid,
            )

            metadata[patch_name] = {
                "bf_xmin_crop_rel": bf_rel[0],
                "theta_deg": theta_deg,
                # เก็บ crop origin ไว้ด้วย เพื่อให้ predict.py แปลง predicted box
                # จากพิกัด crop-relative กลับเป็นพิกัดเต็มภาพ แล้วเป็น UTM ได้
                "crop_xmin": crop_xmin,
                "crop_ymin": crop_ymin,
            }
            predict_ids.append(patch_name)
            n_written += 1

    with open(os.path.join(imagesets_dir, "predict.txt"), "w") as f:
        f.write("\n".join(predict_ids))

    # เก็บ affine transform ของ raster ไว้ด้วย เพื่อให้ predict.py แปลง pixel
    # -> UTM ได้โดยไม่ต้องเปิด GeoTIFF ใหม่
    with open(os.path.join(cfg.output_dir, "predict_metadata.pkl"), "wb") as f:
        pickle.dump({
            "metadata": metadata,
            "pixel_spacing_m": cfg.pixel_spacing_m,
            "sar_transform": list(transform)[:6],   # affine 6 ค่า (a,b,c,d,e,f)
            "sar_crs": str(crs),
        }, f)

    print(f"เขียนไฟล์ทั้งหมด {n_written} อาคาร")
    print(f"ข้าม (nodata): {n_skipped_nodata}")
    print(f"ข้าม (geometry ไม่พอ): {n_skipped_geom}")
    print(f"Output: {cfg.output_dir}")


if __name__ == "__main__":
    cfg = PredictExportConfig(
        sar_tif_path=r"path\tiff",
        footprint_shp_path=r"path\footprint.geojson",
        georef_xml_path=r"path\.xml",
        output_dir=r"path\output",
        override_margin_px=376,   # ต้องตรงกับ margin ตอนเทรนโมเดลจริง (376px)
                                    # ไม่ใช้ assumed_max_height_m อีกต่อไป เพราะ
                                    # ให้ margin ผิด (200px) ทำให้ทำนายความสูง
                                    # ผิดเพี้ยนอย่างเป็นระบบ (ดูปัญหาที่เจอจริงมาแล้ว)
        id_column="objectid",     # ไฟล์นี้เป็นข้อมูลอาคารจากหน่วยงานราชการ
                                    # (columns แบบ BL_TYPE, BL_HEIGHT, ...) ไม่มี
                                    # "building_id" - ใช้ "objectid" เป็น unique id แทน
    )
    export_predict_patches(cfg)
