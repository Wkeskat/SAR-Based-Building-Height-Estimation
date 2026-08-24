"""
10_resplit_geographic_trainvaltest.py

ขยาย 09_resplit_geographic_percrop.py จาก 2-way split (train/test) เป็น
3-way split (train/val/test) โดยมี 2 buffer zone (train-val และ val-test)
ใช้หลักการเดียวกัน: ตัดตามตำแหน่ง column บนภาพ (ซ้าย=train, กลาง=val, ขวา=test)
พร้อมกันชนที่แต่ละรอยต่อ ป้องกัน leakage

TRAIN_RATIO + VAL_RATIO + TEST_RATIO ต้องรวมกันได้ 1.0
"""

import os
import shutil
import datetime
import numpy as np
import pickle
import rasterio
import geopandas as gpd

# ---- แก้ให้ตรงกับเครื่องคุณ ----
VOC_ROOT = r"E:\00_3D\14_SAR_\01_Script_\sar_pipeline"
SAR_TIF_PATH = r"E:\00_3D\14_SAR_\00_DATA_\IMAGE_HH_SRA_spot_055_tiled.tif"
FOOTPRINTS_PKL_PATH = r"E:\00_3D\14_SAR_\00_DATA_\footprints_pixel.pkl"
BBOX_GEOJSON_PATH = r"E:\00_3D\14_SAR_\00_DATA_\building_bbox_labels.geojson"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15   # TRAIN_RATIO + VAL_RATIO + TEST_RATIO ต้องรวมกันได้ 1.0

EXTRA_BUFFER_PX = 50   # กันชนเพิ่มเติมนอกเหนือจากขอบ crop จริง (เผื่อ receptive field ของ CNN)

ann_dir = os.path.join(VOC_ROOT, "Annotations")
imagesets_dir = os.path.join(VOC_ROOT, "ImageSets", "Main")

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, (
    "TRAIN_RATIO + VAL_RATIO + TEST_RATIO ต้องรวมกันได้ 1.0"
)


def load_building_positions():
    """
    คืนค่า dict: building_id -> {
        'crop_xmin': ..., 'crop_xmax': ...,   # ขอบเขตจริงของ crop บนภาพเต็ม (full-image px)
        'margin_px': ...,                      # margin ที่ใช้คำนวณตอน export (สูตรเดียวกัน)
    }
    รวมถึง image_width สำหรับคำนวณเส้นแบ่ง
    """
    with open(FOOTPRINTS_PKL_PATH, "rb") as f:
        fp_data = pickle.load(f)

    bf_by_id = {}
    for rec in fp_data["records"]:
        bf_by_id[rec["building_id"]] = (
            rec["bf_xmin"], rec["bf_ymin"], rec["bf_xmax"], rec["bf_ymax"]
        )

    image_width = fp_data["image_width"]

    with rasterio.open(SAR_TIF_PATH) as src:
        raster_crs = src.crs
        inv_transform = ~src.transform

    gdf = gpd.read_file(BBOX_GEOJSON_PATH)
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    bb_by_id = {}
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.exterior.coords)
        pix = [inv_transform * (x, y) for x, y in coords]
        xs = [p[0] for p in pix]
        bb_by_id[row["building_id"]] = (min(xs), max(xs))

    # margin เดียวกับที่ 04_export_percrop_patches.py ใช้ (สูตรเหมือนกันเป๊ะ
    # เพื่อให้ได้ margin_px ค่าเดียวกัน - ถ้า Step 4 รันด้วย margin_px=0 (auto))
    max_extension = max(
        (bf[0] - bb_by_id[bid][0]) for bid, bf in bf_by_id.items() if bid in bb_by_id
    )
    margin_px = int(max_extension * 1.5) + 20
    print(f"Margin ที่คำนวณใหม่ (ต้องตรงกับตอน export Step 4): {margin_px}px")

    positions = {}
    for bid, bf in bf_by_id.items():
        if bid not in bb_by_id:
            continue
        bb_xmin, bb_xmax = bb_by_id[bid]
        union_xmin = min(bf[0], bb_xmin)
        union_xmax = max(bf[2], bb_xmax)
        positions[bid] = {
            "crop_xmin": union_xmin - margin_px,
            "crop_xmax": union_xmax + margin_px,
        }

    return positions, image_width, margin_px


def parse_building_ids_from_annotations():
    ids = []
    for fname in os.listdir(ann_dir):
        if not fname.endswith(".xml"):
            continue
        stem = os.path.splitext(fname)[0]
        try:
            bid = int(stem)
        except ValueError:
            print(f"⚠️  ข้ามไฟล์ที่ชื่อไม่ใช่ building_id ตัวเลข: {fname}")
            continue
        ids.append((stem, bid))
    return ids


def geographic_split(patch_ids, positions, image_width, train_ratio, val_ratio,
                      extra_buffer_px):
    # ใช้ percentile ของตำแหน่งอาคารที่ "รอดจริง" (survivors หลังกรอง nodata)
    # แทนสัดส่วนคงที่ของความกว้างภาพทั้งใบ - เพราะข้อมูลจริงอาจไม่กระจายเต็มความ
    # กว้างภาพ (เช่น กรณีนี้ label กระจุกอยู่แค่ 24.6%-86.7% ของความกว้างภาพ
    # เนื่องจาก nodata filter ตัดพื้นที่ขอบขวาของภาพออกไปเยอะ)
    survivor_centers = np.array([
        (positions[bid]["crop_xmin"] + positions[bid]["crop_xmax"]) / 2
        for _, bid in patch_ids if bid in positions
    ])

    col_split1 = np.percentile(survivor_centers, train_ratio * 100)
    col_split2 = np.percentile(survivor_centers, (train_ratio + val_ratio) * 100)

    print(f"ความกว้างภาพเต็ม: {image_width} px (ใช้อ้างอิงเท่านั้น ไม่ได้ใช้คำนวณเส้นแบ่งโดยตรง)")
    print(f"ตำแหน่งอาคารที่รอดจริง: min={survivor_centers.min():.0f}px, "
          f"max={survivor_centers.max():.0f}px")
    print(f"เส้นแบ่งที่ 1 (train|val), percentile {train_ratio*100:.0f}: {col_split1:.0f} px")
    print(f"เส้นแบ่งที่ 2 (val|test), percentile {(train_ratio+val_ratio)*100:.0f}: {col_split2:.0f} px")
    print(f"buffer เพิ่มเติมที่แต่ละรอยต่อ: ±{extra_buffer_px} px")

    train_ids, val_ids, test_ids, dropped_ids = [], [], [], []
    missing_position = []

    for stem, bid in patch_ids:
        pos = positions.get(bid)
        if pos is None:
            missing_position.append(stem)
            continue

        crop_xmin, crop_xmax = pos["crop_xmin"], pos["crop_xmax"]

        if crop_xmax <= col_split1 - extra_buffer_px:
            train_ids.append(stem)
        elif crop_xmin >= col_split1 + extra_buffer_px and crop_xmax <= col_split2 - extra_buffer_px:
            val_ids.append(stem)
        elif crop_xmin >= col_split2 + extra_buffer_px:
            test_ids.append(stem)
        else:
            dropped_ids.append(stem)   # อยู่ในโซนกันชนของรอยต่อใดรอยต่อหนึ่ง

    if missing_position:
        print(f"⚠️  {len(missing_position)} ไฟล์ไม่พบตำแหน่งใน footprints_pixel.pkl/"
              f"building_bbox_labels.geojson (building_id ไม่ตรงกัน) เช่น: {missing_position[:5]}")

    print(f"Train: {len(train_ids)} patch")
    print(f"Val: {len(val_ids)} patch")
    print(f"Test: {len(test_ids)} patch")
    print(f"ทิ้ง (โซนกันชน x2 รอยต่อ): {len(dropped_ids)} patch")

    return train_ids, val_ids, test_ids, dropped_ids


def export_buffer_shapefile(dropped_ids, shapefile_path):
    """
    Export อาคารในโซนกันชน (buffer.txt) เป็น shapefile เพื่อดูตำแหน่งจริงใน
    GIS software (QGIS/ArcGIS) - ใช้ geometry จริงจาก building_bbox_labels.geojson
    (ไฟล์เดียวกับที่ใช้คำนวณตำแหน่ง ไม่ต้องเปิดไฟล์ใหม่)

    หมายเหตุ: Shapefile ตัดชื่อ column เหลือ 10 ตัวอักษร - "building_id"
    จะกลายเป็น "building_i" ตอนเปิดใน QGIS (ข้อจำกัดของ format เอง แก้ไม่ได้
    ถ้าไม่เปลี่ยนไปใช้ GeoJSON แทน)
    """
    dropped_bids = set()
    for stem in dropped_ids:
        try:
            dropped_bids.add(int(stem))
        except ValueError:
            dropped_bids.add(stem)  # เผื่อ building_id ไม่ใช่ตัวเลขล้วน

    gdf = gpd.read_file(BBOX_GEOJSON_PATH)
    buffer_gdf = gdf[gdf["building_id"].isin(dropped_bids)]

    print(f"อาคารในโซนกันชนที่หา geometry เจอ: {len(buffer_gdf)} / {len(dropped_ids)}")
    if len(buffer_gdf) < len(dropped_ids):
        print(f"⚠️  บาง building_id ใน buffer.txt ไม่พบใน {BBOX_GEOJSON_PATH} "
              f"(อาจเป็นเพราะ building_id ไม่ตรงกัน - ตรวจสอบ dtype/รูปแบบ)")

    buffer_gdf.to_file(shapefile_path, driver="ESRI Shapefile")
    print(f"บันทึก shapefile ของโซนกันชนที่ {shapefile_path}")


def main():
    positions, image_width, margin_px = load_building_positions()
    patch_ids = parse_building_ids_from_annotations()
    print(f"อ่าน patch id จาก Annotations ได้ {len(patch_ids)} ไฟล์\n")

    train_ids, val_ids, test_ids, dropped_ids = geographic_split(
        patch_ids, positions, image_width, TRAIN_RATIO, VAL_RATIO, EXTRA_BUFFER_PX
    )

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(imagesets_dir, exist_ok=True)

    train_path = os.path.join(imagesets_dir, "train.txt")
    val_path = os.path.join(imagesets_dir, "val.txt")
    test_path = os.path.join(imagesets_dir, "test.txt")
    buffer_path = os.path.join(imagesets_dir, "buffer.txt")
    buffer_shp_path = os.path.join(imagesets_dir, "buffer.shp")

    for path in (train_path, val_path, test_path, buffer_path):
        if os.path.exists(path):
            backup_path = f"{path}.backup_{stamp}"
            shutil.copy(path, backup_path)
            print(f"สำรอง {os.path.basename(path)} ไว้ที่ {os.path.basename(backup_path)}")

    with open(train_path, "w") as f:
        f.write("\n".join(train_ids))
    with open(val_path, "w") as f:
        f.write("\n".join(val_ids))
    with open(test_path, "w") as f:
        f.write("\n".join(test_ids))
    with open(buffer_path, "w") as f:
        f.write("\n".join(dropped_ids))

    print(f"\nเขียน train.txt ({len(train_ids)}), val.txt ({len(val_ids)}), "
          f"test.txt ({len(test_ids)}), buffer.txt ({len(dropped_ids)}) เรียบร้อยแล้ว")

    if dropped_ids:
        export_buffer_shapefile(dropped_ids, buffer_shp_path)


if __name__ == "__main__":
    main()
