"""
14_scan_degenerate_masks.py

สแกนหา degenerate mask (เหมือนที่เจอใน building 62639) ทั่วทั้ง dataset
(train+val+test รวมกัน) โดยไม่ต้อง reprocess/regenerate อะไรเลย - แค่เปิดไฟล์
PNG ที่มีอยู่แล้วมาเช็ค เร็วกว่าการรัน Step 4 ใหม่มาก

ถ้าเจอไม่กี่ตัว: ทางแก้ที่คุ้มที่สุดคือตัด building_id เหล่านั้นออกจาก
train.txt/val.txt/test.txt ตรงๆ ไม่ต้อง reprocess ทั้ง dataset
"""

import os
import numpy as np
from PIL import Image

# ---- แก้ path ให้ตรงกับเครื่องคุณ ----
VOC_ROOT = r"E:\00_3D\14_SAR_\01_Script_\sar_pipeline"
MIN_MASK_PIXELS = 20  # เกณฑ์เดียวกับที่ใช้ใน sar_building.py v5


def scan_degenerate_masks():
    jpeg_dir = os.path.join(VOC_ROOT, "JPEGImages")
    all_files = [f for f in os.listdir(jpeg_dir) if f.endswith(".png")]
    print(f"จำนวนภาพทั้งหมดใน dataset: {len(all_files)}")

    degenerate_ids = []
    mask_areas = []

    for i, fname in enumerate(all_files):
        im = np.array(Image.open(os.path.join(jpeg_dir, fname)))
        mask = im[:, :, 1]
        mask_area = int((mask > 0).sum())
        mask_areas.append(mask_area)

        is_degenerate = bool(np.ptp(mask) == 0) or (mask_area < MIN_MASK_PIXELS)
        if is_degenerate:
            degenerate_ids.append((os.path.splitext(fname)[0], mask_area))

        if (i + 1) % 10000 == 0:
            print(f"สแกนแล้ว {i+1}/{len(all_files)}")

    mask_areas = np.array(mask_areas)

    print()
    print("=" * 60)
    print("ผลการสแกน")
    print("=" * 60)
    print(f"จำนวนภาพที่ degenerate mask: {len(degenerate_ids)} / {len(all_files)} "
          f"({len(degenerate_ids)/len(all_files)*100:.3f}%)")
    print(f"Mask area: min={mask_areas.min()}, median={np.median(mask_areas):.0f}, "
          f"max={mask_areas.max()}")

    if degenerate_ids:
        print()
        print("รายชื่อ building_id ที่ degenerate (แนะนำตัดออกจาก train/val/test.txt):")
        for bid, area in degenerate_ids:
            print(f"  {bid}: mask area = {area} pixels")

        out_path = os.path.join(VOC_ROOT, "degenerate_building_ids.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(bid for bid, _ in degenerate_ids))
        print()
        print(f"บันทึกรายชื่อไว้ที่ {out_path}")
        print("นำไฟล์นี้ไปตัด building_id ที่ตรงกันออกจาก train.txt/val.txt/test.txt "
              "ด้วยมือ หรือรัน script ตัดอัตโนมัติถ้าต้องการ")
    else:
        print()
        print("✅ ไม่พบ degenerate mask อื่นเลยในทั้ง dataset - "
              "building 62639 อาจเป็นเคสเดียวที่มีปัญหานี้จริงๆ")


if __name__ == "__main__":
    scan_degenerate_masks()
