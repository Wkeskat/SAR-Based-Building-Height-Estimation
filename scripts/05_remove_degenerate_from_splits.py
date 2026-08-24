"""
15_remove_degenerate_from_splits.py

ตัด building_id ที่เป็น degenerate mask (จาก 14_scan_degenerate_masks.py)
ออกจาก train.txt, val.txt, test.txt, buffer.txt - สำรองไฟล์เดิมไว้ก่อนเสมอ

หมายเหตุ: สคริปต์นี้แค่แก้ ImageSets/Main/*.txt เท่านั้น ไม่ได้ลบไฟล์ .png/.xml
จริงออกจาก JPEGImages/Annotations (เผื่อย้อนกลับ/ตรวจสอบทีหลัง) และไม่ได้
retrain โมเดลใหม่ - ถ้า building_id ที่ตัดออกอยู่ใน train.txt เดิม โมเดลที่
เทรนไปแล้วก็ยังมีผลจากข้อมูลนั้นอยู่ (ต้อง retrain ถ้าต้องการให้โมเดลไม่เคย
เห็นข้อมูลนี้เลยจริงๆ) - แต่ถ้าตัดออกจาก test.txt/val.txt จะทำให้ผลประเมิน
สะอาดขึ้นได้ทันทีโดยไม่ต้อง retrain
"""

import os
import shutil
import datetime

# ---- แก้ path ให้ตรงกับเครื่องคุณ ----
VOC_ROOT = r"E:\00_3D\14_SAR_\01_Script_\sar_pipeline"
DEGENERATE_IDS_PATH = r"E:\00_3D\14_SAR_\01_Script_\sar_pipeline\degenerate_building_ids.txt"

imagesets_dir = os.path.join(VOC_ROOT, "ImageSets", "Main")
SPLIT_FILES = ["train.txt", "val.txt", "test.txt", "buffer.txt"]


def remove_degenerate_ids():
    with open(DEGENERATE_IDS_PATH) as f:
        degenerate_ids = set(line.strip() for line in f if line.strip())
    print(f"จำนวน building_id ที่จะตัดออก: {len(degenerate_ids)}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total_removed = 0

    for split_file in SPLIT_FILES:
        split_path = os.path.join(imagesets_dir, split_file)
        if not os.path.exists(split_path):
            print(f"⚠️  ไม่พบ {split_file} - ข้าม")
            continue

        with open(split_path) as f:
            ids = [line.strip() for line in f if line.strip()]

        before_count = len(ids)
        kept_ids = [i for i in ids if i not in degenerate_ids]
        removed_count = before_count - len(kept_ids)
        total_removed += removed_count

        if removed_count == 0:
            print(f"{split_file}: ไม่มี id ที่ต้องตัดออก ({before_count} รายการเท่าเดิม)")
            continue

        backup_path = f"{split_path}.backup_{stamp}"
        shutil.copy(split_path, backup_path)
        print(f"สำรอง {split_file} ไว้ที่ {os.path.basename(backup_path)}")

        with open(split_path, "w") as f:
            f.write("\n".join(kept_ids))

        print(f"{split_file}: {before_count} -> {len(kept_ids)} รายการ "
              f"(ตัดออก {removed_count})")

    print()
    print(f"รวมตัดออกทั้งหมด {total_removed} รายการ จาก {len(degenerate_ids)} id "
          f"ที่มีในรายการ (บาง id อาจไม่ได้อยู่ใน split ไหนเลย เช่น ถ้าอยู่ใน "
          f"buffer zone ที่ถูกทิ้งไปแล้วตั้งแต่ตอน resplit)")

    print()
    print("หมายเหตุ: ถ้าตัดออกจาก train.txt ด้วย ต้อง retrain โมเดลใหม่ถ้าต้องการ "
          "ให้โมเดลไม่เคยเห็นข้อมูลนี้เลย - แค่ตัดจาก test.txt/val.txt ก็เพียงพอ "
          "สำหรับได้ผลประเมินที่สะอาดขึ้นทันทีโดยไม่ต้อง retrain")


if __name__ == "__main__":
    remove_degenerate_ids()
