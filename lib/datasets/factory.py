"""Factory method for getting the SAR building imdb by name.

ดัดแปลงจาก datasets/factory.py ของ jwyang/faster-rcnn.pytorch
ตัดส่วน voc/coco/imagenet/vg ทิ้ง เหลือแค่ dataset ของเราเอง (sar_building)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from datasets.sar_building import sar_building

# ----------------------------------------------------------------------
# แก้ path นี้ให้ตรงกับเครื่องคุณ (path เดียวกับที่ใช้ใน Step 3-6 ทั้งหมด)
# เปลี่ยนจาก VOC_SAR (grid tiling เดิม) เป็น sar_pipeline (crop-per-building ใหม่)
# ----------------------------------------------------------------------
SAR_DATA_PATH = r"E:\00_3D\14_SAR_\01_Script_\sar_pipeline"

__sets = {}

# ลงทะเบียนชื่อ dataset ที่ train.py / test.py จะเรียกใช้
__sets['sar_building_train'] = (lambda: sar_building('train', SAR_DATA_PATH))
__sets['sar_building_val'] = (lambda: sar_building('val', SAR_DATA_PATH))
__sets['sar_building_test'] = (lambda: sar_building('test', SAR_DATA_PATH))

# ----------------------------------------------------------------------
# dataset สำหรับ predict.py (พื้นที่ใหม่ ไม่มีความสูงจริง) - ต้องรัน
# 10_export_predict_patches.py ก่อน เพื่อสร้างโฟลเดอร์นี้ (มี JPEGImages/,
# Annotations/, ImageSets/Main/predict.txt) - path มาจาก environment variable
# SAR_PREDICT_DATA_PATH (run_predict_pipeline.py ตั้งค่านี้ให้อัตโนมัติ)
# ----------------------------------------------------------------------
NEW_SCENE_DATA_PATH = os.environ.get(
    "SAR_PREDICT_DATA_PATH", r"path\to\new_scene_output_dir"
)
__sets['sar_building_predict'] = (lambda: sar_building('predict', NEW_SCENE_DATA_PATH))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}. ที่มี: {}'.format(name, list(__sets.keys())))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
