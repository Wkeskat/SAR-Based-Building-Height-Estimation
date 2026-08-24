from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.sparse
import pickle
import xml.etree.ElementTree as ET
from PIL import Image

from .imdb import imdb
from model.utils.config import cfg


class sar_building(imdb):
    """
    Dataset class สำหรับข้อมูล SAR building bbox regression
    (ดัดแปลงจาก pascal_voc.py ของ jwyang/faster-rcnn.pytorch)

    ต่างจาก pascal_voc ต้นฉบับตรงนี้:
      - มีแค่ 1 class คือ 'building' (ไม่ใช่ 20 class ของ VOC ทั่วไป)
      - รูปภาพเป็น .png (ไม่ใช่ .jpg)
      - data_path รับ path เต็มโดยตรง ไม่ต่อ 'VOC' + year แบบ pascal_voc เดิม
        (เพราะโฟลเดอร์ของเราชื่อ VOC_SAR ไม่ใช่ VOC2007/VOC2012)
    """

    def __init__(self, image_set, data_path):
        imdb.__init__(self, 'sar_building_' + image_set)
        self._image_set = image_set          # 'train' หรือ 'test'
        self._data_path = data_path          # path เต็มไปยัง VOCdevkit/VOC_SAR

        self._classes = ('__background__',   # index 0 เสมอ (ตามธรรมเนียม Faster R-CNN)
                          'building')
        self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
        self._image_index = self._load_image_set_index()

        # หา extension จริงจากไฟล์แรกใน image_index แทนการ hardcode
        # (export_voc_patches.py เขียนเป็น .jpg แต่โค้ดเดิม hardcode .png ไว้ -
        #  ถ้าไม่ตรงกัน image_path_from_index() จะ assert fail ทุกภาพ)
        self._image_ext = self._detect_image_ext()

        self._roidb_handler = self.gt_roidb

        self.config = {
            'cleanup': True,
            'use_salt': True,
            'use_diff': False,
            'matlab_eval': False,
            'rpn_file': None,
            'min_size': 2,
        }

        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    # ------------------------------------------------------------------
    def _detect_image_ext(self):
        """
        เช็ค extension จริงของไฟล์ภาพ โดยดูจาก patch id แรกใน image_index
        รองรับทั้ง .jpg และ .png เพื่อกันปัญหา mismatch ระหว่าง export script
        กับ dataset loader (ปัญหานี้เกิดขึ้นจริงระหว่างพัฒนา pipeline นี้)
        """
        assert self._image_index, 'image_index ว่างเปล่า - เช็ค ImageSets/Main/{}.txt'.format(
            self._image_set)
        first_id = self._image_index[0]
        for ext in ('.jpg', '.jpeg', '.png'):
            candidate = os.path.join(self._data_path, 'JPEGImages', first_id + ext)
            if os.path.exists(candidate):
                return ext
        raise FileNotFoundError(
            'ไม่พบไฟล์ภาพสำหรับ patch id "{}" ด้วย extension ที่รองรับ (.jpg/.jpeg/.png) '
            'ใน {}'.format(first_id, os.path.join(self._data_path, 'JPEGImages'))
        )

    # ------------------------------------------------------------------
    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        return i

    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                   index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    # ------------------------------------------------------------------
    def _load_image_set_index(self):
        """
        อ่านรายชื่อ patch id จาก ImageSets/Main/{train,test}.txt
        """
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                       self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines() if x.strip()]
        return image_index

    # ------------------------------------------------------------------
    def gt_roidb(self):
        """
        โหลด ground-truth roidb จาก cache ถ้ามี ไม่งั้นสร้างใหม่จาก XML ทุกไฟล์
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_sar_annotation(index) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    # ------------------------------------------------------------------
    def _load_sar_annotation(self, index):
        """
        อ่าน bounding box จากไฟล์ VOC XML (เหมือน pascal_voc เดิม แต่ class เดียว)
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        all_objs = tree.findall('object')

        # export_voc_patches.py (Step 4) เขียน object 2 class ต่อ 1 อาคาร:
        # "building" (B_b, ground truth) และ "footprint" (B_f, ใช้ตอน generate label
        # เท่านั้น ไม่ใช่ target ของ detector). sar_building มีแค่ 1 class คือ
        # 'building' ดังนั้นต้องข้าม object อื่นที่ไม่รู้จัก ไม่ใช่ crash ด้วย KeyError
        objs = [obj for obj in all_objs
                if obj.find('name').text.lower().strip() in self._class_to_ind]

        # เอา duplicate box ออก (กล่องที่ xmin,ymin,xmax,ymax เหมือนกันเป๊ะ
        # ภายในไฟล์เดียวกัน) - พบ 23 ไฟล์จาก 10,180 ไฟล์ตอน verify -
        # เก็บแค่กล่องแรกที่เจอ ทิ้งตัวซ้ำ
        seen_boxes = set()
        unique_objs = []
        for obj in objs:
            bnd = obj.find('bndbox')
            key = tuple(bnd.find(t).text for t in ('xmin', 'ymin', 'xmax', 'ymax'))
            if key not in seen_boxes:
                seen_boxes.add(key)
                unique_objs.append(obj)
        objs = unique_objs

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # ไม่มี "-1" ตรงนี้เหมือน pascal_voc.py ต้นฉบับ เพราะพิกัดใน XML ของเรา
            # มาจาก rasterio/numpy ซึ่งเป็น 0-indexed อยู่แล้ว (ต่างจาก VOC ดั้งเดิม
            # ที่เป็น 1-indexed แบบ MATLAB) - ใส่ "-1" จะทำให้กล่องเลื่อนผิด 1 pixel
            # และกล่องที่ถูก clip ชิดขอบ patch (xmin=0) จะกลายเป็น -1 (ค่าติดลบ ผิดพลาด)
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            diffc = obj.find('difficult')
            difficult = 0 if diffc is None else int(diffc.text)
            ishards[ix] = difficult

            cls_name = obj.find('name').text.lower().strip()
            cls = self._class_to_ind[cls_name]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_ishard': ishards,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas,
        }

    # ------------------------------------------------------------------
    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        ประเมินผล single-box-per-image bounding box regression - ไม่ใช่ VOC mAP
        มาตรฐาน เพราะสถาปัตยกรรมนี้ (faster_rcnn_giou.py) ทำนายกล่องเดียวต่อภาพ
        ไม่มี multi-object detection/classification จริง และไม่มี confidence
        score จริง (ดู test.py: cls_scores = torch.tensor([[1]]) ค่าคงที่)
        all_boxes[1][i] (class index 1 = "building") จึงมีแค่ 1 แถวเสมอ

        เมตริกที่รายงาน:
          - Mean/median IoU ระหว่างกล่องทำนาย (B_pred) กับกล่องจริง (B_b,
            ground truth ที่ผ่านการปรับด้วย layover L=h*tan(theta) จาก Step 3)
          - สัดส่วนภาพที่ IoU >= 0.5 และ >= 0.7 (threshold มาตรฐานทั่วไป)
          - Mean Absolute Error ต่อพิกัด (xmin,ymin,xmax,ymax) หน่วย pixel -
            xmin คือมิติที่สัมพันธ์กับ layover extension โดยตรง จึงเป็นตัวชี้วัด
            ที่เกี่ยวข้องกับความแม่นยำการทำนายความสูงอาคารมากที่สุด
        """
        num_images = len(self.image_index)
        ious = []
        coord_errors = []

        for i in range(num_images):
            gt_boxes = self.roidb[i]['boxes']
            if gt_boxes.shape[0] == 0:
                continue  # ไม่ควรเกิดขึ้นจริง (ทุกภาพควรมี object อย่างน้อย 1 อัน)
            gt_box = gt_boxes[0].astype(np.float64)

            pred_dets = all_boxes[1][i]
            if pred_dets is None or len(pred_dets) == 0:
                continue
            pred_box = np.array(pred_dets[0][:4], dtype=np.float64)

            ious.append(self._compute_iou(pred_box, gt_box))
            coord_errors.append(np.abs(pred_box - gt_box))

        ious = np.array(ious)
        coord_errors = np.array(coord_errors)

        lines = []
        lines.append(f"จำนวนภาพที่ประเมิน: {len(ious)} / {num_images}")
        lines.append(f"Mean IoU: {ious.mean():.4f}")
        lines.append(f"Median IoU: {np.median(ious):.4f}")
        lines.append(f"IoU >= 0.5: {(ious >= 0.5).mean() * 100:.1f}%")
        lines.append(f"IoU >= 0.7: {(ious >= 0.7).mean() * 100:.1f}%")
        lines.append("")
        lines.append("Mean Absolute Error ต่อพิกัด (pixel):")
        for idx, name in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):
            lines.append(f"  {name}: {coord_errors[:, idx].mean():.2f}")
        lines.append("")
        lines.append(
            "หมายเหตุ: xmin สัมพันธ์กับ layover extension (L = h*tan(theta)) "
            "โดยตรง - MAE ของ xmin จึงเกี่ยวข้องกับความแม่นยำการทำนายความสูง "
            "อาคารมากที่สุดในบรรดา 4 พิกัด"
        )

        results_str = "\n".join(lines)
        print(results_str)

        # --- ประเมินความแม่นยำการทำนายความสูงอาคาร (h = L * pixel_spacing / tan(theta)) ---
        # นี่คือ metric จริงที่ paper สนใจ (ไม่ใช่แค่ IoU ของกล่อง) - แปลง L
        # (layover shift ในหน่วย pixel) กลับเป็นความสูงจริง (เมตร) โดย:
        #   1. หา B_f (footprint box) ด้วยวิธีเดียวกับที่โมเดลใช้ตอน inference
        #      จริง (extract_bboxes บน mask channel) เพื่อให้พิกัดสอดคล้องกับ
        #      ระบบเดียวกับ gt_box/pred_box ที่เป็น crop-relative อยู่แล้ว
        #   2. L_true = B_f.xmin - gt_box.xmin, L_pred = B_f.xmin - pred_box.xmin
        #   3. h = L * PIXEL_SPACING_M / cos(THETA_DEG)
        #
        # ข้อควรระวัง: THETA_DEG และ PIXEL_SPACING_M เป็นค่าคงที่ตัวแทนทั้ง scene
        # (ไม่ใช่ค่าจริงต่ออาคาร) - มุมตกกระทบจริงในข้อมูลเรามีช่วง 41.0-41.6
        # องศา (จาก GEOREF.xml) ค่า default 41.3 คือค่ากลางโดยประมาณ ถ้าต้องการ
        # ความแม่นยำสูงกว่านี้ ควรผูก incidence angle ต่ออาคารจริงจาก
        # footprints_pixel.pkl/GEOREF.xml แทน
        PIXEL_SPACING_M = 0.5   # เมตร/พิกเซล - ต้องตรงกับ GEC image จริง
        THETA_DEG = 41.3        # องศา - ค่ากลางโดยประมาณของ scene (ช่วงจริง 41.0-41.6)

        height_errors = []
        h_true_list, h_pred_list = [], []
        theta_rad = np.deg2rad(THETA_DEG)

        diagnostics = []  # (image_id, h_true, h_pred, abs_error, mask_degenerate)

        for i in range(num_images):
            gt_boxes = self.roidb[i]['boxes']
            if gt_boxes.shape[0] == 0:
                continue
            gt_box = gt_boxes[0].astype(np.float64)

            pred_dets = all_boxes[1][i]
            if pred_dets is None or len(pred_dets) == 0:
                continue
            pred_box = np.array(pred_dets[0][:4], dtype=np.float64)

            im = np.array(Image.open(self.image_path_at(i)))
            mask = im[:, :, 1:2]  # channel เขียว = footprint mask, เก็บมิติ (H,W,1)

            # เช็ค degenerate mask 2 แบบ:
            #   1. ptp==0 (ค่าเดียวกันหมด) -> ทำให้ forward() ของโมเดล
            #      divide-by-zero เป็น NaN ตอน normalize
            #   2. mask เล็กเกินไป (พื้นที่ < MIN_MASK_PIXELS) -> extract_bboxes
            #      ได้กล่องเกือบเป็นจุดเดียว (width/height ~0) ทำให้ candidate
            #      RoI ผิดเพี้ยนมาก แม้ ptp จะไม่เป็น 0 ก็ตาม (พบจริงในอาคาร
            #      00062639: mask มีแค่ 1 pixel เขียว จาก 572,286 pixel รวม -
            #      ทำให้ทำนายความสูงผิดไป 250+ เมตร)
            MIN_MASK_PIXELS = 20
            mask_area = int((mask[:, :, 0] > 0).sum())
            mask_degenerate = bool(np.ptp(mask) == 0) or (mask_area < MIN_MASK_PIXELS)

            horiz = np.where(np.any(mask[:, :, 0], axis=0))[0]
            if horiz.shape[0] == 0:
                continue  # mask ว่างเปล่า (ผิดปกติ) - ข้ามภาพนี้
            bf_xmin = float(horiz[0])

            L_true_px = bf_xmin - gt_box[0]
            L_pred_px = bf_xmin - pred_box[0]

            h_true = (L_true_px * PIXEL_SPACING_M) / np.tan(theta_rad)
            h_pred = (L_pred_px * PIXEL_SPACING_M) / np.tan(theta_rad)

            h_true_list.append(h_true)
            h_pred_list.append(h_pred)
            height_errors.append(h_pred - h_true)
            diagnostics.append((self.image_index[i], h_true, h_pred,
                                 abs(h_pred - h_true), mask_degenerate))

        h_true_arr = np.array(h_true_list)
        h_pred_arr = np.array(h_pred_list)
        height_errors = np.array(height_errors)

        height_lines = []
        height_lines.append("")
        height_lines.append("=" * 60)
        height_lines.append(f"ความแม่นยำการทำนายความสูงอาคาร (สมมติ theta={THETA_DEG} องศาทั้ง scene, "
                             f"pixel spacing={PIXEL_SPACING_M} m/px)")
        height_lines.append("=" * 60)
        height_lines.append(f"จำนวนอาคารที่ประเมินได้: {len(height_errors)} / {num_images}")
        if len(height_errors):
            height_lines.append(f"Mean height (ground truth): {h_true_arr.mean():.2f} m")
            height_lines.append(f"MAE ความสูง: {np.abs(height_errors).mean():.3f} m")
            height_lines.append(f"RMSE ความสูง: {np.sqrt((height_errors**2).mean()):.3f} m")
            # % error เทียบกับความสูงจริง (กัน divide-by-zero ถ้า h_true ใกล้ 0)
            valid = np.abs(h_true_arr) > 0.5
            if valid.sum():
                pct_err = np.abs(height_errors[valid]) / np.abs(h_true_arr[valid]) * 100
                height_lines.append(f"Mean % error (เทียบ ground truth height): {pct_err.mean():.1f}%")
            corr = np.corrcoef(h_true_arr, h_pred_arr)[0, 1] if len(h_true_arr) > 1 else float('nan')
            height_lines.append(f"Correlation (h_true vs h_pred): {corr:.4f}")

            # --- แยกความคลาดเคลื่อนตามช่วงความสูง ---
            # เช็คสมมติฐาน: error กระจุกตัวอยู่ที่อาคารเตี้ย (L เป็นพิกเซลน้อย
            # noise ~2px จึงกลายเป็นสัดส่วน error ที่สูงมาก) หรือกระจายทั่วทุก
            # ช่วงความสูง (จะบ่งชี้ปัญหาอื่นที่ไม่ใช่แค่ noise amplification)
            height_lines.append("")
            height_lines.append("แยกตามช่วงความสูง (ทดสอบสมมติฐาน noise amplification ในอาคารเตี้ย):")
            bucket_edges = [0, 5, 10, 20, 40, np.inf]
            bucket_labels = ["<5m", "5-10m", "10-20m", "20-40m", ">40m"]
            header = f"  {'ช่วง':<10}{'จำนวน':>8}{'MAE(m)':>10}{'RMSE(m)':>10}{'%error':>10}"
            height_lines.append(header)
            for lo, hi, label in zip(bucket_edges[:-1], bucket_edges[1:], bucket_labels):
                bucket_mask = (h_true_arr >= lo) & (h_true_arr < hi)
                n_bucket = bucket_mask.sum()
                if n_bucket == 0:
                    height_lines.append(f"  {label:<10}{0:>8}{'--':>10}{'--':>10}{'--':>10}")
                    continue
                bucket_err = height_errors[bucket_mask]
                bucket_h_true = h_true_arr[bucket_mask]
                mae_b = np.abs(bucket_err).mean()
                rmse_b = np.sqrt((bucket_err ** 2).mean())
                valid_b = np.abs(bucket_h_true) > 0.5
                pct_b = (np.abs(bucket_err[valid_b]) / np.abs(bucket_h_true[valid_b]) * 100).mean() \
                    if valid_b.sum() else float('nan')
                height_lines.append(
                    f"  {label:<10}{n_bucket:>8}{mae_b:>10.3f}{rmse_b:>10.3f}{pct_b:>9.1f}%"
                )

            # --- Top-10 outlier ที่แย่ที่สุด + เช็ค degenerate mask ---
            # ช่วยหาสาเหตุที่ RMSE ในบาง bucket สูงกว่า MAE มาก (บ่งชี้ outlier
            # กระจุกตัว ไม่ใช่ error กระจายสม่ำเสมอ) - degenerate mask (ptp==0)
            # คือสาเหตุที่เป็นไปได้ ทำให้ forward() ของโมเดล normalize แล้วได้ NaN
            # (ดู RuntimeWarning: invalid value encountered in divide ตอนรัน)
            n_degenerate = sum(1 for d in diagnostics if d[4])
            height_lines.append("")
            height_lines.append(f"จำนวนภาพที่ mask channel degenerate (ptp=0, เสี่ยง NaN ใน "
                                 f"forward()): {n_degenerate} / {len(diagnostics)}")
            height_lines.append("")
            height_lines.append("10 อันดับ error สูงสุด (image_id, h_true, h_pred, abs_error, mask_degenerate):")
            worst = sorted(diagnostics, key=lambda d: d[3], reverse=True)[:10]
            for img_id, h_true_i, h_pred_i, err_i, degen in worst:
                flag = "  <-- mask degenerate!" if degen else ""
                height_lines.append(
                    f"  {img_id}: h_true={h_true_i:.2f}m, h_pred={h_pred_i:.2f}m, "
                    f"error={err_i:.2f}m{flag}"
                )

        height_lines.append("")
        height_lines.append(
            "หมายเหตุ: THETA_DEG เป็นค่าคงที่โดยประมาณ ไม่ใช่มุมตกกระทบจริงต่อ"
            "อาคาร (ช่วงจริงในข้อมูล 41.0-41.6 องศา) - ถ้าต้องการความแม่นยำสูงขึ้น "
            "ควรผูก incidence angle ต่ออาคารจริงแทนค่าคงที่นี้"
        )

        height_str = "\n".join(height_lines)
        print(height_str)
        results_str = results_str + "\n" + height_str

        if output_dir is not None:
            results_path = os.path.join(output_dir, 'evaluation_results.txt')
            with open(results_path, 'w', encoding='utf-8') as f:
                f.write(results_str)
            print(f"\nบันทึกผลลัพธ์ที่ {results_path}")

        return {
            'mean_iou': float(ious.mean()) if len(ious) else 0.0,
            'median_iou': float(np.median(ious)) if len(ious) else 0.0,
            'recall_at_50': float((ious >= 0.5).mean()) if len(ious) else 0.0,
            'recall_at_70': float((ious >= 0.7).mean()) if len(ious) else 0.0,
            'mae_per_coord': coord_errors.mean(axis=0).tolist() if len(coord_errors) else [0, 0, 0, 0],
        }

    @staticmethod
    def _compute_iou(box_a, box_b):
        """IoU ระหว่างกล่อง 2 กล่อง แบบ (xmin,ymin,xmax,ymax)"""
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b

        inter_xmin = max(xa1, xb1)
        inter_ymin = max(ya1, yb1)
        inter_xmax = min(xa2, xb2)
        inter_ymax = min(ya2, yb2)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area
