# lib/model/_C.py
#
# Stub แทนที่ compiled C++/CUDA extension "model._C" ที่ปกติต้อง compile จาก
# lib/model/csrc/ (nms.cu, ROIAlign_cuda.cu, ROIPool_cuda.cu) ด้วย lib/setup.py
#
# ทำไมถึงใช้ stub แทนการ compile จริง:
#   1. ya0-sun/bboxRegNet4BldHeight (repo ของ paper) ไม่ได้แนบ lib/setup.py
#      หรือ lib/model/csrc/ มาด้วยเลย - source code สำหรับ compile ไม่มีอยู่จริง
#      ในโค้ดที่เผยแพร่ ต้องไปหยิบจาก jwyang/faster-rcnn.pytorch (ต้นทาง) เอง
#   2. source นั้นเป็นโค้ดยุค PyTorch 1.0 (~2018) ใช้ API เก่าที่ถูกลบ/เปลี่ยนชื่อ
#      ไปแล้วใน PyTorch 2.5.1 (เช่น AT_CHECK -> TORCH_CHECK) การ compile จริง
#      บน Windows + PyTorch สมัยใหม่มีความเสี่ยงสูงที่จะพังหลายจุด
#   3. ที่สำคัญที่สุด: ตรวจสอบโค้ดจริงแล้วยืนยันว่า forward() ของ
#      faster_rcnn_giou.py ไม่เคยเรียกใช้ RPN, RoIAlign, หรือ RoIPool เลย -
#      สถาปัตยกรรมจริงของโมเดลนี้คำนวณ RoI จาก footprint mask ตรงๆ
#      (extract_bboxes) ไม่ได้ใช้ RPN proposal + RoI pooling แบบ Faster R-CNN
#      มาตรฐาน ฟังก์ชันทั้ง 5 ตัวด้านล่างนี้จึงไม่เคยถูกเรียกจริงระหว่างเทรน
#
# ถ้าในอนาคตมีการแก้โมเดลให้กลับมาใช้ RPN/RoIAlign/RoIPool จริง (เช่น ปรับ
# สถาปัตยกรรมให้รองรับหลายอาคารต่อภาพแบบ multi-object detection) จะต้อง
# compile extension จริงจาก jwyang/faster-rcnn.pytorch (pytorch-1.0 branch)
# lib/model/csrc/ ก่อน - stub นี้จะโยน RuntimeError ทันทีถ้ามีอะไรเรียกใช้จริง
# แทนที่จะคำนวณผลลัพธ์ผิดๆ แบบเงียบๆ

_STUB_MSG = (
    "model._C.{name}() ถูกเรียกใช้จริง แต่นี่เป็นแค่ stub (ไม่มี compiled "
    "extension จริง) - ฟังก์ชันนี้ไม่ควรถูกเรียกในสถาปัตยกรรมปัจจุบันของ "
    "faster_rcnn_giou.py (ที่ไม่ใช้ RPN/RoIAlign/RoIPool จริง) ถ้าเห็น error "
    "นี้แปลว่ามีการแก้โมเดลไปใช้ RPN/RoIAlign/RoIPool จริงแล้ว ต้อง compile "
    "extension จริงจาก jwyang/faster-rcnn.pytorch (pytorch-1.0 branch) "
    "lib/model/csrc/ ก่อนใช้งาน"
)


def nms(dets, scores, threshold):
    raise RuntimeError(_STUB_MSG.format(name="nms"))


def roi_align_forward(input, roi, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    raise RuntimeError(_STUB_MSG.format(name="roi_align_forward"))


def roi_align_backward(grad, rois, spatial_scale, pooled_height, pooled_width,
                        batch_size, channels, height, width, sampling_ratio):
    raise RuntimeError(_STUB_MSG.format(name="roi_align_backward"))


def roi_pool_forward(input, roi, spatial_scale, pooled_height, pooled_width):
    raise RuntimeError(_STUB_MSG.format(name="roi_pool_forward"))


def roi_pool_backward(grad, input, rois, argmax, spatial_scale, pooled_height,
                       pooled_width, batch_size, channels, height, width):
    raise RuntimeError(_STUB_MSG.format(name="roi_pool_backward"))
