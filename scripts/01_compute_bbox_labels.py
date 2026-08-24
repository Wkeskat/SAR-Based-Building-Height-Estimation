"""
Step 2b: คำนวณ Near-Range Vector + สร้าง Building Bounding Box (B_b) = LABEL
==============================================================================

Input:
  - heading_deg     : จาก 02a_compute_heading.py
  - look_side       : "RIGHT" หรือ "LEFT" (จาก metadata)
  - footprint (UTM) : shapefile/GeoJSON ที่มี attribute height
  - incidence angle raster (GeoTIFF) : จาก Phase 1

Output:
  - footprint polygon เดิม (B_f)
  - building polygon ที่เลื่อนแล้ว (B_b) = LABEL สำหรับเทรนโมเดล
"""

import numpy as np
import geopandas as gpd
import rasterio
from shapely.affinity import translate
from shapely.ops import unary_union


# ----------------------------------------------------------------------
# 1) ฟังก์ชันคำนวณ near-range unit vector จาก heading + look side ----- หาทิศทางของ layover
# ----------------------------------------------------------------------
def near_range_unit_vector(heading_deg: float, look_side: str) -> tuple[float, float]:
    """
    คำนวณ unit vector (dE, dN) ในพิกัดแผนที่ (East, North) ที่ชี้ไปทาง
    "near range" คือทิศทางที่ยอดอาคาร (ผลจาก layover) จะถูกขยับไป

    Parameters
    ----------
    heading_deg : ทิศทางการเคลื่อนที่ของดาวเทียม (0-360, ตามเข็มนาฬิกาจากเหนือ)
                  ได้จาก compute_heading_from_georef() ใน 02a
    look_side   : "RIGHT" หรือ "LEFT" (จาก metadata: antenna_pointing)

    Returns
    -------
    (dE, dN) : unit vector ในพิกัด (East, North) หน่วยเมตร/เมตร (length=1)

    หลักการ:
      - Right-looking: near_range_azimuth = heading - 90 deg
      - Left-looking : near_range_azimuth = heading + 90 deg
      (เพราะทิศ near-range = ทิศจากพื้นดินไปหาดาวเทียม = ตรงข้ามกับทิศ
       illumination ที่เรดาร์ยิงออกไปทางขวา/ซ้ายของทิศบิน)
    """
    look_side = look_side.strip().upper()
    if look_side == "RIGHT":
        near_range_azimuth = (heading_deg - 90.0) % 360.0
    elif look_side == "LEFT":
        near_range_azimuth = (heading_deg + 90.0) % 360.0
    else:
        raise ValueError(f"look_side ต้องเป็น 'RIGHT' หรือ 'LEFT', ได้ '{look_side}'")

    az_rad = np.deg2rad(near_range_azimuth)
    dE = np.sin(az_rad)   # ทิศตะวันออก
    dN = np.cos(az_rad)   # ทิศเหนือ

    return dE, dN, near_range_azimuth


# ----------------------------------------------------------------------
# 2) ฟังก์ชันอ่านค่า incidence angle จาก raster ที่ตำแหน่งใดๆ (bilinear)
# ----------------------------------------------------------------------
def sample_incidence_angle(raster_path: str, x_coords, y_coords):
    """
    อ่านค่า incidence angle จาก raster ที่ตำแหน่ง (x, y) ในหน่วย UTM
    ใช้ rasterio .sample() ซึ่ง default เป็น nearest-neighbor
    (ถ้าต้องการ bilinear จริง ให้ใช้ scipy.ndimage.map_coordinates แทน
     ตามตัวอย่างที่ comment ไว้ด้านล่าง)
    """
    with rasterio.open(raster_path) as src:
        coords = list(zip(x_coords, y_coords))
        values = [v[0] for v in src.sample(coords)]
    return np.array(values, dtype=float)


def sample_incidence_angle_bilinear(raster_path: str, x_coords, y_coords):
    """เวอร์ชัน bilinear interpolation จริง (แม่นยำกว่า nearest-neighbor)"""
    from scipy.ndimage import map_coordinates

    with rasterio.open(raster_path) as src:
        band = src.read(1).astype(float)
        transform = src.transform
        inv_transform = ~transform

        rows, cols = [], []
        for x, y in zip(x_coords, y_coords):
            col, row = inv_transform * (x, y)
            rows.append(row)
            cols.append(col)

        values = map_coordinates(band, [rows, cols], order=1, mode="nearest")
    return values


# ----------------------------------------------------------------------
# 3) ฟังก์ชันหลัก: สร้าง B_b (building bbox label) จาก B_f + height + near-range vector
# ----------------------------------------------------------------------
def compute_building_bbox(footprint_geom, height_m, theta_deg, dE, dN):
    """
    เลื่อน footprint polygon ไปตาม near-range vector เป็นระยะ L = h * tan(theta)
    แล้วคืนค่า:
      - shifted_geom  : footprint ที่เลื่อนแล้ว (ตำแหน่ง "ยอดอาคาร" โดยประมาณ)
      - bbox_union    : bounding box ของ (footprint เดิม รวม footprint ที่เลื่อน) = B_b

    สูตร: L = h * tan(theta)   <-- ใช้กับภาพ ground-range/GEC (ตามที่คุยกันไปก่อนหน้า)
    """
    theta_rad = np.deg2rad(theta_deg)
    L = height_m * np.tan(theta_rad)   # ระยะ layover หน่วยเมตร

    dx = L * dE   # ระยะเลื่อนในแกน East (เมตร)
    dy = L * dN   # ระยะเลื่อนในแกน North (เมตร)

    shifted_geom = translate(footprint_geom, xoff=dx, yoff=dy)

    # B_b = bounding box ของ union(footprint เดิม, footprint ที่เลื่อนแล้ว)
    bbox_union = unary_union([footprint_geom, shifted_geom]).envelope

    return shifted_geom, bbox_union, L


# ----------------------------------------------------------------------
# 4) ตัวอย่างการใช้งานรวม (main)
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # ---- ค่าที่ต้องกรอกตามข้อมูลจริง ----
    HEADING_DEG = 349.733          # จาก 02a_compute_heading.py
    LOOK_SIDE = "RIGHT"            # จาก metadata (antenna_pointing)
    HEIGHT_FIELD = "BL_HEIGHT"            #ชื่อ column ความสูงใน footprint file
    footprint_path =r"E:\00_3D\14_SAR_\00_DATA_\KKC_ SAR_spot_055.geojson"
    incidence_raster_path = r"E:\00_3D\14_SAR_\00_DATA_\incidence_angle_TDX1.tif"

    # ---- 1) คำนวณ near-range vector ----
    dE, dN, near_range_az = near_range_unit_vector(HEADING_DEG, LOOK_SIDE)
    print(f"Near-range azimuth: {near_range_az:.3f} deg")
    print(f"Unit vector (dE, dN) = ({dE:.4f}, {dN:.4f})")

    # ---- 2) โหลด footprint แล้ว reproject ให้ตรงกับ CRS ของ incidence raster (UTM, เมตร) ----
    gdf = gpd.read_file(footprint_path)
    with rasterio.open(incidence_raster_path) as _ref:
        raster_crs = _ref.crs
    if gdf.crs != raster_crs:
        print(f"Reprojecting footprints: {gdf.crs} -> {raster_crs}")
        gdf = gdf.to_crs(raster_crs)

    # ---- 3) อ่าน incidence angle ที่ centroid ของแต่ละอาคาร ----
    centroids = gdf.geometry.centroid
    theta_values = sample_incidence_angle_bilinear(
        incidence_raster_path,
        centroids.x.values,
        centroids.y.values,
    )
    gdf["incidence_angle"] = theta_values

    # ---- 4) คำนวณ B_b (label) ให้ทุกอาคาร ----
    results = []
    for idx, row in gdf.iterrows():
        shifted_geom, bbox_b, L = compute_building_bbox(
            row.geometry, row[HEIGHT_FIELD], row["incidence_angle"], dE, dN
        )
        results.append({
            "building_id": idx,
            "height": row[HEIGHT_FIELD],
            "incidence_angle": row["incidence_angle"],
            "layover_length_m": L,
            "footprint_geom": row.geometry,   # B_f (geometry เดิม)
            "building_bbox_geom": bbox_b,      # B_b (LABEL ที่จะใช้เทรน)
        })

    print(f"\nคำนวณ B_b สำเร็จ {len(results)} อาคาร")
    print(f"ตัวอย่าง layover length เฉลี่ย: {np.mean([r['layover_length_m'] for r in results]):.2f} m")

    # บันทึกผลลัพธ์เป็น GeoDataFrame ใหม่ (B_b) ไว้ใช้ในขั้นตอน crop patch / export VOC ถัดไป
    out_gdf = gpd.GeoDataFrame(
        [{"building_id": r["building_id"], "height": r["height"],
          "incidence_angle": r["incidence_angle"], "layover_m": r["layover_length_m"]}
         for r in results],
        geometry=[r["building_bbox_geom"] for r in results],
        crs=gdf.crs,
    )
    out_gdf.to_file(r"E:\00_3D\14_SAR_\00_DATA_\building_bbox_labels.geojson", driver="GeoJSON")
    print("บันทึก building_bbox_labels.geojson แล้ว")
