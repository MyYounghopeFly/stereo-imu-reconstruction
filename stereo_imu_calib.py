#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stereo_imu_calib.py
双目+IMU 标定参数共享模块
所有数值来源于 Kalibr 标定结果，单位统一为 mm（视觉侧）和 m（VIO/IMU侧）
"""
import numpy as np
import cv2

# ===================== A. 图像分辨率 =====================
IMG_WIDTH = 2448
IMG_HEIGHT = 2048
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# ===================== B. 相机内参与畸变 =====================
# Cam0 (左目)
M1 = np.array([
    [1809.864805721, 0.0,            1238.492343789],
    [0.0,            1809.917365722, 1015.862822050],
    [0.0,            0.0,            1.0           ]
])
d1 = np.array([-0.063040578, 0.067970468, 0.000266825, 0.000515366])

# Cam1 (右目)
M2 = np.array([
    [1807.598929005, 0.0,            1232.514926074],
    [0.0,            1808.031005923, 995.642324168 ],
    [0.0,            0.0,            1.0           ]
])
d2 = np.array([-0.068156536, 0.079784099, -0.000231534, -0.000193266])

# ===================== C. 双目外参 T_c1_c0 (Cam0 → Cam1) =====================
T_cam0_to_cam1_meters = np.array([
    [ 0.99498668,  0.02378106,  0.09713887, -0.09683436],
    [-0.02568884,  0.99949997,  0.01843625,  0.00250419],
    [-0.09665186, -0.02083921,  0.99510007,  0.00810463],
    [ 0.0,         0.0,         0.0,         1.0       ]
])

# mm 版本（视觉重建使用）
T_cam0_to_cam1_mm = T_cam0_to_cam1_meters.copy()
T_cam0_to_cam1_mm[:3, 3] *= 1000.0

# ===================== D. 视惯外参 T_ic (Cam0 → IMU) =====================
T_cam0_to_imu_meters = np.array([
    [-0.9997124,  -0.02384959,  0.00251334,  0.01109784],
    [-0.00156136, -0.03985071, -0.99920442, -0.00751317],
    [ 0.02393077, -0.99892098,  0.03980201,  0.00151265],
    [ 0.0,         0.0,         0.0,         1.0       ]
])

# mm 版本
T_cam0_to_imu_mm = T_cam0_to_imu_meters.copy()
T_cam0_to_imu_mm[:3, 3] *= 1000.0

# IMU → Cam0（VIO 输出的 T_w_imu 需要转到相机坐标系）
T_imu_to_cam0_meters = np.linalg.inv(T_cam0_to_imu_meters)
T_imu_to_cam0_mm = np.linalg.inv(T_cam0_to_imu_mm)

# ===================== E. SGM / SGBM 参数 =====================
SGM_BLOCK_SIZE = 5
SGM_MULTIPLIER = 1
SGM_UNIQUENESS = 25
SGM_MAX_DIFF = 1

# 手动视差范围（根据实际工作距离调整）
MANUAL_MIN_DISP = 486
MANUAL_NUM_DISP = 128

ACTUAL_P1 = 8 * SGM_MULTIPLIER * SGM_BLOCK_SIZE ** 2
ACTUAL_P2 = 32 * SGM_MULTIPLIER * SGM_BLOCK_SIZE ** 2

# ===================== F. 点云后处理参数 =====================
MIN_Z_DIST = 100.0      # mm
MAX_Z_DIST = 500.0      # mm
VOXEL_SIZE = 0.5        # mm

# ===================== G. 极线校正 LUT 生成 =====================
_rect_maps_cache = {}

def get_rectification_maps(size=None, scale=None):
    """
    生成并缓存极线校正 LUT。
    - size: 直接指定图像尺寸 (w, h)
    - scale: 按比例缩放全分辨率内参 (如 0.5)，size 自动计算
    二者至少给一个；都不给则用全分辨率。
    """
    if size is None:
        if scale is None:
            scale = 1.0
        size = (int(round(IMG_WIDTH * scale)), int(round(IMG_HEIGHT * scale)))
    else:
        scale = float(size[0]) / IMG_WIDTH   # py2: 必须 float, 否则整数除法得 0

    if size in _rect_maps_cache:
        return _rect_maps_cache[size]

    K1 = M1.copy(); K1[0, 0] *= scale; K1[1, 1] *= scale; K1[0, 2] *= scale; K1[1, 2] *= scale
    K2 = M2.copy(); K2[0, 0] *= scale; K2[1, 1] *= scale; K2[0, 2] *= scale; K2[1, 2] *= scale

    R_mat = T_cam0_to_cam1_mm[:3, :3]
    t_vec = T_cam0_to_cam1_mm[:3, 3].reshape(3, 1)

    R1, R2, P1, P2, Q = cv2.stereoRectify(
        cameraMatrix1=K1, cameraMatrix2=K2,
        distCoeffs1=d1, distCoeffs2=d2,
        R=R_mat, T=t_vec,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1, imageSize=size, newImageSize=size
    )[0:5]

    map1x, map1y = cv2.initUndistortRectifyMap(K1, d1, R1, P1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, d2, R2, P2, size, cv2.CV_32FC1)

    maps = {
        'size': size,
        'scale': scale,
        'map1x': map1x, 'map1y': map1y,
        'map2x': map2x, 'map2y': map2y,
        'Q': Q, 'R1': R1, 'P1': P1, 'P2': P2,
        # 有效 f'*|B| (像素*mm): 由 reprojectImageTo3D 深度公式 Z = Q[2,3]/(Q[3,2]*d) 得
        # 视差 d = fB / Z_mm。(本设备 Tx<0, Q[3,2]>0, 视差 d>0, fB 为正)
        'fB': Q[2, 3] / Q[3, 2],
    }
    _rect_maps_cache[size] = maps
    return maps


def compute_sgm_disparity(rectL, rectR,
                          min_disp=MANUAL_MIN_DISP,
                          num_disp=MANUAL_NUM_DISP,
                          block_size=SGM_BLOCK_SIZE,
                          multiplier=SGM_MULTIPLIER,
                          max_diff=SGM_MAX_DIFF,
                          uniqueness=SGM_UNIQUENESS):
    """计算 SGBM 视差图。"""
    P1 = 8 * multiplier * block_size ** 2
    P2 = 32 * multiplier * block_size ** 2

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1, P2=P2,
        disp12MaxDiff=max_diff,
        uniquenessRatio=uniqueness,
        speckleWindowSize=1000,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_HH
    )

    disparity = left_matcher.compute(rectL, rectR)
    return disparity.astype(np.float32) / 16.0


def disparity_to_pointcloud(disparity, gray_img, Q,
                            min_z=MIN_Z_DIST,
                            max_z=MAX_Z_DIST):
    """从视差图生成局部点云 (Open3D)。"""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D is required for point cloud generation.")

    grad_x = cv2.Sobel(disparity, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(disparity, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    EDGE_GRADIENT_THRESHOLD = 6.0

    points_3d = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True).reshape(-1, 3)
    disp_flat = disparity.reshape(-1)
    gray_flat = gray_img.reshape(-1)
    grad_flat = grad_mag.reshape(-1)

    mask = (
        np.isfinite(points_3d[:, 0]) &
        (disp_flat > 0) &
        (points_3d[:, 2] > min_z) &
        (points_3d[:, 2] < max_z) &
        (grad_flat < EDGE_GRADIENT_THRESHOLD) &
        (gray_flat > 3) &
        (gray_flat < 255)
    )

    valid_points = points_3d[mask]
    if len(valid_points) == 0:
        return o3d.geometry.PointCloud()

    valid_gray = gray_flat[mask].astype(np.float64) / 255.0
    colors_rgb = np.stack((valid_gray, valid_gray, valid_gray), axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
    return pcd


def transform_pcd_to_world(pcd, T_w_imu_meters):
    """
    将局部点云（Cam0 校正坐标系）变换到世界坐标系。
    输入 T_w_imu_meters 为 VIO 输出的 4x4 齐次矩阵（单位：米）。
    """
    T_wi = np.eye(4)
    T_wi[:3, :3] = T_w_imu_meters[:3, :3]
    T_wi[:3, 3] = T_w_imu_meters[:3, 3] * 1000.0  # m -> mm

    # T_rect0_to_raw0: 校正图像坐标系 -> 原始相机坐标系
    rect_maps = get_rectification_maps()
    R1 = rect_maps['R1']
    T_rect0_to_raw0 = np.eye(4)
    T_rect0_to_raw0[:3, :3] = R1.T

    T_wi_ic = np.dot(T_wi, T_cam0_to_imu_mm)
    T_final = np.dot(T_wi_ic, T_rect0_to_raw0)

    pcd.transform(T_final)
    return pcd


if __name__ == "__main__":
    # 自检：验证所有矩阵的可逆性和基本一致性
    print("[CALIB CHECK] 标定参数自检...")
    assert np.linalg.det(T_cam0_to_cam1_meters[:3, :3]) > 0.99, "旋转矩阵行列式异常"
    assert np.linalg.det(T_cam0_to_imu_meters[:3, :3]) > 0.99, "旋转矩阵行列式异常"

    maps = get_rectification_maps()
    print("[CALIB CHECK] 极线校正 LUT 生成成功，尺寸: {}".format(maps['size']))
    print("[CALIB CHECK] Q 矩阵:\n{}".format(maps['Q']))
    print("[CALIB CHECK] Baseline (mm): {:.2f}".format(np.linalg.norm(T_cam0_to_cam1_mm[:3, 3])))
    print("[CALIB CHECK] 全部通过！")
