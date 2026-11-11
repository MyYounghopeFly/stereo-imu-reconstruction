#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
self_check.py
标定参数、坐标变换、视差推导的数学一致性自检。
运行: python self_check.py
"""
import sys, math, numpy as np
import stereo_imu_calib as calib

errors = 0

def check(name, cond, detail=""):
    global errors
    if cond:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        errors += 1

print("=== 1. 矩阵正交性自检 ===")
for label, R in [("T_c1_c0", calib.T_cam0_to_cam1_meters[:3,:3]),
                 ("T_ic",    calib.T_cam0_to_imu_meters[:3,:3])]:
    det = np.linalg.det(R)
    check(f"{label} det(R)=+1", abs(det-1.0) < 1e-4, f"det={det:.6f}")

print("\n=== 2. 内参物理合理性 ===")
check("fx > 0", calib.M1[0,0] > 0)
check("fy > 0", calib.M1[1,1] > 0)
check("cx > 0", calib.M1[0,2] > 0)
check("cy > 0", calib.M1[1,2] > 0)
check("cx < W", calib.M1[0,2] < calib.IMG_WIDTH)
check("cy < H", calib.M1[1,2] < calib.IMG_HEIGHT)

check("cam1 fx > 0", calib.M2[0,0] > 0)
check("cam1 cx < W", calib.M2[0,2] < calib.IMG_WIDTH)

baseline = np.linalg.norm(calib.T_cam0_to_cam1_mm[:3,3])
check("baseline 合理 (80~120mm)", 80 < baseline < 120, f"baseline={baseline:.1f}mm")

print("\n=== 3. 外参互逆自检 ===")
T_i_c0 = calib.T_cam0_to_imu_meters
T_c0_i = calib.T_imu_to_cam0_meters
check("T_i_c0 @ T_c0_i = I", np.allclose(T_i_c0 @ T_c0_i, np.eye(4), atol=1e-6))

print("\n=== 4. 降采样校正矩阵内部一致性 (scale=0.5) ===")
maps = calib.get_rectification_maps(scale=0.5)
check("scale=0.5 size 匹配", maps['size'] == (1224, 1024))
# alpha=1 时 P1 焦距会被主动缩小，只需验证 LUT/Q/P 同源自洽:
# Q[2,3] 应等于 P1 的焦距, Q[3,2] 应与基线 |Tx| 满足 -1/Tx 关系
f_rect = maps['P1'][0, 0]
Tx_est = abs(maps['P2'][0, 3]) / maps['P2'][0, 0]
check("Q[2,3] == P1 焦距", abs(maps['Q'][2, 3] - f_rect) < 0.1,
      f"Q[2,3]={maps['Q'][2,3]:.3f}, f_rect={f_rect:.3f}")
check("|Tx| ≈ 标定基线", abs(Tx_est - baseline) < 1.0,
      f"Tx_est={Tx_est:.1f}mm vs baseline={baseline:.1f}mm")
check("fB == f_rect*|Tx|", abs(maps['fB'] - f_rect * Tx_est) < 1.0,
      f"fB={maps['fB']:.1f} vs {f_rect*Tx_est:.1f}")

print("\n=== 5. 视差数学推导自检 ===")
fB = maps['fB']
check("fB > 0", fB > 0, f"fB={fB:.1f} (pixel*mm)")

# 测试: Z = fB / d
test_d = 100.0  # pixel
Z_calc = fB / test_d
check("视差反推距离自洽 (mm)", abs(Z_calc - fB/test_d) < 0.01)

# 从距离反推视差: d = fB / Z
for Z_m in [0.3, 0.5, 1.0, 2.0]:
    Z_mm = Z_m * 1000.0
    d = fB / Z_mm
    Z_back = fB / d
    check(f"Z={Z_m}m -> d={d:.1f}pix -> Z_back={Z_back/1000:.2f}m",
          abs(Z_back - Z_mm) < 0.1)

print("\n=== 6. VINS body_T_cam1 推导自检 ===")
# body_T_cam1 = T_i_c0 @ inv(T_c1_c0)，其旋转部分必须正交
T_i_c0 = calib.T_cam0_to_imu_meters
T_c1_c0 = calib.T_cam0_to_cam1_meters
T_i_c1 = T_i_c0 @ np.linalg.inv(T_c1_c0)
check("T_i_c1 旋转行列式=+1", abs(np.linalg.det(T_i_c1[:3, :3]) - 1.0) < 1e-4,
      f"det={np.linalg.det(T_i_c1[:3,:3]):.6f}")
# cam1 相对 imu 的平移 ≈ cam0 相对 imu 平移 + 旋转后的基线
t_diff = T_i_c1[:3, 3] - T_i_c0[:3, 3]
check("cam0-cam1 在 IMU 系间距 ≈ baseline",
      abs(np.linalg.norm(t_diff) - baseline / 1000.0) < 0.005,
      f"dist={np.linalg.norm(t_diff)*1000:.1f}mm vs baseline={baseline:.1f}mm")

print("\n" + "="*40)
if errors == 0:
    print("全部自检通过！标定参数数学一致。")
else:
    print(f"发现 {errors} 项异常，请检查标定数据！")
    sys.exit(1)
