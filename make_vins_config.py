#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_vins_config.py
根据 stereo_imu_calib.py 中的标定参数，生成 VINS-Fusion 所需的配置文件。

生成 3 个文件到 --outdir (默认 ./vins_config/):
  handheld_cam0.yaml                  左目相机参数 (OpenCV FileStorage 格式)
  handheld_cam1.yaml                  右目相机参数
  handheld_stereo_imu_config.yaml     VINS-Fusion 主配置 (双目+IMU)

用法:
  python make_vins_config.py --scale 0.5 --outdir ./vins_config

  --scale 必须与 ros_publisher_stereo_imu.py 的 ~scale 参数一致！
  (发布节点把图像缩小到 scale 倍，内参线性缩放，畸变系数不变)

把生成的 handheld_stereo_imu_config.yaml 传给 VINS-Fusion:
  rosrun vins vins_node <outdir>/handheld_stereo_imu_config.yaml
"""

import os
import argparse
import numpy as np

import stereo_imu_calib as calib


def cam_yaml(intrinsics, distortion, width, height):
    fx, fy, cx, cy = intrinsics
    k1, k2, p1, p2 = distortion
    return f"""%YAML:1.0
---
model_type: PINHOLE
camera_name: camera
image_width: {width}
image_height: {height}
distortion_parameters:
   k1: {k1:.9e}
   k2: {k2:.9e}
   p1: {p1:.9e}
   p2: {p2:.9e}
projection_parameters:
   fx: {fx:.6f}
   fy: {fy:.6f}
   cx: {cx:.6f}
   cy: {cy:.6f}
"""


def opencv_mat44(T):
    flat = ", ".join(f"{v:.9e}" for v in np.asarray(T).flatten())
    return f"""!!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [ {flat} ]"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=float, default=0.5,
                        help='图像降采样比例，必须与发布节点 ~scale 一致')
    parser.add_argument('--outdir', type=str, default='./vins_config')
    parser.add_argument('--output_path', type=str, default='./vins_output/',
                        help='VINS 输出位姿/回环文件目录')
    args = parser.parse_args()

    s = args.scale
    W = int(round(calib.IMG_WIDTH * s))
    H = int(round(calib.IMG_HEIGHT * s))
    print(f"[VINS CFG] 工作分辨率: {W}x{H} (scale={s})")

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    # ---- 缩放后的内参 ----
    K0 = calib.M1.copy(); K0[0, 0] *= s; K0[1, 1] *= s; K0[0, 2] *= s; K0[1, 2] *= s
    K1 = calib.M2.copy(); K1[0, 0] *= s; K1[1, 1] *= s; K1[0, 2] *= s; K1[1, 2] *= s

    cam0_path = os.path.join(args.outdir, 'handheld_cam0.yaml')
    cam1_path = os.path.join(args.outdir, 'handheld_cam1.yaml')
    with open(cam0_path, 'w') as f:
        f.write(cam_yaml((K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]), calib.d1, W, H))
    with open(cam1_path, 'w') as f:
        f.write(cam_yaml((K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]), calib.d2, W, H))

    # ---- 外参: body_T_camN = T_imu_camN (camN 坐标系点 -> IMU 坐标系) ----
    # T_i_c0 直接来自标定; T_i_c1 = T_i_c0 @ inv(T_c1_c0)
    T_i_c0 = calib.T_cam0_to_imu_meters
    T_c1_c0 = calib.T_cam0_to_cam1_meters
    T_i_c1 = T_i_c0 @ np.linalg.inv(T_c1_c0)

    print(f"[VINS CFG] body_T_cam0 (T_imu_cam0):\n{T_i_c0}")
    print(f"[VINS CFG] body_T_cam1 (T_imu_cam1):\n{T_i_c1}")

    cam0_abs = os.path.basename(cam0_path)
    cam1_abs = os.path.basename(cam1_path)

    main_cfg = f"""%YAML:1.0

# ================= 话题 =================
imu_topic: "/imu0"
image0_topic: "/cam0/image_raw"
image1_topic: "/cam1/image_raw"
output_path: "{os.path.abspath(args.output_path)}"

# ================= 相机标定 =================
model_type: PINHOLE
# 注意: VINS-Fusion 会把 cam0_calib 拼接到配置文件所在目录后, 必须用相对文件名!
cam0_calib: "{os.path.basename(args.cam0_out)}"
cam1_calib: "{os.path.basename(args.cam1_out)}"
image_width: {W}
image_height: {H}

# ============ IMU 与相机外参 ============
# 0: 固定使用 body_T_cam 给定外参不优化; 1: 以给定外参为初值在线优化; 2: 无先验从零在线标定(忽略 body_T_cam)
estimate_extrinsic: 0
body_T_cam0: {opencv_mat44(T_i_c0)}
body_T_cam1: {opencv_mat44(T_i_c1)}

multiple_thread: 1

# ============ 特征跟踪前端 ============
max_cnt: 150
min_dist: 30
freq: 10
F_threshold: 1.0
show_track: 1
flow_back: 1

# ============ 优化参数 ============
max_solver_time: 0.04
max_num_iterations: 8
keyframe_parallax: 10.0

# ============ IMU 噪声 (HIPNUC 级 MEMS 典型值, 建议用 Allan 方差标定后替换) ============
acc_n: 0.1
gyr_n: 0.01
acc_w: 0.001
gyr_w: 1.0e-4
g_norm: 9.80665

# ============ 时间同步 ============
estimate_td: 1
td: 0.0

rolling_shutter: 0
rolling_shutter_tr: 0

# ============ 可视化 ============
save_image: 0
visualize_imu_forward: 0
visualize_camera_size: 0.4

# ============ 回环检测 ============
load_previous_pose_graph: 0
pose_graph_save_path: "{os.path.abspath(args.output_path)}/pose_graph.txt"
"""

    main_path = os.path.join(args.outdir, 'handheld_stereo_imu_config.yaml')
    with open(main_path, 'w') as f:
        f.write(main_cfg)

    print(f"\n[VINS CFG] 生成完成:")
    print(f"  主配置:   {main_path}")
    print(f"  左目标定: {cam0_path}")
    print(f"  右目标定: {cam1_path}")
    print(f"\n启动 VINS-Fusion:")
    print(f"  rosrun vins vins_node {os.path.abspath(main_path)}")


if __name__ == '__main__':
    main()
