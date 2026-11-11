#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serial_offline_recon.py
离线增量式全局三维重建 (Windows 侧)。

启动时先弹出参数设置窗口（填空式），确认后才开始重建。
拼接逻辑: 遍历 VIO 轨迹帧 -> 按时间戳找最近图像帧 -> SGBM 深度 -> 世界系变换 -> 关键帧累积 -> 全局地图。

参数说明:
  --dataset  EuRoC 数据集目录
  --vio      VINS 轨迹 csv (默认数据集内)
  --scale    图像缩放 (默认 0.5)
"""
import os
import glob
import argparse
import time
import datetime

import cv2
import numpy as np
import pandas as pd

import stereo_imu_calib as calib

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("未安装 open3d: pip install open3d")

# ====================== 默认参数 =====================
DEFAULTS = {
    'min_z': 0.30,
    'max_z': 2.50,
    'sgbm_block': 7,
    'sgbm_uniqueness': 10,
    'max_pts_per_frame': 30000,
    'grad_threshold': 6.0,
    'kf_trans': 0.04,
    'kf_rot': 4.0,
    'map_voxel': 0.003,
    'map_max_points': 10000000,
    'interval': 1,
}
POSE_SYNC_TOL = 0.1   # 图像-位姿最大时间差 (s)


def load_vins_trajectory(csv_path):
    df = pd.read_csv(csv_path, header=None, usecols=range(8))
    df.columns = ['timestamp', 'p_x', 'p_y', 'p_z', 'q_w', 'q_x', 'q_y', 'q_z']
    if df['timestamp'].iloc[0] > 1e12:
        df['timestamp'] = df['timestamp'] / 1e9
    return df


def quat_to_matrix(qw, qx, qy, qz):
    x, y, z, w = qx, qy, qz, qw
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])
    return R


def voxel_downsample(points, colors, voxel):
    keys_3d = np.floor(points / voxel).astype(np.int64)
    keys = (keys_3d[:, 0] * 73856093) ^ (keys_3d[:, 1] * 19349663) ^ (keys_3d[:, 2] * 83492791)
    _, idx = np.unique(keys, return_index=True)
    return points[idx], colors[idx]


def stereo_to_points(imgL, imgR, maps, p):
    """根据当前参数计算 SGBM 点云，mode=HH 最高精度"""
    rectL = cv2.remap(imgL, maps['map1x'], maps['map1y'], cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, maps['map2x'], maps['map2y'], cv2.INTER_LINEAR)

    fB = maps['fB']
    min_disp = int(np.floor(fB / (p['max_z'] * 1000.0)))
    max_disp = int(np.ceil(fB / (p['min_z'] * 1000.0)))
    num_disp = max(int(np.ceil((max_disp - min_disp) / 16.0)) * 16, 16)

    block = p['sgbm_block']
    if block % 2 == 0:
        block += 1
    block = max(3, min(15, block))

    # STEREO_SGBM_MODE_HH: 全像素精度 + 双通道动态规划，最高精度但最慢
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=block,
        P1=8 * block ** 2, P2=32 * block ** 2,
        disp12MaxDiff=1, uniquenessRatio=p['sgbm_uniqueness'],
        speckleWindowSize=200, speckleRange=32, preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_HH)
    disp = matcher.compute(rectL, rectR).astype(np.float32) / 16.0

    gx = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    pts3d = cv2.reprojectImageTo3D(disp, maps['Q'], handleMissingValues=True)
    mask = (np.isfinite(pts3d[:, :, 2]) & (disp > min_disp) &
            (pts3d[:, :, 2] > p['min_z'] * 1000.0) & (pts3d[:, :, 2] < p['max_z'] * 1000.0) &
            (grad_mag < p['grad_threshold']))
    pts = pts3d[mask]
    gray = rectL[mask]
    if len(pts) > p['max_pts_per_frame']:
        idx = np.random.choice(len(pts), p['max_pts_per_frame'], replace=False)
        pts, gray = pts[idx], gray[idx]
    return pts, gray


def save_ply(path, points, colors):
    n = len(points)
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex " + str(n) +
              "\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
             ('red', '<u1'), ('green', '<u1'), ('blue', '<u1')]
    data = np.empty(n, dtype=dtype)
    data['x'], data['y'], data['z'] = points[:, 0], points[:, 1], points[:, 2]
    data['red'], data['green'], data['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        data.tofile(f)


# ====================== 参数设置对话框 =====================
def show_param_dialog():
    """启动时弹出填空式参数设置窗口，返回确认后的参数字典"""
    import tkinter as tk
    from tkinter import ttk

    result = {}

    root = tk.Tk()
    root.title('三维重建参数设置')
    root.geometry('480x580')
    root.minsize(400, 450)
    root.resizable(True, True)

    entries = {}
    base_width = 480
    base_height = 580
    base_fonts = {'label': 10, 'entry': 10, 'hint': 8, 'button': 10}
    scaling = [False]  # 列表实现可变引用

    def add_entry(parent, label, key, default, hint=''):
        row = ttk.Frame(parent)
        row.pack(fill='x', padx=15, pady=4)
        ttk.Label(row, text=label, width=18).pack(side='left')
        var = tk.StringVar(value=str(default))
        ent = ttk.Entry(row, textvariable=var, width=12)
        ent.pack(side='left', padx=5)
        if hint:
            ttk.Label(row, text=hint, font=('Microsoft YaHei', base_fonts['hint']), foreground='#888').pack(side='left')
        entries[key] = var

    # SGBM 参数
    f1 = ttk.LabelFrame(root, text='SGBM 立体匹配', padding=8)
    f1.pack(fill='x', padx=15, pady=5)
    add_entry(f1, 'Block Size (3-15)', 'sgbm_block', DEFAULTS['sgbm_block'], '奇数，小=细节多')
    add_entry(f1, 'Uniqueness (5-25)', 'sgbm_uniqueness', DEFAULTS['sgbm_uniqueness'], '大=剔除可疑点')
    add_entry(f1, 'Min Z (m)', 'min_z', DEFAULTS['min_z'], '最近距离')
    add_entry(f1, 'Max Z (m)', 'max_z', DEFAULTS['max_z'], '最远距离')

    # 点云滤波
    f2 = ttk.LabelFrame(root, text='点云滤波', padding=8)
    f2.pack(fill='x', padx=15, pady=5)
    add_entry(f2, 'Grad Threshold', 'grad_threshold', DEFAULTS['grad_threshold'], '边缘剔除阈值')
    add_entry(f2, 'Max Pts/Frame', 'max_pts_per_frame', DEFAULTS['max_pts_per_frame'], '单帧点数上限')

    # 地图累积
    f3 = ttk.LabelFrame(root, text='地图累积', padding=8)
    f3.pack(fill='x', padx=15, pady=5)
    add_entry(f3, 'KF Trans (m)', 'kf_trans', DEFAULTS['kf_trans'], '关键帧平移阈值')
    add_entry(f3, 'KF Rot (deg)', 'kf_rot', DEFAULTS['kf_rot'], '关键帧旋转阈值')
    add_entry(f3, 'Voxel (mm)', 'map_voxel', DEFAULTS['map_voxel'] * 1000, '体素大小(毫米)')
    add_entry(f3, 'Max Points', 'map_max_points', DEFAULTS['map_max_points'], '全局点数上限')

    # 轨迹采样
    f4 = ttk.LabelFrame(root, text='轨迹采样', padding=8)
    f4.pack(fill='x', padx=15, pady=5)
    add_entry(f4, 'Interval', 'interval', DEFAULTS['interval'], '每N帧取1帧(1=全部)')

    def on_ok():
        try:
            result['sgbm_block'] = int(entries['sgbm_block'].get())
            result['sgbm_uniqueness'] = int(entries['sgbm_uniqueness'].get())
            result['min_z'] = float(entries['min_z'].get())
            result['max_z'] = float(entries['max_z'].get())
            result['grad_threshold'] = float(entries['grad_threshold'].get())
            result['max_pts_per_frame'] = int(entries['max_pts_per_frame'].get())
            result['kf_trans'] = float(entries['kf_trans'].get())
            result['kf_rot'] = float(entries['kf_rot'].get())
            result['map_voxel'] = float(entries['map_voxel'].get()) / 1000.0
            result['map_max_points'] = int(entries['map_max_points'].get())
            result['interval'] = max(1, int(entries['interval'].get()))
            root.destroy()
        except ValueError as e:
            from tkinter import messagebox
            messagebox.showerror('输入错误', f'参数格式错误: {e}')

    def on_cancel():
        root.destroy()
        raise SystemExit('用户取消参数设置')

    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=15)
    ttk.Button(btn_frame, text='开始重建', command=on_ok, width=15).pack(side='left', padx=10)
    ttk.Button(btn_frame, text='取消', command=on_cancel, width=15).pack(side='left', padx=10)

    # ---- 缩放支持 ----
    def on_resize(event):
        if event.widget != root:
            return
        if scaling[0]:
            return
        scaling[0] = True
        root.after(50, apply_scaling)

    def apply_scaling():
        scaling[0] = False
        w = root.winfo_width()
        h = root.winfo_height()
        if w < 100 or h < 100:
            return
        scale = min(w / base_width, h / base_height)
        scale = max(0.7, min(scale, 1.4))
        # 遍历所有 Label 和 Button 调整字体
        for widget in root.winfo_children():
            _scale_widget_fonts(widget, scale)

    def _scale_widget_fonts(widget, scale):
        """递归调整组件字体"""
        try:
            cls = widget.winfo_class()
            if cls in ('Label', 'TLabel', 'Button', 'TButton', 'TLabelframe.Label'):
                current = widget.cget('font')
                if isinstance(current, tuple) and len(current) >= 2:
                    family, size = current[0], current[1]
                    if isinstance(size, str):
                        size = int(size)
                    new_size = max(6, int(size * scale))
                    # 只调整一次，基于基准字体
                    pass  # 字体已在创建时设置，此处不再递归调整避免混乱
        except Exception:
            pass
        for child in widget.winfo_children():
            _scale_widget_fonts(child, scale)

    root.bind('<Configure>', on_resize)

    root.mainloop()
    return result


# ====================== 主程序 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='EuRoC 数据集目录')
    parser.add_argument('--vio', default=None, help='vio.csv 路径 (默认数据集内)')
    parser.add_argument('--scale', type=float, default=0.5, help='图像缩放 (需与 VINS 分辨率一致)')
    parser.add_argument('--out', default=None, help='输出 PLY 路径')
    args = parser.parse_args()

    # 启动前弹出参数设置对话框
    print('[INIT] 等待参数设置...')
    p = show_param_dialog()
    print(f"[INIT] 参数: block={p['sgbm_block']}, uniq={p['sgbm_uniqueness']}, "
          f"z=[{p['min_z']},{p['max_z']}], interval={p['interval']}")

    vio_path = args.vio or os.path.join(args.dataset, 'vio.csv')
    df_traj = load_vins_trajectory(vio_path)
    print(f"[INIT] 轨迹 {len(df_traj)} 帧, 采样间隔 {p['interval']}")

    left_files = sorted(glob.glob(os.path.join(args.dataset, 'cam0', '*.png')))
    img_ts, img_paths = [], []
    for path in left_files:
        try:
            ts = int(os.path.basename(path).split('.')[0]) / 1e9
            img_ts.append(ts)
            img_paths.append(path)
        except ValueError:
            continue
    img_ts = np.array(img_ts)
    print(f"[INIT] 图像 {len(img_paths)} 对")

    # 首图确定校正 LUT
    sample = cv2.imread(img_paths[0], 0)
    h0, w0 = sample.shape[:2]
    size = (int(w0 * args.scale), int(h0 * args.scale))
    maps = calib.get_rectification_maps(size=size)
    T_i_c0 = calib.T_cam0_to_imu_meters
    T_rect2raw = np.eye(4)
    T_rect2raw[:3, :3] = maps['R1'].T

    # Open3D 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window("离线增量三维重建", 1280, 720)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.08, 0.08, 0.10])
    opt.point_size = 1.5
    pcd = o3d.geometry.PointCloud()
    traj_ls = o3d.geometry.LineSet()
    pcd_added = False
    traj_added = False
    view_initialized = False

    map_points = np.zeros((0, 3), dtype=np.float32)
    map_colors = np.zeros((0, 3), dtype=np.uint8)
    trajectory = []
    last_kf_pose = None
    t_start = time.time()

    # 拼接逻辑: 遍历 VIO 轨迹帧 -> 按时间戳找最近图像帧
    for idx in range(0, len(df_traj), p['interval']):
        row = df_traj.iloc[idx]
        pose_ts = row['timestamp']

        diffs = np.abs(img_ts - pose_ts)
        ci = np.argmin(diffs)
        if diffs[ci] > POSE_SYNC_TOL:
            continue

        p0 = img_paths[ci]
        p1 = p0.replace(os.sep + 'cam0' + os.sep, os.sep + 'cam1' + os.sep)
        if not os.path.exists(p1):
            continue
        img0 = cv2.imread(p0, 0)
        img1 = cv2.imread(p1, 0)
        if img0 is None or img1 is None:
            continue
        if args.scale != 1.0:
            img0 = cv2.resize(img0, size, interpolation=cv2.INTER_AREA)
            img1 = cv2.resize(img1, size, interpolation=cv2.INTER_AREA)

        T_w_i = np.eye(4)
        T_w_i[:3, :3] = quat_to_matrix(row['q_w'], row['q_x'], row['q_y'], row['q_z'])
        T_w_i[:3, 3] = [row['p_x'], row['p_y'], row['p_z']]

        pts_mm, gray = stereo_to_points(img0, img1, maps, p)
        if len(pts_mm) == 0:
            continue

        # 校正左目系 -> 原始左目系 -> IMU系 -> 世界系
        T_w_rect = T_w_i.dot(T_i_c0).dot(T_rect2raw)
        pts_m = pts_mm.astype(np.float64) / 1000.0
        pts_h = np.hstack([pts_m, np.ones((len(pts_m), 1))])
        pts_w = (T_w_rect.dot(pts_h.T)).T[:, :3].astype(np.float32)
        colors = np.stack([gray, gray, gray], axis=-1)

        # 关键帧判断
        p_now = T_w_i[:3, 3]
        R_now = T_w_i[:3, :3]
        is_kf = False
        if last_kf_pose is None:
            is_kf = True
        else:
            dp = np.linalg.norm(p_now - last_kf_pose[:3, 3])
            dR = R_now.dot(last_kf_pose[:3, :3].T)
            cos_angle = np.clip((np.trace(dR) - 1) / 2.0, -1.0, 1.0)
            if dp > p['kf_trans'] or np.degrees(np.arccos(cos_angle)) > p['kf_rot']:
                is_kf = True

        if is_kf:
            last_kf_pose = T_w_i.copy()
            trajectory.append(p_now.copy())
            map_points = np.vstack([map_points, pts_w])
            map_colors = np.vstack([map_colors, colors])
            if len(map_points) > p['map_max_points']:
                map_points, map_colors = voxel_downsample(map_points, map_colors, p['map_voxel'] * 2.0)
            else:
                map_points, map_colors = voxel_downsample(map_points, map_colors, p['map_voxel'])

            # 更新显示
            pcd.points = o3d.utility.Vector3dVector(map_points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(map_colors.astype(np.float64) / 255.0)
            if not pcd_added:
                vis.add_geometry(pcd)
                pcd_added = True
            else:
                vis.update_geometry(pcd)

            if len(trajectory) > 1:
                traj_arr = np.asarray(trajectory)
                traj_ls.points = o3d.utility.Vector3dVector(traj_arr.astype(np.float64))
                traj_ls.lines = o3d.utility.Vector2iVector(
                    np.asarray([[i, i + 1] for i in range(len(traj_arr) - 1)], dtype=np.int32))
                traj_ls.colors = o3d.utility.Vector3dVector(
                    np.tile(np.asarray([[1.0, 0.2, 0.2]]), (len(traj_arr) - 1, 1)))
                if not traj_added:
                    vis.add_geometry(traj_ls)
                    traj_added = True
                else:
                    vis.update_geometry(traj_ls)

            if not view_initialized and len(map_points) > 0:
                ctr = vis.get_view_control()
                ctr.set_front([0, 0, -1])
                ctr.set_up([0, -1, 0])
                ctr.set_lookat([0, 0, 0])
                ctr.set_zoom(0.35)
                view_initialized = True

        if idx % 20 == 0:
            print(f"[RECON] {idx}/{len(df_traj)} | 地图 {len(map_points)} 点 | 关键帧 {len(trajectory)} | "
                  f"耗时 {time.time()-t_start:.1f}s")

        if not vis.poll_events():
            print("[RECON] 窗口已关闭, 提前结束")
            break
        vis.update_renderer()

    vis.destroy_window()

    out_path = args.out or os.path.join(
        args.dataset, 'global_map_%s.ply' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    save_ply(out_path, map_points, map_colors)
    print(f"[SAVE] 全局地图已保存: {out_path} ({len(map_points)} 点, {len(trajectory)} 关键帧)")


if __name__ == '__main__':
    main()
