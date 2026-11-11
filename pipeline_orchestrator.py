#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_orchestrator.py
一键式双目 IMU 三维重建流水线编排器。

支持两种模式:
  1. 连接硬件: 启动采集 GUI -> 录制 -> VINS 解算 -> 三维重建
  2. 已有数据集: 选择数据集目录 -> VINS 解算 -> 三维重建

每个阶段弹出窗口提示进度，无需手动切换脚本。
窗口支持全方向拖拽缩放，内部元素自适应布局。
"""

import os
import sys
import time
import glob
import shutil
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

# ====================== 路径配置 =====================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WSL_PROJECT = '/mnt/c/Users/12939/Desktop/stereo-imu-reconstruction'
DATASET_ROOT = os.path.join(PROJECT_DIR, 'SGM_dataset_serial')
VIO_CSV = os.path.join(PROJECT_DIR, 'vins_output', 'vio.csv')

# WSL 命令模板
WSL_CLEAN = 'pkill -9 -f vins_nod[e] 2>/dev/null; pkill -9 -f rosmast[e] 2>/dev/null; pkill -9 -f serial_euroc_repla[y] 2>/dev/null; sleep 1'
WSL_START_VINS = 'bash %s/start_serial_vins.sh' % WSL_PROJECT
WSL_REPLAY = 'python %s/serial_euroc_replay.py --dataset %s --speed 1.0'


class PipelineOrchestrator:
    def __init__(self, root):
        self.root = root
        self.root.title('双目 IMU 三维重建流水线')
        self.root.geometry('720x520')
        self.root.minsize(520, 420)
        self.root.resizable(True, True)

        self.dataset_dir = None
        self.processes = []
        self.running = True
        self.mode = None  # 'capture' or 'existing'

        self._build_ui()
        self._bind_resize()

    # ====================== UI =====================
    def _build_ui(self):
        # 主窗口 grid 权重配置：状态区域可扩展，其余固定
        self.root.grid_rowconfigure(0, weight=0)   # 标题
        self.root.grid_rowconfigure(1, weight=0)   # 模式选择
        self.root.grid_rowconfigure(2, weight=0)   # 数据集选择（隐藏时也占位）
        self.root.grid_rowconfigure(3, weight=1)   # 状态区域（可扩展）
        self.root.grid_rowconfigure(4, weight=0)   # 按钮
        self.root.grid_columnconfigure(0, weight=1)

        # ---- 标题 ----
        self.title_label = ttk.Label(self.root, text='双目 IMU 三维重建流水线',
                          font=('Microsoft YaHei', 18, 'bold'))
        self.title_label.grid(row=0, column=0, pady=(15, 5), sticky='ew')

        # ---- 模式选择 ----
        mode_frame = ttk.LabelFrame(self.root, text='运行模式', padding=10)
        mode_frame.grid(row=1, column=0, padx=20, pady=8, sticky='ew')
        mode_frame.grid_columnconfigure(0, weight=1)

        self.mode_var = tk.StringVar(value='capture')
        ttk.Radiobutton(mode_frame, text='连接硬件采集新数据',
                        variable=self.mode_var, value='capture',
                        command=self._on_mode_change).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Radiobutton(mode_frame, text='选择已有数据集离线处理',
                        variable=self.mode_var, value='existing',
                        command=self._on_mode_change).grid(row=1, column=0, sticky='w', pady=2)

        # ---- 数据集选择（仅 existing 模式显示） ----
        self.dataset_var = tk.StringVar(value='未选择数据集')
        self.dataset_frame = ttk.Frame(self.root)
        self.dataset_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(self.dataset_frame, text='数据集:').grid(row=0, column=0, padx=5)
        ttk.Label(self.dataset_frame, textvariable=self.dataset_var,
                  foreground='#0066aa').grid(row=0, column=1, padx=5, sticky='ew')
        ttk.Button(self.dataset_frame, text='浏览...',
                   command=self._browse_dataset).grid(row=0, column=2, padx=5)

        # ---- 状态显示 ----
        status_frame = ttk.LabelFrame(self.root, text='状态', padding=10)
        status_frame.grid(row=3, column=0, padx=20, pady=8, sticky='nsew')
        status_frame.grid_columnconfigure(0, weight=1)
        status_frame.grid_rowconfigure(3, weight=1)  # 日志区域可扩展

        self.stage_var = tk.StringVar(value='就绪')
        self.stage_label = ttk.Label(status_frame, textvariable=self.stage_var,
                  font=('Microsoft YaHei', 12))
        self.stage_label.grid(row=0, column=0, sticky='w')

        self.detail_var = tk.StringVar(value='请选择运行模式后点击开始')
        self.detail_label = ttk.Label(status_frame, textvariable=self.detail_var,
                  font=('Microsoft YaHei', 10), foreground='#555')
        self.detail_label.grid(row=1, column=0, sticky='w', pady=2)

        self.progress = ttk.Progressbar(status_frame, mode='determinate', maximum=100)
        self.progress.grid(row=2, column=0, sticky='ew', pady=8)

        # 日志区域：使用 ScrolledText 替代 Label，支持滚动和自适应
        self.log_text = scrolledtext.ScrolledText(
            status_frame, wrap=tk.WORD, font=('Consolas', 9),
            foreground='#0066aa', background='#f8f8f8',
            height=6, state='disabled'
        )
        self.log_text.grid(row=3, column=0, sticky='nsew')

        # ---- 按钮 ----
        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=4, column=0, pady=15)
        self.start_btn = ttk.Button(btn_frame, text='开始流水线',
                                    command=self._start, width=15)
        self.start_btn.pack(side='left', padx=5)
        self.cancel_btn = ttk.Button(btn_frame, text='取消',
                                     command=self._cancel, state='disabled', width=15)
        self.cancel_btn.pack(side='left', padx=5)

        # 基准尺寸，用于缩放计算
        self._base_width = 720
        self._base_height = 520
        self._base_fonts = {
            'title': 18,
            'stage': 12,
            'detail': 10,
            'log': 9,
        }
        self._scaling = False

    def _bind_resize(self):
        """绑定窗口缩放事件，动态调整字体保持比例"""
        self.root.bind('<Configure>', self._on_window_resize)

    def _on_window_resize(self, event):
        """窗口大小改变时，按比例调整字体"""
        if event.widget != self.root:
            return
        # 防抖：避免频繁触发
        if self._scaling:
            return
        self._scaling = True
        self.root.after(50, self._apply_scaling)

    def _apply_scaling(self):
        """实际执行缩放调整"""
        self._scaling = False
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        if w < 100 or h < 100:
            return

        # 按窗口面积比例计算缩放因子，限制在 0.7~1.5 之间
        scale = min(w / self._base_width, h / self._base_height)
        scale = max(0.7, min(scale, 1.5))

        # 调整字体
        def scale_font(base_size):
            return int(base_size * scale)

        self.title_label.config(font=('Microsoft YaHei', scale_font(self._base_fonts['title']), 'bold'))
        self.stage_label.config(font=('Microsoft YaHei', scale_font(self._base_fonts['stage'])))
        self.detail_label.config(font=('Microsoft YaHei', scale_font(self._base_fonts['detail'])))
        self.log_text.config(font=('Consolas', scale_font(self._base_fonts['log'])))

    def _on_mode_change(self):
        if self.mode_var.get() == 'existing':
            self.dataset_frame.grid(row=2, column=0, padx=20, pady=5, sticky='ew')
        else:
            self.dataset_frame.grid_forget()

    def _browse_dataset(self):
        path = filedialog.askdirectory(initialdir=DATASET_ROOT, title='选择数据集目录')
        if path:
            # 验证数据集结构
            if not os.path.isdir(os.path.join(path, 'cam0')) or not os.path.isdir(os.path.join(path, 'cam1')):
                messagebox.showerror('错误', '所选目录缺少 cam0/cam1 子目录，不是有效数据集')
                return
            if not os.path.exists(os.path.join(path, 'imu0', 'data.csv')):
                messagebox.showerror('错误', '所选目录缺少 imu0/data.csv，不是有效数据集')
                return
            n_cam0 = len(glob.glob(os.path.join(path, 'cam0', '*.png')))
            if n_cam0 == 0:
                messagebox.showerror('错误', '所选目录 cam0 中没有图像数据')
                return
            self.dataset_var.set(os.path.basename(path))
            self.dataset_dir = path
            self._log(f'已选择数据集: {os.path.basename(path)} ({n_cam0} 对图像)')

    def _log(self, msg):
        """线程安全的日志追加，自动滚动到底部"""
        def append():
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, msg + '\n')
            self.log_text.see(tk.END)
            self.log_text.config(state='disabled')
        self.root.after(0, append)

    def _set_stage(self, stage, detail, progress):
        self.root.after(0, lambda: self._update_ui(stage, detail, progress))

    def _update_ui(self, stage, detail, progress):
        self.stage_var.set(stage)
        self.detail_var.set(detail)
        self.progress['value'] = progress

    def _start(self):
        self.mode = self.mode_var.get()
        if self.mode == 'existing' and not self.dataset_dir:
            messagebox.showwarning('提示', '请先选择数据集目录')
            return

        self.start_btn.config(state='disabled')
        self.cancel_btn.config(state='normal')
        threading.Thread(target=self._run, daemon=True).start()

    def _cancel(self):
        self.running = False
        self._cleanup_processes()
        self._set_stage('已取消', '流水线已被用户取消', 0)
        self.start_btn.config(state='normal')
        self.cancel_btn.config(state='disabled')

    def _cleanup_processes(self):
        for p in self.processes:
            try:
                p.terminate()
            except Exception:
                pass
        subprocess.run(['wsl', '-e', 'bash', '-c', WSL_CLEAN], capture_output=True, timeout=5)

    # ====================== 流水线 =====================
    def _run(self):
        try:
            if self.mode == 'capture':
                self._stage_capture()
                if not self.running:
                    return
            else:
                self._set_stage('模式: 已有数据集', f'使用 {os.path.basename(self.dataset_dir)}', 20)

            self._stage_vins()
            if not self.running:
                return
            self._stage_recon()

            self._set_stage('全部完成', '三维重建已结束，地图已保存', 100)
            self.root.after(0, lambda: messagebox.showinfo('完成', '三维重建流水线已完成！\n地图保存在数据集目录中。'))
        except Exception as exc:
            self._set_stage('错误', str(exc), 0)
            self.root.after(0, lambda e=exc: messagebox.showerror('错误', f'流水线执行失败:\n{e}'))
        finally:
            self.root.after(0, lambda: self.start_btn.config(state='normal'))
            self.root.after(0, lambda: self.cancel_btn.config(state='disabled'))

    # ---------- 阶段 1: 采集 ----------
    def _stage_capture(self):
        self._set_stage('阶段 1/3: 数据采集', '启动采集程序...', 5)
        self._log('启动 serial_stereo_capture.py')

        existing = set(glob.glob(os.path.join(DATASET_ROOT, '*')))

        proc = subprocess.Popen(
            [sys.executable, os.path.join(PROJECT_DIR, 'serial_stereo_capture.py')],
            cwd=PROJECT_DIR
        )
        self.processes.append(proc)

        self._set_stage('阶段 1/3: 数据采集', '请调节曝光/增益，按 R 录制，Q 退出', 10)
        self._log('等待用户完成采集...')

        # 等待新数据集目录出现（serial_stereo_capture 启动即创建目录）
        while self.running:
            time.sleep(2)
            current = set(glob.glob(os.path.join(DATASET_ROOT, '*')))
            new = current - existing
            if new:
                self.dataset_dir = list(new)[0]
                break
            if proc.poll() is not None:
                raise RuntimeError('采集程序已退出但未检测到新数据集')

        if not self.running:
            return

        name = os.path.basename(self.dataset_dir)
        self._set_stage('阶段 1/3: 数据采集', f'数据集 {name} 采集中... 请按 R 录制，Q 退出', 15)
        self._log(f'检测到数据集目录: {name}，等待采集程序退出...')

        # 等待用户按 Q 退出采集程序
        while self.running and proc.poll() is None:
            time.sleep(1)

        if not self.running:
            return

        # 确认目录内有实际数据
        n_cam0 = len(glob.glob(os.path.join(self.dataset_dir, 'cam0', '*.png')))
        n_imu = sum(1 for _ in open(os.path.join(self.dataset_dir, 'imu0', 'data.csv'))) - 1
        if n_cam0 == 0:
            raise RuntimeError(f'数据集 {name} 中没有图像数据，请重新采集')
        if n_imu <= 1:
            raise RuntimeError(f'数据集 {name} 中 IMU 数据为空，请检查串口连接')

        self._set_stage('阶段 1/3: 数据采集', f'数据集 {name} 已保存 ({n_cam0} 图, {n_imu} IMU)', 30)
        self._log(f'采集完成: {n_cam0} 对图像, {n_imu} 条 IMU')

    # ---------- 阶段 2: VINS 解算 ----------
    def _stage_vins(self):
        self._set_stage('阶段 2/3: VINS 轨迹解算', '清理 WSL 残留进程...', 35)
        self._log('启动 VINS-Fusion...')

        subprocess.run(['wsl', '-e', 'bash', '-c', WSL_CLEAN], capture_output=True, timeout=10)

        vins_proc = subprocess.Popen(
            ['wsl', '-e', 'bash', '-c', WSL_START_VINS],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        self.processes.append(vins_proc)

        # 等待 VINS 真正就绪 (检测 vins.log 出现 "waiting for image and imu")
        self._set_stage('阶段 2/3: VINS 轨迹解算', '等待 VINS 初始化...', 40)
        vins_ready = False
        for i in range(30):
            time.sleep(2)
            try:
                out = subprocess.run(
                    ['wsl', '-e', 'bash', '-c', 'grep -c "waiting for image and imu" /tmp/vins.log 2>/dev/null || echo 0'],
                    capture_output=True, text=True, timeout=5
                )
                if out.stdout.strip() != '0':
                    vins_ready = True
                    break
            except Exception:
                pass
            self._log(f'等待 VINS 就绪... ({i*2}s)')
        if not vins_ready:
            self._log('VINS 就绪检测超时，继续执行')

        self._set_stage('阶段 2/3: VINS 轨迹解算', '回放数据集中...', 45)
        self._log('启动数据回放...')

        dataset_wsl = self.dataset_dir.replace('\\', '/').replace(PROJECT_DIR.replace('\\', '/'), WSL_PROJECT)
        replay_cmd = WSL_REPLAY % (WSL_PROJECT, dataset_wsl)
        replay_proc = subprocess.Popen(
            ['wsl', '-e', 'bash', '-c', replay_cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        self.processes.append(replay_proc)

        # 等待回放结束
        while self.running and replay_proc.poll() is None:
            time.sleep(2)
            lines = sum(1 for _ in open(VIO_CSV)) if os.path.exists(VIO_CSV) else 0
            if lines > 0:
                prog = min(45 + int(lines * 30 / 400), 70)
                self._set_stage('阶段 2/3: VINS 轨迹解算', f'已解算 {lines} 帧轨迹', prog)
                self._log(f'回放中... vio.csv: {lines} 帧')

        if not self.running:
            return

        # 回放结束后，等待 VINS 处理完队列数据 (vio.csv 稳定 + 缓冲)
        self._set_stage('阶段 2/3: VINS 轨迹解算', '等待 VINS 收尾...', 72)
        self._log('回放完成，等待 VINS 处理剩余数据...')
        last_size = -1
        stable_count = 0
        max_lines = 0
        while self.running:
            time.sleep(5)
            if not os.path.exists(VIO_CSV):
                continue
            size = os.path.getsize(VIO_CSV)
            lines = sum(1 for _ in open(VIO_CSV))
            max_lines = max(max_lines, lines)
            if size == last_size:
                stable_count += 1
                if stable_count >= 3:
                    break
            else:
                stable_count = 0
                last_size = size
                self._log(f'VINS 收尾中... vio.csv: {lines} 帧, {size/1024:.1f} KB')
        time.sleep(5)  # 额外缓冲

        if not self.running:
            return

        if max_lines == 0:
            raise RuntimeError('VINS 未输出任何轨迹，请检查数据集和 VINS 配置')

        # 拷贝轨迹到数据集
        shutil.copy(VIO_CSV, os.path.join(self.dataset_dir, 'vio.csv'))
        self._set_stage('阶段 2/3: VINS 轨迹解算', f'轨迹已保存 ({max_lines} 帧)', 80)
        self._log(f'轨迹已拷贝到数据集, 共 {max_lines} 帧')

        subprocess.run(['wsl', '-e', 'bash', '-c', WSL_CLEAN], capture_output=True, timeout=10)

    # ---------- 阶段 3: 增量重建 ----------
    def _stage_recon(self):
        self._set_stage('阶段 3/3: 增量三维重建', '启动离线重建...', 85)
        self._log('启动 serial_offline_recon.py')

        proc = subprocess.Popen(
            [sys.executable, os.path.join(PROJECT_DIR, 'serial_offline_recon.py'),
             '--dataset', self.dataset_dir],
            cwd=PROJECT_DIR
        )
        self.processes.append(proc)

        self._set_stage('阶段 3/3: 增量三维重建', '重建进行中，请查看 Open3D 窗口...', 90)
        self._log('等待重建完成...')

        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f'重建程序异常退出 (code={proc.returncode})')

        ply_files = glob.glob(os.path.join(self.dataset_dir, 'global_map_*.ply'))
        if ply_files:
            latest = max(ply_files, key=os.path.getctime)
            size_mb = os.path.getsize(latest) / 1024 / 1024
            self._set_stage('阶段 3/3: 增量三维重建', f'地图已保存 ({size_mb:.1f} MB)', 98)
            self._log(f'地图: {os.path.basename(latest)} ({size_mb:.1f} MB)')
        else:
            self._set_stage('阶段 3/3: 增量三维重建', '重建完成', 98)


def main():
    root = tk.Tk()
    app = PipelineOrchestrator(root)
    root.mainloop()


if __name__ == '__main__':
    main()
