#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serial_stereo_capture.py
串行采集 GUI：预览双目 + 调整曝光/增益 + 录制 EuRoC 数据集
替代原实时链路中的 win_tcp_publisher.py，采集后不再实时推送，而是存盘供后续离线 VINS/重建。
"""
import os
import time
import datetime
import threading
import queue
import serial
import struct
import math
from ctypes import *

import cv2
import numpy as np

from MvCameraControl_class import *

# ====================== 配置 =====================
COM_PORT = 'COM5'
IMU_BAUDRATE = 115200
INIT_EXPOSURE = 9000      # us
INIT_GAIN = 150           # 0.1 dB
MAX_EXPOSURE = 100000
MAX_GAIN = 240
TARGET_FPS = 20
PREVIEW_SCALE = 0.45
SAVE_ROOT = 'SGM_dataset_serial'
WINDOW_NAME = 'Serial Stereo Capture'

is_running = True
is_recording = False
save_queue = queue.Queue(maxsize=500)

# ====================== HIPNUC IMU =====================
def crc16_update(src_bytes, current_crc=0):
    crc = current_crc
    for byte in src_bytes:
        crc = crc ^ (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
            crc &= 0xFFFF
    return crc


class HIPNUC_Reader:
    def __init__(self, port, baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        self.buffer = bytearray()

    def read_frame(self):
        while is_running:
            if self.ser.in_waiting > 0:
                self.buffer.extend(self.ser.read(self.ser.in_waiting))
            else:
                self.buffer.extend(self.ser.read(64))
            while len(self.buffer) >= 2:
                if self.buffer[0] == 0x5A and self.buffer[1] == 0xA5:
                    break
                self.buffer.pop(0)
            if len(self.buffer) < 4:
                continue
            payload_len = struct.unpack('<H', self.buffer[2:4])[0]
            frame_len = 6 + payload_len
            if len(self.buffer) < frame_len:
                continue
            frame = self.buffer[:frame_len]
            self.buffer = self.buffer[frame_len:]
            recv_crc = struct.unpack('<H', frame[4:6])[0]
            payload = frame[6:frame_len]
            calc_crc = crc16_update(frame[0:4])
            calc_crc = crc16_update(payload, calc_crc)
            if calc_crc != recv_crc:
                continue
            return self.parse_payload(payload)
        return None

    def parse_payload(self, payload):
        if payload[0] == 0x91 and len(payload) == 76:
            parsed = struct.unpack('< B H b f I 3f 3f 3f f f f 4f', payload)
            acc = [v * 9.80665 for v in parsed[5:8]]
            gyr = [v * math.pi / 180.0 for v in parsed[8:11]]
            return parsed[4], gyr, acc
        return None


class ImuTimeAligner:
    def __init__(self):
        self.offset_ns = None

    def stamp(self, dev_ms, pc_ns):
        dev_ns = dev_ms * 1000000
        obs = pc_ns - dev_ns
        if self.offset_ns is None or obs < self.offset_ns:
            self.offset_ns = obs
        else:
            self.offset_ns += int((obs - self.offset_ns) * 0.002)
        return dev_ns + self.offset_ns


# ====================== 相机 =====================
def process_frame_to_image(stFrame):
    if stFrame.pBufAddr is None or stFrame.stFrameInfo.nFrameLen == 0:
        return None
    nSize = stFrame.stFrameInfo.nFrameLen
    width, height = stFrame.stFrameInfo.nWidth, stFrame.stFrameInfo.nHeight
    pixel_type = stFrame.stFrameInfo.enPixelType
    pData = (c_ubyte * nSize)()
    cdll.msvcrt.memcpy(byref(pData), stFrame.pBufAddr, nSize)
    data = np.frombuffer(pData, dtype=np.uint8)
    try:
        if pixel_type == PixelType_Gvsp_Mono8:
            return data.reshape((height, width))
        elif pixel_type == PixelType_Gvsp_BGR8_Packed:
            return cv2.cvtColor(data.reshape((height, width, 3)), cv2.COLOR_BGR2GRAY)
        elif pixel_type == PixelType_Gvsp_RGB8_Packed:
            return cv2.cvtColor(data.reshape((height, width, 3)), cv2.COLOR_RGB2GRAY)
        elif pixel_type == PixelType_Gvsp_BayerGB8:
            bgr = cv2.cvtColor(data.reshape((height, width)), cv2.COLOR_BAYER_GB2BGR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None
    return None


def init_cameras(exposure_us, gain_raw):
    MvCamera.MV_CC_Initialize()
    deviceList = MV_CC_DEVICE_INFO_LIST()
    MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, deviceList)
    if deviceList.nDeviceNum < 2:
        print(f"[CAM 错误] 只发现 {deviceList.nDeviceNum} 台相机")
        return None
    cams = []
    for i in range(2):
        stDevInfo = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        cam = MvCamera()
        if cam.MV_CC_CreateHandle(stDevInfo) != 0:
            continue
        if cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
            continue
        cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)
        cam.MV_CC_SetEnumValue("TriggerMode", 1)           # MV_TRIGGER_MODE_ON
        cam.MV_CC_SetEnumValue("TriggerSource", 7)         # MV_TRIGGER_SOURCE_SOFTWARE
        cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us))
        cam.MV_CC_SetFloatValue("Gain", float(gain_raw / 10.0))
        if cam.MV_CC_StartGrabbing() != 0:
            continue
        cams.append(cam)
        print(f"[CAM] Cam {i} 就绪")
    return cams if len(cams) == 2 else None


# ====================== 后台线程 =====================
def imu_worker(csv_path):
    print(f"[IMU] 连接串口 {COM_PORT} ...")
    try:
        imu = HIPNUC_Reader(COM_PORT, IMU_BAUDRATE)
        print("[IMU] 连接成功")
    except Exception as e:
        print(f"[IMU 错误] 无法打开 {COM_PORT}: {e}")
        return
    aligner = ImuTimeAligner()
    with open(csv_path, 'w') as f:
        f.write("timestamp,w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n")
        count = 0
        while is_running:
            data = imu.read_frame()
            if data is None:
                continue
            dev_ms, gyr, acc = data
            ts = aligner.stamp(dev_ms, time.time_ns())
            if is_recording:
                f.write(f"{ts},{gyr[0]:.6f},{gyr[1]:.6f},{gyr[2]:.6f},{acc[0]:.6f},{acc[1]:.6f},{acc[2]:.6f}\n")
                count += 1
                if count % 500 == 0:
                    print(f"   [IMU] 已录制 {count} 帧")


def save_worker():
    while is_running or not save_queue.empty():
        try:
            task = save_queue.get(timeout=1.0)
            cam0_path, img0, cam1_path, img1 = task
            cv2.imwrite(cam0_path, img0, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(cam1_path, img1, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            save_queue.task_done()
        except queue.Empty:
            pass


# ====================== 主程序 =====================
def create_session_folder():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_path = os.path.join(SAVE_ROOT, timestamp)
    os.makedirs(os.path.join(session_path, "cam0"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "cam1"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "imu0"), exist_ok=True)
    return session_path, os.path.join(session_path, "cam0"), os.path.join(session_path, "cam1"), \
        os.path.join(session_path, "imu0", "data.csv")


def set_exposure(cams, exp_us):
    for cam in cams:
        cam.MV_CC_SetFloatValue("ExposureTime", float(exp_us))


def set_gain(cams, gain_raw):
    for cam in cams:
        cam.MV_CC_SetFloatValue("Gain", float(gain_raw / 10.0))


def draw_ui(frame, is_recording, img_counter, qsize, curr_exp, curr_gain, fps_actual):
    """在预览帧上绘制 UI 信息"""
    h, w = frame.shape[:2]

    # 顶部状态栏背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # 录制状态
    status = "● REC" if is_recording else "○ PREVIEW"
    color = (0, 0, 255) if is_recording else (0, 200, 200)
    cv2.putText(frame, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 统计信息
    cv2.putText(frame, f"Saved: {img_counter}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"Queue: {qsize}", (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps_actual:.1f}", (350, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # 参数显示
    cv2.putText(frame, f"Exp: {curr_exp} us  |  Gain: {curr_gain/10.0:.1f} dB", (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # 底部操作提示
    cv2.rectangle(frame, (0, h-40), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "[R] Record  [S] Single Shot  [Q] Quit", (15, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    return frame


def main():
    global is_recording, is_running
    session_dir, cam0_dir, cam1_dir, imu_csv = create_session_folder()
    print(f"[SYS] 数据集将保存到: {session_dir}")

    cams = init_cameras(INIT_EXPOSURE, INIT_GAIN)
    if cams is None:
        return

    threading.Thread(target=imu_worker, args=(imu_csv,), daemon=True).start()
    threading.Thread(target=save_worker, daemon=True).start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 560)
    cv2.createTrackbar("Exp(us)", WINDOW_NAME, INIT_EXPOSURE, MAX_EXPOSURE, lambda x: None)
    cv2.createTrackbar("Gain(0.1dB)", WINDOW_NAME, INIT_GAIN, MAX_GAIN, lambda x: None)

    last_exp, last_gain = INIT_EXPOSURE, INIT_GAIN
    frame_interval = 1.0 / TARGET_FPS
    last_trigger = time.time()
    img_counter = 0
    fps_counter = 0
    fps_time = time.time()
    fps_actual = 0.0

    print("\n[ R ] 开始/停止录制   [ S ] 单拍   [ Q ] 退出")
    try:
        while is_running:
            curr_exp = max(50, cv2.getTrackbarPos("Exp(us)", WINDOW_NAME))
            curr_gain = cv2.getTrackbarPos("Gain(0.1dB)", WINDOW_NAME)
            if curr_exp != last_exp:
                set_exposure(cams, curr_exp)
                last_exp = curr_exp
            if curr_gain != last_gain:
                set_gain(cams, curr_gain)
                last_gain = curr_gain

            now = time.time()
            if now - last_trigger < frame_interval:
                time.sleep(0.0005)
                continue
            last_trigger += frame_interval
            if now - last_trigger > frame_interval:
                last_trigger = now
                for cam in cams:
                    cam.MV_CC_ClearImageBuffer()

            ts_ns = time.time_ns()
            for cam in cams:
                cam.MV_CC_SetCommandValue("TriggerSoftware")

            frames = []
            for cam in cams:
                stFrame = MV_FRAME_OUT()
                if cam.MV_CC_GetImageBuffer(stFrame, 2000) == 0:
                    frames.append(process_frame_to_image(stFrame))
                    cam.MV_CC_FreeImageBuffer(stFrame)
                else:
                    frames.append(None)
            if frames[0] is None or frames[1] is None:
                continue

            if frames[0].shape != frames[1].shape:
                frames[1] = cv2.resize(frames[1], (frames[0].shape[1], frames[0].shape[0]))

            # FPS 计算
            fps_counter += 1
            if now - fps_time >= 1.0:
                fps_actual = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            # 预览 + UI
            preview = np.hstack((frames[0], frames[1]))
            ph, pw = preview.shape[:2]
            preview = cv2.resize(preview, (int(pw * PREVIEW_SCALE), int(ph * PREVIEW_SCALE)))
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            preview = draw_ui(preview, is_recording, img_counter, save_queue.qsize(),
                              curr_exp, curr_gain, fps_actual)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                is_recording = not is_recording
                print(f"\n>>> 录制状态: {'ON' if is_recording else 'OFF'} <<<\n")
            if key == ord('s') and not is_recording:
                is_recording = True
                time.sleep(0.01)
                is_recording = False
                print(">>> 单拍已触发 <<<")

            # 存盘
            if is_recording or key == ord('s'):
                img_name = f"{ts_ns}.png"
                p0 = os.path.join(cam0_dir, img_name)
                p1 = os.path.join(cam1_dir, img_name)
                try:
                    save_queue.put_nowait((p0, frames[0], p1, frames[1]))
                    img_counter += 1
                except queue.Full:
                    print("[WARN] 存盘队列满，丢弃当前帧")

    except KeyboardInterrupt:
        pass
    finally:
        is_running = False
        if not save_queue.empty():
            print(f"[SYS] 等待写入剩余 {save_queue.qsize()} 张图像...")
            save_queue.join()
        for cam in cams:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        cv2.destroyAllWindows()
        print(f"[SYS] 采集结束，数据集保存在 {session_dir}")


if __name__ == '__main__':
    main()
