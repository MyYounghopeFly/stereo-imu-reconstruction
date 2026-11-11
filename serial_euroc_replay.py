#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
serial_euroc_replay.py
WSL 侧离线回放节点：读取 EuRoC 数据集 (cam0/ cam1/ imu0/data.csv)，
按原始时间戳发布 /cam0/image_raw /cam1/image_raw /imu0 供 VINS-Fusion 解算。

运行 (WSL, python2.7 + rospy):
  source /opt/ros/melodic/setup.bash
  python serial_euroc_replay.py --dataset /mnt/c/.../SGM_dataset_serial/xxx --speed 1.0
"""
import os
import sys
import glob
import argparse
import time

if '/opt/ros/melodic/lib/python2.7/dist-packages' not in sys.path:
    sys.path.insert(0, '/opt/ros/melodic/lib/python2.7/dist-packages')
import rospy
from sensor_msgs.msg import Image, Imu
import cv2
import numpy as np

RECON_SCALE = 0.5  # 与 VINS 配置 image_width/height 一致 (2448x2048 -> 1224x1024)


def load_dataset(dataset_dir):
    cam0_files = sorted(glob.glob(os.path.join(dataset_dir, 'cam0', '*.png')))
    cam1_dir = os.path.join(dataset_dir, 'cam1')
    imu_csv = os.path.join(dataset_dir, 'imu0', 'data.csv')

    # (ts_sec, kind, payload)  kind: 0=imu 1=stereo
    events = []
    with open(imu_csv) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 7:
                continue
            ts = float(parts[0]) / 1e9
            vals = [float(x) for x in parts[1:]]
            events.append((ts, 0, vals))
    for p0 in cam0_files:
        ts = float(os.path.basename(p0).split('.')[0]) / 1e9
        p1 = os.path.join(cam1_dir, os.path.basename(p0))
        if os.path.exists(p1):
            events.append((ts, 1, (p0, p1)))
    events.sort(key=lambda e: e[0])
    return events


def make_image_msg(img, stamp, frame_id):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = img.shape[:2]
    msg.encoding = 'mono8'
    msg.is_bigendian = 0
    msg.step = msg.width
    msg.data = img.tobytes()
    return msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--speed', type=float, default=1.0, help='回放倍速, 默认 1.0 实时')
    args = parser.parse_args()

    events = load_dataset(args.dataset)
    if not events:
        print('[REPLAY] 数据集为空: %s' % args.dataset)
        return
    print('[REPLAY] 共 %d 个事件, 时长 %.1f s' % (
        len(events), events[-1][0] - events[0][0]))

    rospy.init_node('serial_euroc_replay')
    pub0 = rospy.Publisher('/cam0/image_raw', Image, queue_size=50)
    pub1 = rospy.Publisher('/cam1/image_raw', Image, queue_size=50)
    pub_imu = rospy.Publisher('/imu0', Imu, queue_size=1000)

    t0_data = events[0][0]
    t0_wall = time.time()
    n_imu, n_img = 0, 0

    for ts, kind, payload in events:
        if rospy.is_shutdown():
            break
        # 按倍速对齐墙钟
        target = t0_wall + (ts - t0_data) / args.speed
        dt = target - time.time()
        if dt > 0:
            time.sleep(dt)

        stamp = rospy.Time.from_sec(ts)
        if kind == 0:
            msg = Imu()
            msg.header.stamp = stamp
            msg.header.frame_id = 'imu0'
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = payload[0:3]
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = payload[3:6]
            pub_imu.publish(msg)
            n_imu += 1
        else:
            img0 = cv2.imread(payload[0], cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(payload[1], cv2.IMREAD_GRAYSCALE)
            if img0 is None or img1 is None:
                continue
            if RECON_SCALE != 1.0:
                w = int(img0.shape[1] * RECON_SCALE)
                h = int(img0.shape[0] * RECON_SCALE)
                img0 = cv2.resize(img0, (w, h), interpolation=cv2.INTER_AREA)
                img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
            pub0.publish(make_image_msg(img0, stamp, 'cam0'))
            pub1.publish(make_image_msg(img1, stamp, 'cam1'))
            n_img += 1
            if n_img % 100 == 0:
                print('[REPLAY] 已回放 %d 对图像, %d 条 IMU' % (n_img, n_imu))

    print('[REPLAY] 回放完成: %d 对图像, %d 条 IMU' % (n_img, n_imu))
    print('[REPLAY] 保持节点 5s 让 VINS 收尾...')
    time.sleep(5)


if __name__ == '__main__':
    main()
