#!/bin/bash
# start_serial_vins.sh
# WSL 侧离线串行解算: roscore + vins_node (无实时桥接/建图节点)
# 用法: wsl -e bash -lc "bash /mnt/c/Users/12939/Desktop/stereo-imu-reconstruction/start_serial_vins.sh"

set -eo pipefail

source /opt/ros/melodic/setup.bash
source /root/vins_ws/devel/setup.bash
export LD_LIBRARY_PATH=/opt/ros/melodic/lib:$LD_LIBRARY_PATH

PROJ="/mnt/c/Users/12939/Desktop/stereo-imu-reconstruction"

pkill -9 -f 'vins_nod[e]' 2>/dev/null || true
pkill -9 -f 'serial_euroc_repla[y]' 2>/dev/null || true
pkill -9 -f 'rosmast[e]r' 2>/dev/null || true
sleep 2

setsid nohup roscore > /tmp/roscore.log 2>&1 < /dev/null &
sleep 5
echo "[SERIAL] roscore started"

CONFIG="$PROJ/vins_config/handheld_stereo_imu_config.yaml"
setsid nohup /root/vins_ws/devel/lib/vins/vins_node "$CONFIG" > /tmp/vins.log 2>&1 < /dev/null &
sleep 5
echo "[SERIAL] vins started"
tail -n 10 /tmp/vins.log

echo ""
echo "[SERIAL] VINS 已就绪, 请回放数据集:"
echo "  wsl -e bash -lc \"python $PROJ/serial_euroc_replay.py --dataset $PROJ/SGM_dataset_serial/<数据集目录> --speed 1.0\""
echo ""
echo "[SERIAL] 轨迹输出: $PROJ/vins_output/vio.csv"
