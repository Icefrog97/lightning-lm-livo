#!/usr/bin/env python3
import sys
import numpy as np
import rosbag
from scipy.spatial.transform import Rotation as R

def estimate_tilt(bag_path, imu_topic):
    print(f"正在读取 Bag 文件: {bag_path}")
    print(f"订阅的 IMU Topic: {imu_topic}")
    
    acc_list = []
    
    # 1. 提取 IMU 加速度数据
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[imu_topic]):
                acc = [msg.linear_acceleration.x, 
                       msg.linear_acceleration.y, 
                       msg.linear_acceleration.z]
                acc_list.append(acc)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    if not acc_list:
        print(f"错误: 在 Topic '{imu_topic}' 中未找到任何消息，请检查 Topic 名称。")
        return

    acc_data = np.array(acc_list)
    
    # 2. 计算平均重力向量 (取平均值以过滤高频振动噪声)
    mean_acc = np.mean(acc_data, axis=0)
    norm_acc = np.linalg.norm(mean_acc)
    
    if norm_acc == 0:
        print("错误: 加速度数据异常，模长为 0。")
        return
        
    # 归一化加速度方向向量 (重力方向)
    g_unit = mean_acc / norm_acc
    ax, ay, az = g_unit[0], g_unit[1], g_unit[2]

    # 3. 计算相对于地面的 Roll 和 Pitch
    # 注意：标准 ISO 80000-2 / ROS 坐标系约定 (Z 向上, X 向前, Y 向左)
    # Roll  (绕 X 轴旋转角度)
    roll_rad = np.arctan2(ay, az)
    # Pitch (绕 Y 轴旋转角度)
    pitch_rad = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

    roll_deg = np.degrees(roll_rad)
    pitch_deg = np.degrees(pitch_rad)
    
    # 计算整体与水平面（Z轴正方向）的模长夹角/总倾角
    # 假定理想情况下 Z 轴垂直向上 (az -> 1)
    total_tilt_deg = np.degrees(np.arccos(np.clip(az, -1.0, 1.0)))

    # 4. 输出结果
    print("\n" + "="*40)
    print("      IMU 相对地面倾斜角度估计结果      ")
    print("="*40)
    print(f"采样数据帧数: {len(acc_list)}")
    print(f"平均加速度向量 (m/s²): X={mean_acc[0]:.3f}, Y={mean_acc[1]:.3f}, Z={mean_acc[2]:.3f}")
    print(f"测得重力加速度大小: {norm_acc:.3f} m/s²")
    print("-" * 40)
    print(f"横滚角 (Roll)  : {roll_deg:+.2f}°  (绕 X 轴侧倾)")
    print(f"俯仰角 (Pitch) : {pitch_deg:+.2f}°  (绕 Y 轴前后仰)")
    print(f"总倾斜角 (Tilt): {total_tilt_deg:.2f}°   (Z 轴偏离铅垂线的夹角)")
    print("="*40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使用方法: python3 estimate_imu_tilt.py <your_bag_file.bag> <imu_topic>")
        print("示  例  : python3 estimate_imu_tilt.py imu_data.bag /imu/data")
        sys.exit(1)

    bag_file = sys.argv[1]
    topic_name = sys.argv[2]
    estimate_tilt(bag_file, topic_name)