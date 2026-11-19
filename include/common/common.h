//
// Created by yxliang on 26-1-28.
//

#ifndef LIDAR_SLAM_COMMON_H
#define LIDAR_SLAM_COMMON_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iomanip>

namespace lightning {

// 工具函数
template <typename T>
void printV(T &lidar_slam_out, const Eigen::Vector3d& v) {
    lidar_slam_out << std::fixed << std::setprecision(8) << "[" << v.x() << ", " << v.y() << ", " << v.z() << "]\n";
}

template <typename T>
void printQ(T &lidar_slam_out, const Eigen::Quaterniond& q) {
    lidar_slam_out << std::fixed << std::setprecision(8) << "[" << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << "]\n";
}

template <typename T, typename T1, typename T2>
void setPoseStamp(T &out, T1 t, T2 q)
{
    out.pose.position.x = t.x();
    out.pose.position.y = t.y();
    out.pose.position.z = t.z();
    out.pose.orientation.x = q.x();
    out.pose.orientation.y = q.y();
    out.pose.orientation.z = q.z();
    out.pose.orientation.w = q.w();
};

}  // namespace 

#endif