#pragma once

#include "lio/laser_mapping.h"
#include "localization/localization.h"
#include "loop_closing/online_loop.h"
#include "pub/pub_ROS1.h"

#include <string>
#include <vector>

namespace lightning
{

struct SlamOptions
{
    LaserMapping::Options lio_ctor;
    LaserMapping::Config lio;
    OnlineLoop::Options loop;
    PubROSTopic::Options pub;
    bool with_loop_closing = false;
    std::string lidar_topic = "/livox/lidar";
    std::string imu_topic = "/livox/imu";
    std::vector<std::string> image_topics;
    int lidar_type = 4;
};

struct LocSystemParams
{
    std::string lidar_topic = "/livox/lidar";
    std::string imu_topic = "/livox/imu";
    std::vector<std::string> image_topics;
    bool visual_enabled = false;
};

struct LocOptions
{
    LaserMapping::Options lio_ctor;
    LaserMapping::Config lio;
    loc::LidarLoc::Options lidar_loc;
    loc::Localization::Options localization;
    PubROSTopic::Options pub;
    LocSystemParams system;
};

SlamOptions LoadSlamOptions(const rclcpp::Node::SharedPtr &nh);
LocOptions LoadLocOptions(const rclcpp::Node::SharedPtr &nh);

}  // namespace lightning
