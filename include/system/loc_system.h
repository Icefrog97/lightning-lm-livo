#ifndef LIGHTNING_LOC_SYSTEM_H
#define LIGHTNING_LOC_SYSTEM_H

#include <tf2_ros/transform_broadcaster.h>
#include "ros/ros2_types.h"
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/keyframe.h"
#include "common/timer.h"
#include "ros/param_loader.h"
#include <vector>

namespace lightning {

namespace loc {
class Localization;
}

class LocSystem {
   public:
    struct Options {
        bool pub_tf_ = true;  // 是否发布tf
    };

    explicit LocSystem(Options options);
    ~LocSystem();

    /// 初始化。opt 由 LoadLocOptions 填好。
    bool Init(const rclcpp::Node::SharedPtr &nh, const LocOptions &opt);

    /// 设置初始化位姿
    void SetInitPose(const SE3& pose);

    /// 处理IMU
    void ImuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &msg);

    /// 处理点云
    void LidarCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
    void LivoxLidarCallback(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg);

    void ImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg, int cam_id);

   private:
    Options options_;

    std::shared_ptr<loc::Localization> loc_ = nullptr;  // 定位接口

    std::atomic_bool loc_started_{false};  // 是否开启定位

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_ = nullptr;

    std::string imu_topic_;
    std::string lidar_topic_;
    std::vector<std::string> image_topics_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::SubscriptionBase::SharedPtr lidar_sub_;
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> image_subs_;
};

};  // namespace lightning

#endif  // LIGHTNING_LOC_SYSTEM_H
