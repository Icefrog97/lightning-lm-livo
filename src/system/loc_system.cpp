#include "system/loc_system.h"
#include "localization/localization.h"
#include <cmath>
#include <csignal>
#include <functional>
#include <boost/bind/bind.hpp>

namespace lightning
{
    LocSystem::LocSystem(LocSystem::Options options) : options_(options)
    {
        signal(SIGINT, lightning::debug::SigHandle);
    }

    LocSystem::~LocSystem() { loc_->Finish(); }

    bool LocSystem::Init(const rclcpp::Node::SharedPtr &nh, const LocOptions &opt)
    {
        lidar_topic_ = opt.system.lidar_topic;
        imu_topic_ = opt.system.imu_topic;
        image_topics_ = opt.system.image_topics;

        const int lidar_type = opt.localization.preprocess_lidar_type_;
        loc_ = std::make_shared<loc::Localization>(opt.localization);

        const int sensor_queue_size = opt.system.visual_enabled ? 1000 : 10;
        const int lidar_queue_size = opt.system.visual_enabled ? sensor_queue_size : 2;
        (void)lidar_queue_size;
        if (lidar_type == LidarType::LIVOX_CUSTOM)
            lidar_sub_ = nh->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                lidar_topic_, CloudQos(2), std::bind(&LocSystem::LivoxLidarCallback, this, std::placeholders::_1));
        else
            lidar_sub_ = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
                lidar_topic_, CloudQos(2), std::bind(&LocSystem::LidarCallback, this, std::placeholders::_1));
        imu_sub_ = nh->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, CloudQos(100), std::bind(&LocSystem::ImuCallback, this, std::placeholders::_1));

        bool ret = loc_->Init(nh, opt.lio, opt.lidar_loc, opt.pub);
        if (ret && loc_->NeedImageInput())
        {
            image_subs_.reserve(image_topics_.size());
            for (size_t i = 0; i < image_topics_.size(); ++i)
            {
                image_subs_.push_back(nh->create_subscription<sensor_msgs::msg::Image>(
                    image_topics_[i], ReliableQos(static_cast<size_t>(sensor_queue_size)),
                    [this, i](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
                        ImageCallback(msg, static_cast<int>(i));
                    }));
            }
        }
        if (ret)
            LOG(INFO) << get_time_us() << "[INFO][LocSystem::" << __FUNCTION__ << "] " << "online loc node has been created.";

        return ret;
    }

    void LocSystem::SetInitPose(const SE3 &pose)
    {
        LOG(INFO) << get_time_us() << "[INFO][LocSystem::" << __FUNCTION__ << "] " << "set init pose: " << pose.translation().transpose() << ", "
                  << pose.unit_quaternion().coeffs().transpose();

        loc_->SetExternalPose(pose.unit_quaternion(), pose.translation());
        loc_started_ = true;
    }

    void LocSystem::ImuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &msg)
    {
        if (loc_started_)
        {
            IMUPtr imu = std::make_shared<IMU>();

            imu->timestamp = SecFromStamp(msg->header.stamp);
            imu->linear_acceleration =
                Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
            imu->angular_velocity =
                Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

            loc_->ProcessIMUMsg(imu);
        }
    }

    void LocSystem::LidarCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
    {
        if (loc_started_)
            loc_->ProcessLidarMsg(msg);
    }

    void LocSystem::LivoxLidarCallback(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg)
    {
        if (loc_started_)
            loc_->ProcessLivoxLidarMsg(msg);
    }

    void LocSystem::ImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg, int cam_id)
    {
        if (loc_started_)
            loc_->ProcessImageMsg(msg, cam_id);
    }

} // namespace lightning