#include "system/slam.h"
#include "lio/laser_mapping.h"
#include <cmath>
#include <csignal>
#include "common/imu.h"
#include "common/timer.h"
#include "pub/pub_ROS1.h"
#include "loop_closing/online_loop.h"
#include <functional>
#include <boost/bind/bind.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>

namespace lightning
{
    SlamSystem::SlamSystem(lightning::SlamSystem::Options options) : options_(options)
    {
        signal(SIGINT, lightning::debug::SigHandle);
    }

    bool SlamSystem::Init(const rclcpp::Node::SharedPtr &nh, const SlamOptions &opt)
    {
        lio_ = std::make_shared<LaserMapping>(opt.lio_ctor);
        pub_ = std::make_shared<PubROSTopic>();
        lio_->SetRviz(pub_);
        if (!lio_->Init(opt.lio, nh))
        {
            LOG(ERROR) << get_time_us() << "[ERROR][SlamSystem::" << __FUNCTION__ << "] " << "failed to init lio module";
            return false;
        }

        lidar_topic_ = opt.lidar_topic;
        imu_topic_ = opt.imu_topic;
        image_topics_ = opt.image_topics;
        lidar_type_ = opt.lidar_type;

        bool is_slam_mode = true;
        pub_->Init(nh, is_slam_mode, opt.pub);

        bool with_loop = opt.with_loop_closing;
        if (with_loop)
        {
            lc_ = std::make_shared<OnlineLoop>();
            if (!lc_->Init(opt.loop, nh))
            {
                LOG(ERROR) << get_time_us() << "[ERROR][SlamSystem::" << __FUNCTION__ << "] "
                           << "failed to init online loop";
                return false;
            }
        }

        if (options_.online_mode_)
        {
            LOG(INFO) << get_time_us() << "[INFO][SlamSystem::" << __FUNCTION__ << "] " << "online mode, creating node ... ";
            LOG(INFO) << get_time_us() << "[INFO][SlamSystem::" << __FUNCTION__ << "] " << "Setting up ROS subscribers and publishers...";
            
            if (lidar_type_ == LidarType::LIVOX_CUSTOM)
                lidar_sub_ = nh->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                    lidar_topic_, CloudQos(10),
                    std::bind(&SlamSystem::LivoxLidarCallback, this, std::placeholders::_1));
            else
                lidar_sub_ = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
                    lidar_topic_, CloudQos(10),
                    std::bind(&SlamSystem::LidarCallback, this, std::placeholders::_1));
            imu_sub_ = nh->create_subscription<sensor_msgs::msg::Imu>(
                imu_topic_, CloudQos(100), std::bind(&SlamSystem::ImuCallback, this, std::placeholders::_1));
            if (lio_->NeedImageInput())
            {
                image_subs_.reserve(image_topics_.size());
                for (size_t i = 0; i < image_topics_.size(); ++i)
                {
                    image_subs_.push_back(nh->create_subscription<sensor_msgs::msg::Image>(
                        image_topics_[i], ReliableQos(100),
                        [this, i](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
                            ImageCallback(msg, static_cast<int>(i));
                        }));
                }
            }

            LOG(INFO) << get_time_us() << "[INFO][SlamSystem::" << __FUNCTION__ << "] " << "online slam node has been created. ";
        }

        return true;
    }

    void SlamSystem::StartSLAM()
    {
        running_ = true;
    }

    bool SlamSystem::VisualEnabled() const
    {
        return lio_ && lio_->VisualEnabled();
    }

    bool SlamSystem::ProcessOnce()
    {
        if (!running_ || !lio_ || !lio_->Run())
            return false;

        auto kf = lio_->GetKeyframe();
        if (kf != cur_kf_)
        {
            cur_kf_ = kf;
            if (lc_ && kf)
                lc_->AddKF(kf);
        }

        return true;
    }

    void SlamSystem::ImuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &msg)
    {
        if (running_ == false)
            return;

        IMUPtr imu = std::make_shared<IMU>();

        imu->timestamp = SecFromStamp(msg->header.stamp);
        imu->linear_acceleration =
            Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
        imu->angular_velocity =
            Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

        // mid360 的重力加速度单位是 G
        if (lidar_type_ == 4 || lidar_type_ == 5 || lidar_type_ == 3)
            imu->linear_acceleration *= G_m_s2;

        lio_->ProcessIMU(imu);
    }

    void SlamSystem::LidarCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
    {
        if (running_ == false)
            return;

        lio_->ProcessPointCloud2(msg);
        if (lio_->VisualEnabled())
            return;

        ProcessOnce();
    }

    void SlamSystem::LivoxLidarCallback(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg)
    {
        if (running_ == false)
            return;

        lio_->ProcessPointCloud2(msg);
        if (lio_->VisualEnabled())
            return;

        ProcessOnce();
    }

    void SlamSystem::ImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg, int cam_id)
    {
        if (running_ == false || !lio_->NeedImageInput())
            return;

        try
        {
            // Always convert to OpenCV BGR so debug publish (BGR8) and point coloring stay consistent.
            // Bags/cameras often publish rgb8; keeping raw encoding here swaps R/B in RViz.
            cv_bridge::CvImageConstPtr image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            lio_->ProcessImage(cam_id, image->image, SecFromStamp(msg->header.stamp));
        }
        catch (const cv_bridge::Exception &e)
        {
            LOG(ERROR) << get_time_us() << "[ERROR][SlamSystem::" << __FUNCTION__ << "] " << e.what();
        }
    }

    void SlamSystem::FinishLoop()
    {
        if (lc_)
            lc_->Finish();
    }

    void SlamSystem::SaveMap()
    {
        lio_->SaveMap();

        // yxliang 2026.2.5
        // 删除瓦片地图部分，大规模场景建图使用

        LOG(INFO) << get_time_us() << "[INFO][SlamSystem::" << __FUNCTION__ << "] " << "[slam] map saved ";
    }

}