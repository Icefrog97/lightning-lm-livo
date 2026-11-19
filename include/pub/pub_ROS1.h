#pragma once

#ifndef PUB_ROS1_H
#define PUB_ROS1_H

#include "ros/ros2_types.h"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <memory>
#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "common/nav_state.h"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include "localization/localization_result.h"
#include <mutex>
namespace lightning
{
    class PubROSTopic
    {
    public:
        enum MapSaveType
        {
            DOWNSAMPLED, // 降采样地图
            DENSE        // 全量/浓密地图
        };

        struct Options
        {
            std::string pcd_path;
            std::string dense_map_name;
            std::string down_map_name;
            bool rgb_img_en = false;
            bool rgb_cloud_en = false;
            int pub_path_num = 10;
        };

        void Init(const rclcpp::Node::SharedPtr &nh, bool is_slam, const Options &opt);

        void PublishPath(double timestamp = -1.0);

        /// Localization-mode only: LIO trajectory aligned into map/Loc frame.
        void PublishLioPath(const SE3 &pose, double timestamp);
        void ClearLioPath();

        void PublishFrameWorld(CloudPtr &laserCloudWorld, double &timestamp);

        void PublishRgbFrameWorld(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &laserCloudWorld, double timestamp);

        void PublishRgbImage(const cv::Mat &image, double timestamp);

        void PublishFrameBody(CloudPtr &laserCloudIMUBody, double &timestamp);

        void PublishOdom(double &timestamp);

        void SetPosestamp(const NavState &state)
        {
            // std::lock_guard<std::mutex> lock(mtx_pub_);
            state_point_ = state;
        }

        void SaveMap(CloudPtr &global_map, MapSaveType type);
        void SaveRgbMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &global_map);

        void publishGlobalMap(const CloudPtr &mapcloud);

    private:
        void FillPose(geometry_msgs::msg::PoseStamped &out) { SetPoseInterface(out.pose); }

        void FillPose(nav_msgs::msg::Odometry &out) { SetPoseInterface(out.pose.pose); }

        void SetPoseInterface(geometry_msgs::msg::Pose &p);

        NavState state_point_;
        std::mutex mtx_pub_;

        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_full_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_rgb_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_rgb_img_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_full_body_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_aft_mapped_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_lio_path_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_aligned_lio_path_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_global_map_;

        nav_msgs::msg::Path lio_path_;          // mapping: /lio_path; localization: /localization_path
        nav_msgs::msg::Path aligned_lio_path_;  // localization: /aligned_lio_path
        nav_msgs::msg::Odometry odom_aft_mapped_;
        geometry_msgs::msg::PoseStamped msg_body_pose_;

        bool rgb_img_pub_en_ = false;
        bool rgb_cloud_pub_en_ = false;

        bool is_slam_;
        std::string pcd_path_;
        std::string dense_map_name_, down_map_name_;
        std::unique_ptr<tf2_ros::TransformBroadcaster> br_;
        double last_tf_stamp_ = -1.0;
        int pub_path_num_ = 10;
    };
}

#endif
