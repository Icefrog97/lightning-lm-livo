#include "pub/pub_ROS1.h"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>
namespace lightning
{
    void PubROSTopic::Init(const rclcpp::Node::SharedPtr &nh, bool is_slam, const Options &opt)
    {
        LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << " start";
        odom_aft_mapped_.header.frame_id = "world";
        br_ = std::make_unique<tf2_ros::TransformBroadcaster>(nh);
        pub_laser_cloud_full_body_ =
            nh->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", ReliableQos(1));
        pub_odom_aft_mapped_ = nh->create_publisher<nav_msgs::msg::Odometry>("/Odometry", ReliableQos(10));

        is_slam_ = is_slam;
        if (is_slam)
        {
            pcd_path_ = opt.pcd_path;
            dense_map_name_ = opt.dense_map_name;
            down_map_name_ = opt.down_map_name;
            rgb_img_pub_en_ = opt.rgb_img_en;
            rgb_cloud_pub_en_ = opt.rgb_cloud_en;

            lio_path_.header.frame_id = "world";
            msg_body_pose_.header.frame_id = "world";

            pub_laser_cloud_full_ =
                nh->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", ReliableQos(1));
            pub_laser_cloud_rgb_ =
                nh->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_rgb", ReliableQos(1));
            pub_rgb_img_ = nh->create_publisher<sensor_msgs::msg::Image>("/rgb_img", ReliableQos(1));
            pub_lio_path_ = nh->create_publisher<nav_msgs::msg::Path>("/lio_path", ReliableQos(1));

            pub_path_num_ = opt.pub_path_num;
        }
        else
        {
            pub_global_map_ = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/global_map", MapQos());
            lio_path_.header.frame_id = "world";
            aligned_lio_path_.header.frame_id = "world";
            pub_lio_path_ = nh->create_publisher<nav_msgs::msg::Path>("/localization_path", ReliableQos(1));
            pub_aligned_lio_path_ = nh->create_publisher<nav_msgs::msg::Path>("/aligned_lio_path", ReliableQos(1));
            pub_path_num_ = opt.pub_path_num;
        }

        LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << " end";
    }

    void PubROSTopic::PublishPath(double timestamp)
    {
        static int publish_liopath_counter = 0;
        publish_liopath_counter++;
        if (publish_liopath_counter % pub_path_num_ == 0)
        {
            FillPose(msg_body_pose_);
            msg_body_pose_.header.stamp = StampFromSec(timestamp);
            lio_path_.header.stamp = StampFromSec(timestamp);
            lio_path_.poses.push_back(msg_body_pose_);
            pub_lio_path_->publish(lio_path_);

            while (lio_path_.poses.size() > 50000)
                lio_path_.poses.erase(lio_path_.poses.begin());
        }
    }

    void PubROSTopic::PublishLioPath(const SE3 &pose, double timestamp)
    {
        if (is_slam_)
            return;

        static int publish_aligned_lio_path_counter = 0;
        publish_aligned_lio_path_counter++;
        if (publish_aligned_lio_path_counter % 2 != 0)
            return;

        geometry_msgs::msg::PoseStamped pose_stamp;
        pose_stamp.header.stamp = StampFromSec(timestamp);
        pose_stamp.header.frame_id = "world";
        const Vec3d t = pose.translation();
        const Quatd q = pose.unit_quaternion();
        pose_stamp.pose.position.x = t.x();
        pose_stamp.pose.position.y = t.y();
        pose_stamp.pose.position.z = t.z();
        pose_stamp.pose.orientation.x = q.x();
        pose_stamp.pose.orientation.y = q.y();
        pose_stamp.pose.orientation.z = q.z();
        pose_stamp.pose.orientation.w = q.w();

        aligned_lio_path_.header.stamp = pose_stamp.header.stamp;
        aligned_lio_path_.poses.push_back(pose_stamp);
        pub_aligned_lio_path_->publish(aligned_lio_path_);

        while (aligned_lio_path_.poses.size() > 50000)
            aligned_lio_path_.poses.erase(aligned_lio_path_.poses.begin());
    }

    void PubROSTopic::ClearLioPath()
    {
        aligned_lio_path_.poses.clear();
    }
    void PubROSTopic::PublishFrameWorld(CloudPtr &laserCloudWorld, double &timestamp)
    {
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = StampFromSec(timestamp);
        laserCloudmsg.header.frame_id = "world";
        pub_laser_cloud_full_->publish(laserCloudmsg);
    }

    void PubROSTopic::PublishRgbFrameWorld(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &laserCloudWorld, double timestamp)
    {
        if (!rgb_cloud_pub_en_ || !laserCloudWorld || laserCloudWorld->empty())
            return;

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = StampFromSec(timestamp);
        laserCloudmsg.header.frame_id = "world";
        pub_laser_cloud_rgb_->publish(laserCloudmsg);
    }

    void PubROSTopic::PublishRgbImage(const cv::Mat &image, double timestamp)
    {
        if (!rgb_img_pub_en_ || image.empty())
            return;

        cv_bridge::CvImage msg;
        msg.header.stamp = StampFromSec(timestamp);
        msg.encoding = image.channels() == 1 ? sensor_msgs::image_encodings::MONO8 : sensor_msgs::image_encodings::BGR8;
        msg.image = image;
        pub_rgb_img_->publish(*msg.toImageMsg());
    }

    void PubROSTopic::PublishFrameBody(CloudPtr &laserCloudIMUBody, double &timestamp)
    {
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
        laserCloudmsg.header.stamp = StampFromSec(timestamp);
        laserCloudmsg.header.frame_id = "body";
        if (pub_laser_cloud_full_body_->get_subscription_count() > 0)
            pub_laser_cloud_full_body_->publish(laserCloudmsg);
    }

    void PubROSTopic::publishGlobalMap(const CloudPtr &mapcloud)
    {
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*mapcloud, laserCloudmsg);

        laserCloudmsg.header.stamp = rclcpp::Clock().now();
        laserCloudmsg.header.frame_id = "world";
        pub_global_map_->publish(laserCloudmsg);
    }

    // IMU 高频位姿发布
    void PubROSTopic::PublishOdom(double &timestamp)
    {
        if (timestamp <= last_tf_stamp_)
        {
            return;
        }
        last_tf_stamp_ = timestamp;

        FillPose(odom_aft_mapped_);
        odom_aft_mapped_.header.stamp = StampFromSec(timestamp);

        geometry_msgs::msg::TransformStamped transform_stamped;

        transform_stamped.header.stamp = odom_aft_mapped_.header.stamp;
        transform_stamped.header.frame_id = "world";
        transform_stamped.child_frame_id = "body";
        transform_stamped.transform.translation.x = odom_aft_mapped_.pose.pose.position.x;
        transform_stamped.transform.translation.y = odom_aft_mapped_.pose.pose.position.y;
        transform_stamped.transform.translation.z = odom_aft_mapped_.pose.pose.position.z;
        transform_stamped.transform.rotation = odom_aft_mapped_.pose.pose.orientation;

        br_->sendTransform(transform_stamped);

        pub_odom_aft_mapped_->publish(odom_aft_mapped_);
    }

    void PubROSTopic::SetPoseInterface(geometry_msgs::msg::Pose &p)
    {
        std::lock_guard<std::mutex> lock(mtx_pub_);
        p.position.x = state_point_.pos_(0);
        p.position.y = state_point_.pos_(1);
        p.position.z = state_point_.pos_(2);
        auto q = state_point_.rot_.unit_quaternion();
        p.orientation.x = q.coeffs()[0];
        p.orientation.y = q.coeffs()[1];
        p.orientation.z = q.coeffs()[2];
        p.orientation.w = q.coeffs()[3];
    }

    // TODO: savemap写在full_map里
    void PubROSTopic::SaveMap(CloudPtr &global_map, MapSaveType type)
    {
        std::string save_path;
        if (type == MapSaveType::DENSE)
            save_path = pcd_path_ + dense_map_name_;
        else
            save_path = pcd_path_ + down_map_name_;

        if (!global_map || global_map->points.empty())
        {
            LOG(WARNING) << get_time_us() << "[WARN][PubROSTopic::" << __FUNCTION__ << "] " << " Global world is empty, skip saving.";
            return;
        }

        if (pcl::io::savePCDFileBinaryCompressed(save_path, *global_map) == 0)
            LOG(INFO) << get_time_us() << "[INFO][PubROSTopic::" << __FUNCTION__ << "] " << " Map successfully saved to: " << save_path;
        else
            LOG(ERROR) << get_time_us() << "[ERROR][PubROSTopic::" << __FUNCTION__ << "] " << " Failed to save world to: " << save_path;
    }

    void PubROSTopic::SaveRgbMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &global_map)
    {
        const std::string save_path = pcd_path_ + dense_map_name_;
        if (!global_map || global_map->points.empty())
        {
            LOG(WARNING) << get_time_us() << "[WARN][PubROSTopic::" << __FUNCTION__ << "] "
                         << " RGB dense map is empty, skip saving.";
            return;
        }
        if (pcl::io::savePCDFileBinaryCompressed(save_path, *global_map) == 0)
            LOG(INFO) << get_time_us() << "[INFO][PubROSTopic::" << __FUNCTION__ << "] "
                      << " RGB dense map saved to: " << save_path << " pts=" << global_map->size();
        else
            LOG(ERROR) << get_time_us() << "[ERROR][PubROSTopic::" << __FUNCTION__ << "] "
                       << " Failed to save RGB dense map to: " << save_path;
    }

}