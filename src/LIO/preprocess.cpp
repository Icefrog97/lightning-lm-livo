#include "lio/preprocess.h"
#include <algorithm>
#include <cmath>
#include <glog/logging.h>
#include "common/timer.h"

namespace lightning
{

    void PointCloudPreprocess::Process(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg,
                                       PointCloudType::Ptr &pcl_out)
    {
        switch (lidar_type_)
        {
        case mid360:
            Mid360Handler(msg);
            break;

        case OUST64:
            Oust64Handler(msg);
            break;

        case VELO16:
            VeclodyneHandler(msg);
            break;

        case ROBOSENSE:
            RoboSenseHandler(msg);
            break;

        default:
            LOG(ERROR) << get_time_us() << "[ERROR][PointCloudPreprocess::" << __FUNCTION__ << "] " << "Error LiDAR Type";
            break;
        }
        *pcl_out = pl_surf_;
    }

    void PointCloudPreprocess::Process(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg,
                                       PointCloudType::Ptr &pcl_out)
    {
        LivoxCustomHandler(msg);
        *pcl_out = pl_surf_;
    }

    // 若点云处理不过来，可根据 lidar_slam-lm 改为多线程预处理点云
    void PointCloudPreprocess::Mid360Handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
    {
        pl_surf_.clear();
        pl_full_.clear();

        pcl::PointCloud<mid360_ros::Point> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);

        int plsize = pl_orig.points.size();

        pl_surf_.reserve(plsize);
        pl_full_.resize(plsize);

        const uint64_t head_time_ns = static_cast<uint64_t>(msg->header.stamp.sec) * 1000000000ULL +
                                      msg->header.stamp.nanosec;
        uint valid_num = 0;

        for (uint i = 1; i < plsize; ++i)
        {
            if ((pl_orig.points[i].line < N_SCANS_) &&
                ((pl_orig.points[i].tag & 0x30) == 0x10 || (pl_orig.points[i].tag & 0x30) == 0x00))
            {
                valid_num++;

                if (pl_orig.points[i].z < height_min_ || pl_orig.points[i].z > height_max_)
                    continue;

                if (valid_num % point_filter_num_ == 0)
                {
                    pl_full_[i].x = pl_orig.points[i].x;
                    pl_full_[i].y = pl_orig.points[i].y;
                    pl_full_[i].z = pl_orig.points[i].z;
                    pl_full_[i].intensity = pl_orig.points[i].intensity;

                    pl_full_[i].time = (double(pl_orig.points[i].timestamp) - double(head_time_ns)) / 1e6;

                    if (((abs(pl_full_[i].x - pl_full_[i - 1].x) > 1e-7) ||
                         (abs(pl_full_[i].y - pl_full_[i - 1].y) > 1e-7) ||
                         (abs(pl_full_[i].z - pl_full_[i - 1].z) > 1e-7)) &&
                        (pl_full_[i].x * pl_full_[i].x + pl_full_[i].y * pl_full_[i].y +
                             pl_full_[i].z * pl_full_[i].z >
                         (blind_ * blind_)))
                    {
                        pl_surf_.push_back(pl_full_[i]);
                    }
                }
            }
        }
        pl_surf_.width = pl_surf_.size();
        pl_surf_.height = 1;
        pl_surf_.is_dense = false;
    }

    void PointCloudPreprocess::LivoxCustomHandler(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg)
    {
        pl_surf_.clear();
        pl_full_.clear();

        const uint plsize = std::min<uint>(msg->point_num, msg->points.size());

        pl_surf_.reserve(plsize);
        pl_full_.resize(plsize);

        uint valid_num = 0;

        for (uint i = 1; i < plsize; ++i)
        {
            if ((msg->points[i].line < N_SCANS_) &&
                ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
            {
                valid_num++;

                if (msg->points[i].z < height_min_ || msg->points[i].z > height_max_)
                    continue;

                if (valid_num % point_filter_num_ == 0)
                {
                    pl_full_[i].x = msg->points[i].x;
                    pl_full_[i].y = msg->points[i].y;
                    pl_full_[i].z = msg->points[i].z;
                    pl_full_[i].intensity = msg->points[i].reflectivity;
                    pl_full_[i].time = msg->points[i].offset_time / double(1000000); // ms

                    if (((abs(pl_full_[i].x - pl_full_[i - 1].x) > 1e-7) ||
                         (abs(pl_full_[i].y - pl_full_[i - 1].y) > 1e-7) ||
                         (abs(pl_full_[i].z - pl_full_[i - 1].z) > 1e-7)) &&
                        (pl_full_[i].x * pl_full_[i].x + pl_full_[i].y * pl_full_[i].y +
                             pl_full_[i].z * pl_full_[i].z >
                         (blind_ * blind_)))
                    {
                        pl_surf_.push_back(pl_full_[i]);
                    }
                }
            }
        }

        pl_surf_.width = pl_surf_.size();
        pl_surf_.height = 1;
        pl_surf_.is_dense = false;
    }

    void PointCloudPreprocess::RoboSenseHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
        pl_surf_.clear();
        pl_full_.clear();

        pcl::PointCloud<PointRobotSense> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);

        int plsize = pl_orig.size();
        pl_surf_.reserve(plsize);

        double head_time = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9;

        /// RoboSense的时间戳是double, 均为linux时间且单位为秒，这里减去header time并乘以1000得到毫秒为单位的时间戳

        size_t invalid_num = 0;
        for (int i = 0; i < pl_orig.points.size(); i++) {
            if (i % point_filter_num_ != 0) {
                continue;
            }

            if (!std::isfinite(pl_orig.points[i].x) || !std::isfinite(pl_orig.points[i].y) ||
                !std::isfinite(pl_orig.points[i].z) || !std::isfinite(pl_orig.points[i].intensity) ||
                !std::isfinite(pl_orig.points[i].timestamp)) {
                invalid_num++;
                continue;
            }

            double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                        pl_orig.points[i].z * pl_orig.points[i].z;

            if (range < (blind_ * blind_)) {
                continue;
            }

            if (pl_orig.points[i].z < height_min_ || pl_orig.points[i].z > height_max_) {
                continue;
            }

            PointType added_pt;
            added_pt.x = pl_orig.points[i].x;
            added_pt.y = pl_orig.points[i].y;
            added_pt.z = pl_orig.points[i].z;
            added_pt.intensity = pl_orig.points[i].intensity;

            added_pt.time = (pl_orig.points[i].timestamp - head_time) * 1e3;  //  / 1e6;  // curvature unit: ms

            pl_surf_.points.push_back(added_pt);
        }

        if (invalid_num > 0) {
            LOG_EVERY_N(WARNING, 50) << get_time_us() << "[WARN][PointCloudPreprocess::" << __FUNCTION__ << "] "
                                     << "drop invalid robosense points: " << invalid_num;
        }

        pl_surf_.width = pl_surf_.size();
        pl_surf_.height = 1;
        pl_surf_.is_dense = false;
    }

    void PointCloudPreprocess::Oust64Handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
    {
        pl_surf_.clear();
        pl_full_.clear();
        pcl::PointCloud<ouster_ros::Point> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);
        int plsize = pl_orig.size();
        pl_surf_.reserve(plsize);

        for (int i = 0; i < pl_orig.points.size(); i++)
        {
            if (i % point_filter_num_ != 0)
            {
                continue;
            }

            double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                           pl_orig.points[i].z * pl_orig.points[i].z;

            if (range < (blind_ * blind_))
            {
                continue;
            }

            PointType added_pt;
            added_pt.x = pl_orig.points[i].x;
            added_pt.y = pl_orig.points[i].y;
            added_pt.z = pl_orig.points[i].z;
            added_pt.intensity = pl_orig.points[i].intensity;
            added_pt.time = pl_orig.points[i].t * 1e3; // curvature unit: ms
            pl_surf_.points.push_back(added_pt);
        }

        pl_surf_.width = pl_surf_.size();
        pl_surf_.height = 1;
        pl_surf_.is_dense = false;
    }

    void PointCloudPreprocess::VeclodyneHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
    {
        pl_surf_.clear();
        pl_full_.clear();

        pcl::PointCloud<velodyne_ros::Point> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);
        int plsize = pl_orig.points.size();
        pl_surf_.reserve(plsize);

        /*** These variables only works when no point timestamps given ***/
        double omega_l = 3.61; // scan angular velocity
        std::vector<bool> is_first(N_SCANS_, true);
        std::vector<double> yaw_fp(N_SCANS_, 0.0);   // yaw of first scan point
        std::vector<float> yaw_last(N_SCANS_, 0.0);  // yaw of last scan point
        std::vector<float> time_last(N_SCANS_, 0.0); // last offset time
        /*****************************************************************/

        if (pl_orig.points[plsize - 1].time > 0)
        {
            given_offset_time_ = true;
        }
        else
        {
            given_offset_time_ = false;
            double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
            double yaw_end = yaw_first;
            int layer_first = pl_orig.points[0].ring;
            for (uint i = plsize - 1; i > 0; i--)
            {
                if (pl_orig.points[i].ring == layer_first)
                {
                    yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
                    break;
                }
            }
        }

        for (int i = 0; i < plsize; i++)
        {
            PointType added_pt;

            added_pt.x = pl_orig.points[i].x;
            added_pt.y = pl_orig.points[i].y;
            added_pt.z = pl_orig.points[i].z;
            added_pt.intensity = pl_orig.points[i].intensity;
            added_pt.time = pl_orig.points[i].time * 1e-3; // curvature unit: ms

            if (!given_offset_time_)
            {
                int layer = pl_orig.points[i].ring;
                double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

                if (is_first[layer])
                {
                    yaw_fp[layer] = yaw_angle;
                    is_first[layer] = false;
                    added_pt.time = 0.0;
                    yaw_last[layer] = yaw_angle;
                    time_last[layer] = added_pt.time;
                    continue;
                }

                // compute offset time
                if (yaw_angle <= yaw_fp[layer])
                {
                    added_pt.time = (yaw_fp[layer] - yaw_angle) / omega_l;
                }
                else
                {
                    added_pt.time = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
                }

                if (added_pt.time < time_last[layer])
                {
                    added_pt.time += 360.0 / omega_l;
                }

                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.time;
            }

            if (i % point_filter_num_ == 0)
            {
                if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind_ * blind_))
                {
                    pl_surf_.points.push_back(added_pt);
                }
            }
        }

        pl_surf_.width = pl_surf_.size();
        pl_surf_.height = 1;
        pl_surf_.is_dense = false;
    }

}