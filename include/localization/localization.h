#pragma once

#include "common/imu.h"
#include "lio/laser_mapping.h"
#include "localization/lidar_loc.h"
#include "pub/pub_ROS1.h"
#include "localization/localization_result.h"
#include "system/async_message_process.h"
#include <atomic>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/int32.hpp>
#include <thread>

/// 预声明
namespace lightning{
    class PubROSTopic;
}

namespace lightning::loc
{

    class LidarLoc;
    class PGO;
    
    /**
     * 实时定位接口实现
     */
    class Localization
    {
    public:
        using UL = std::unique_lock<std::mutex>;

        struct Options
        {
            Options() {}

            bool online_mode_ = false; // 在线模式还是离线模式
            /// 参数
            SE3 T_body_lidar_;

            bool enable_lidar_odom_skip_ = false; // 是否允许激光里程计跳帧
            int lidar_odom_skip_num_ = 1;         // 如果允许跳帧，跳多少帧
            bool enable_lidar_loc_skip_ = true;   // 是否允许激光定位跳帧
            int lidar_loc_skip_num_ = 4;          // 如果允许跳帧，跳多少帧
            bool loc_on_kf_ = false;

            double preprocess_blind_ = 0.01;
            int preprocess_lidar_type_ = 4;
            int preprocess_point_filter_num_ = 1;
            int preprocess_scan_line_ = 1;
            double preprocess_height_max_ = 0.01;
            double preprocess_height_min_ = 0.01;
        };

        Localization(Options options = Options());
        ~Localization() = default;

        /**
         * 初始化，读配置参数
         * @param yaml_path
         * @param global_map_path
         * @param init_reloc_pose
         */
        bool Init(const rclcpp::Node::SharedPtr &nh, const lightning::LaserMapping::Config &lio_cfg,
                  const LidarLoc::Options &lidar_loc_options, const lightning::PubROSTopic::Options &pub_opt);

        /// 处理lidar消息
        void ProcessLidarMsg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr laser_msg);
        void ProcessLivoxLidarMsg(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr laser_msg);

        void ProcessImageMsg(const sensor_msgs::msg::Image::ConstSharedPtr image_msg, int cam_id);

        bool VisualEnabled() const;
        bool NeedImageInput() const;

        /// 处理IMU消息
        void ProcessIMUMsg(IMUPtr imu);

        /// 由外部设置pose，适用于手动重定位
        void SetExternalPose(const Eigen::Quaterniond &q, const Eigen::Vector3d &t);

        /// 结束，保存临时地图
        void Finish();

        /// 异步处理函数
        void LidarOdomProcCloud(CloudPtr, const bool &trigger_imu = false);
        void LidarLocProcCloud(CloudPtr cloud, double timestamp);

        struct LidarLocInput
        {
            CloudPtr cloud;
            double timestamp = 0.0; // LIO end time
        };

        /// 定位发散后，根据配置等待车辆静止再执行重置
        void HandlePendingReset();

        void Reset();

        using LocStateCallback = std::function<void(const std_msgs::msg::Int32 &state)>;
        using PointcloudBodyCallback = std::function<void(const sensor_msgs::msg::PointCloud2 &pointcloud)>;
        using PointcloudWorldCallback = std::function<void(const sensor_msgs::msg::PointCloud2 &pointcloud)>;

        // void SetPathCallback(std::function<void(const nav_msgs::msg::Path& path)>&& callback);
        // void SetPointcloudWorldCallback(std::function<void(const sensor_msgs::msg::PointCloud2& pointcloud)>&& callback);
        // void SetPointcloudBodyCallback(std::function<void(const sensor_msgs::msg::PointCloud2& pointcloud)>&& callback);
        // void SetLocStateCallback(std::function<void(const std_msgs::msg::Int32& state)>&& callback);
        // void SetHealthDiagNormalCallback(interface::health_diag_normal_callback&& callback);

    private:
        /// 模块  ========================================================================================================
        std::mutex global_mutex_; // 防止处理过程中被重复init
        Options options_;

        /// 预处理
        std::shared_ptr<PointCloudPreprocess> p_pre_ = nullptr; // point cloud preprocess

        /// 前端
        std::shared_ptr<LaserMapping> lio_ = nullptr;
        Keyframe::Ptr lio_kf_ = nullptr;

        // lidar localization
        std::shared_ptr<LidarLoc> lidar_loc_;

        std::shared_ptr<PGO> pgo_ = nullptr;

        /// TODO async 处理
        sys::AsyncMessageProcess<CloudPtr> lidar_odom_proc_cloud_; // lidar odom 处理点云
        sys::AsyncMessageProcess<LidarLocInput> lidar_loc_proc_cloud_;  // lidar loc 处理点云

        /// LIVO 的 LIO/VIO 是两步状态机，不能由雷达/图像消息数量决定 Run 次数。
        void LivoFrontendLoop();
        std::thread livo_frontend_thread_;
        std::atomic<bool> stop_livo_frontend_{true};

        /// 结果数据 =====================================================================================================
        LocalizationResult loc_result_;

        /// 框架相关
        LocStateCallback loc_state_callback_;
        PointcloudBodyCallback pointcloud_body_callback_;
        PointcloudWorldCallback pointcloud_world_callback_;

        /// 输入检查
        double last_imu_time_ = 0;
        double last_cloud_time_ = 0;

        std::deque<lightning::IMUPtr> difference_in_acceleration_;

        bool ParkingCheck(std::deque<lightning::IMUPtr> &msg);
        bool enable_debug_log_ = false;

        std::shared_ptr<lightning::PubROSTopic> pub_ = nullptr;
        int lidar_type_;
        LocalizationStatus global_status_ = LocalizationStatus::FAIL;

        Eigen::Isometry3d T_body_lio_;

        /// Align LIO odom into Loc/map frame at Loc init: T_map_lio = T_loc * T_lio^{-1}
        bool lio_path_align_set_ = false;
        SE3 T_map_from_lio_;

        /// DR 增量诊断：PGO 吃的是相邻两帧 DR 的相对位姿
        bool last_dr_dbg_set_ = false;
        SE3 last_dr_dbg_pose_;
        double last_dr_dbg_time_ = 0.0;
    };

} // namespace lightning