#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include "ros/ros2_types.h"
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Geometry>

#include "STDesc/STDesc.h"
#include "common/keyframe.h"

namespace lightning
{

/// 在线 STD 回环。LIO 线程只 AddKF，图优化在独立线程。
/// 因子和描述子在该线程累积，运行中不重拼 pgo 点云。
/// 位姿为 V^{-1} * T_lio。pcd_save/enu_results 仍乘原点变换；没有外部原点时该项是单位阵。
class OnlineLoop
{
public:
    struct Options
    {
        STDesc::ConfigSetting stdesc;
        bool use_height_factor = true;
        double filter_size_map = 0.2;
        double filter_size_src = 0.2;
        int point_filter_num = 5;
        double update_graph_freq = 1.0;
        std::string pcd_path;
        std::string loop_map_name = "global_loop_map.pcd";
        std::string map_path;
        double bbs_min_level_res = 1.0;
        int bbs_max_level = 6;
        bool use_enu_results = true;
        std::vector<double> T_body_wrt_lio{-1, 0, -0.1};
        std::vector<double> E_body_wrt_lio{-0.8611351114, -9.9666036744, -0.5294170198};
        double odom_rot_sigma_deg = 0.2;
        double odom_trans_sigma = 0.03;
    };

    OnlineLoop() = default;
    ~OnlineLoop();

    bool Init(const Options &opt, const rclcpp::Node::SharedPtr &nh);
    void AddKF(const Keyframe::Ptr &kf);
    /// 排空队列并写出优化后的 PCD。可重复调用。
    void Finish();

private:
    struct PendingKF
    {
        Keyframe::Ptr kf;
        pcl::PointCloud<pcl::PointXYZI>::Ptr body;
        double stamp{0.0};
    };

    void Loop();
    void HandleKF(const PendingKF &job);
    void ProcessKF(const PendingKF &job);
    void FlushPendingOrigin();
    Eigen::Isometry3d ToEnuPose(const SE3 &T_lio) const;
    void AddOdometryFactor(int id);
    bool TryLoop(int id);
    void UpdateAndPublish(bool loop_closed);
    void LogLoopResiduals() const;
    void SaveMap();
    void SaveVoxelMap(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud);
    bool NeedWaitOrigin() const;

    STDesc::ConfigSetting config_;
    STDesc::STDescManager *std_manager_{nullptr};

    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<PendingKF> queue_;
    std::deque<PendingKF> pending_origin_;
    std::atomic<bool> exit_{false};
    std::atomic<bool> origin_ready_{false};
    bool started_{false};
    bool finished_{false};

    int sub_frame_num_{10};
    int skip_near_num_{10};
    bool use_height_factor_{true};
    bool use_enu_results_{true};
    double filter_size_map_{0.2};
    double pgo_update_warn_sec_{1.0};
    double bbs_min_level_res_{1.0};
    int bbs_max_level_{6};
    std::string save_path_;
    std::string voxelmap_root_;

    Eigen::Isometry3d T_level_{Eigen::Isometry3d::Identity()};
    Eigen::Isometry3d origin_rot_{Eigen::Isometry3d::Identity()};
    Eigen::Isometry3d diff_{Eigen::Isometry3d::Identity()};

    std::vector<Keyframe::Ptr> kfs_;
    std::vector<Eigen::Vector3d> lio_t_;
    std::vector<Eigen::Matrix3d> lio_R_;
    /// PGO 系下的优化位姿。新帧用 last_opt * Δ_lio 递推，ISAM2 后再写回。
    std::vector<Eigen::Vector3d> opt_t_;
    std::vector<Eigen::Matrix3d> opt_R_;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> body_clouds_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud_;
    int desc_cloud_ind_{0};

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_;
    gtsam::ISAM2 isam_;
    gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr odom_noise_;
    gtsam::noiseModel::Base::shared_ptr loop_noise_;
    gtsam::noiseModel::Base::shared_ptr height_noise_;

    /// 每条回环只留当前关键帧那一条测量，存图前用最终位姿算残差。
    struct LoopEdge
    {
        int src{0};
        int tar{0};
        double score{0.0};
        Eigen::Vector3d t_rel{Eigen::Vector3d::Zero()};
        Eigen::Matrix3d R_rel{Eigen::Matrix3d::Identity()};
    };
    std::vector<LoopEdge> loop_edges_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
    nav_msgs::msg::Path path_;
    nav_msgs::msg::Odometry odom_;
};

}  // namespace lightning
