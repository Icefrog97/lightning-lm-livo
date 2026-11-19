#include "loop_closing/online_loop.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <boost/make_shared.hpp>
#include <filesystem>

#include "common/gtsam_newfactor.h"
#include "common/timer.h"
#include "../maps/voxelmaps_saver.hpp"

namespace lightning
{
namespace
{
Eigen::Isometry3d MakePose(const Eigen::Matrix3d &R, const Eigen::Vector3d &t)
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = R;
    T.translation() = t;
    return T;
}
}  // namespace

OnlineLoop::~OnlineLoop() { Finish(); }

bool OnlineLoop::Init(const Options &opt, const rclcpp::Node::SharedPtr &nh)
{
    config_ = opt.stdesc;
    std_manager_ = new STDesc::STDescManager(config_);
    sub_frame_num_ = std::max(1, config_.sub_frame_num_);
    skip_near_num_ = config_.skip_near_num_;

    use_height_factor_ = opt.use_height_factor;
    filter_size_map_ = opt.filter_size_map;
    const double filter_size_src = opt.filter_size_src;
    const int point_filter_num = opt.point_filter_num;
    const double update_graph_freq = opt.update_graph_freq;
    if (update_graph_freq > 1e-3)
        pgo_update_warn_sec_ = 1.0 / update_graph_freq;
    std::string pcd_path = opt.pcd_path;
    const std::string map_name = opt.loop_map_name;
    if (!pcd_path.empty() && pcd_path.back() != '/')
        pcd_path += "/";
    save_path_ = pcd_path + map_name;

    voxelmap_root_ = opt.map_path;
    if (!voxelmap_root_.empty() && voxelmap_root_.back() != '/')
        voxelmap_root_ += "/";
    bbs_min_level_res_ = opt.bbs_min_level_res;
    bbs_max_level_ = opt.bbs_max_level;

    use_enu_results_ = opt.use_enu_results;

    const std::vector<double> &T_rtk_vct = opt.T_body_wrt_lio;
    const std::vector<double> &E_rtk_vct = opt.E_body_wrt_lio;
    Eigen::Isometry3d T_body_lio = Eigen::Isometry3d::Identity();
    if (T_rtk_vct.size() >= 3)
        T_body_lio.translation() << T_rtk_vct[0], T_rtk_vct[1], T_rtk_vct[2];
    if (E_rtk_vct.size() >= 3)
    {
        const Eigen::Vector3d E(E_rtk_vct[0] * M_PI / 180.0, E_rtk_vct[1] * M_PI / 180.0,
                                E_rtk_vct[2] * M_PI / 180.0);
        T_body_lio.linear() =
            (Eigen::AngleAxisd(E(0), Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(E(1), Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(E(2), Eigen::Vector3d::UnitX()))
                .toRotationMatrix();
    }
    T_level_ = T_body_lio.inverse();

    const double odom_rot_sigma_deg = opt.odom_rot_sigma_deg;
    const double odom_trans_sigma = opt.odom_trans_sigma;
    const double odom_rot_var = std::pow(odom_rot_sigma_deg * M_PI / 180.0, 2);
    const double odom_trans_var = std::pow(odom_trans_sigma, 2);
    gtsam::Vector odom_var(6);
    odom_var << odom_rot_var, odom_rot_var, odom_rot_var, odom_trans_var,
        odom_trans_var, odom_trans_var;
    odom_noise_ = gtsam::noiseModel::Diagonal::Variances(odom_var);

    // Keep the graph gauge prior independent from the relative odometry model.
    gtsam::Vector prior_var(6);
    prior_var << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;
    prior_noise_ = gtsam::noiseModel::Diagonal::Variances(prior_var);

    gtsam::Vector loop_var(6);
    loop_var << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
    loop_noise_ = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(1),
        gtsam::noiseModel::Diagonal::Variances(loop_var));
    height_noise_ = gtsam::noiseModel::Isotropic::Sigma(1, 0.1);

    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam_ = gtsam::ISAM2(parameters);

    temp_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    pub_odom_ = nh->create_publisher<nav_msgs::msg::Odometry>("/odom_corrected", ReliableQos(10));
    pub_path_ = nh->create_publisher<nav_msgs::msg::Path>("/pgo_path", ReliableQos(10));
    path_.header.frame_id = "world";
    odom_.header.frame_id = "world";

    origin_ready_ = true;

    started_ = true;
    thread_ = std::thread(&OnlineLoop::Loop, this);
    LOG(INFO) << get_time_us() << "[INFO][OnlineLoop::" << __FUNCTION__ << "] "
              << "Loaded loop params: map/filter_size_map=" << filter_size_map_
              << ", map/filter_size_src=" << filter_size_src
              << ", point_filter_num=" << point_filter_num
              << ", sub_frame_num=" << sub_frame_num_
              << ", skip_near_num=" << skip_near_num_
              << ", use_height_factor=" << use_height_factor_
              << ", use_enu_results=" << use_enu_results_
              << ", odom_rot_sigma_deg=" << odom_rot_sigma_deg
              << ", odom_trans_sigma=" << odom_trans_sigma
              << ", save_map_path=" << save_path_
              << ", voxelmap=" << voxelmap_root_ << "VoxelMap/VoxelMap"
              << ", bbs_min_level_res=" << bbs_min_level_res_
              << ", bbs_max_level=" << bbs_max_level_;
    LOG(INFO) << get_time_us() << "[INFO][OnlineLoop::" << __FUNCTION__ << "] " << "Init Success!";
    return true;
}

void OnlineLoop::AddKF(const Keyframe::Ptr &kf)
{
    if (!started_ || !kf)
        return;

    // scan_undistort_ 是复用缓冲，下一帧会原地覆盖。拷贝必须在调用线程完成。
    PendingKF job;
    job.kf = kf;
    job.stamp = kf->GetState().timestamp_;
    job.body.reset(new pcl::PointCloud<pcl::PointXYZI>());
    const auto raw = kf->GetCloud();
    if (raw)
    {
        job.body->reserve(raw->size());
        for (const auto &pt : raw->points)
        {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
                continue;
            pcl::PointXYZI p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = pt.z;
            p.intensity = pt.intensity;
            job.body->push_back(p);
        }
        job.body->width = job.body->size();
        job.body->height = 1;
        job.body->is_dense = false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (exit_ || finished_)
            return;
        queue_.push_back(std::move(job));
    }
    cv_.notify_one();
}

bool OnlineLoop::NeedWaitOrigin() const
{
    return use_enu_results_ && !origin_ready_;
}

Eigen::Isometry3d OnlineLoop::ToEnuPose(const SE3 &T_lio) const
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = T_lio.rotationMatrix();
    T.translation() = T_lio.translation();
    T = T_level_ * T;
    if (use_enu_results_)
        T = origin_rot_ * diff_ * T;
    return T;
}

void OnlineLoop::Loop()
{
    while (true)
    {
        PendingKF job;
        bool have_job = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(10), [&] {
                return exit_ || !queue_.empty() || (origin_ready_ && !pending_origin_.empty());
            });
            if (!queue_.empty())
            {
                job = std::move(queue_.front());
                queue_.pop_front();
                have_job = true;
            }
            else if (exit_)
                break;
        }
        if (origin_ready_)
            FlushPendingOrigin();
        if (have_job)
            HandleKF(job);
        UpdateAndPublish(false);
    }
}

void OnlineLoop::FlushPendingOrigin()
{
    std::deque<PendingKF> pending;
    pending.swap(pending_origin_);
    for (const auto &job : pending)
        ProcessKF(job);
}

void OnlineLoop::HandleKF(const PendingKF &job)
{
    if (NeedWaitOrigin())
    {
        pending_origin_.push_back(job);
        return;
    }
    ProcessKF(job);
}

void OnlineLoop::ProcessKF(const PendingKF &job)
{
    const Keyframe::Ptr &kf = job.kf;
    const int id = static_cast<int>(kf->GetID());
    if (id != static_cast<int>(kfs_.size()))
    {
        LOG(ERROR) << get_time_us() << "[ERROR][OnlineLoop::" << __FUNCTION__ << "] "
                   << "keyframe id gap, expect " << kfs_.size() << " got " << id;
        return;
    }

    const Eigen::Isometry3d T_lio = ToEnuPose(kf->GetLIOPose());
    Eigen::Isometry3d T_opt = T_lio;
    if (id > 0)
    {
        const Eigen::Isometry3d T_lio_prev = MakePose(lio_R_[id - 1], lio_t_[id - 1]);
        const Eigen::Isometry3d T_opt_prev = MakePose(opt_R_[id - 1], opt_t_[id - 1]);
        T_opt = T_opt_prev * T_lio_prev.inverse() * T_lio;
    }
    if (id == 0)
    {
        LOG(INFO) << get_time_us() << "[INFO][OnlineLoop::" << __FUNCTION__ << "] "
                  << "First keyframe forced into PGO as node 0, t=" << T_lio.translation().transpose();
    }
    lio_t_.push_back(T_lio.translation());
    lio_R_.push_back(T_lio.linear());
    opt_t_.push_back(T_opt.translation());
    opt_R_.push_back(T_opt.linear());
    kfs_.push_back(kf);

    pcl::PointCloud<pcl::PointXYZI> world;
    world.reserve(job.body->size());
    for (const auto &pt : job.body->points)
    {
        pcl::PointXYZI p = pt;
        const Eigen::Vector3d pw = opt_R_.back() * Eigen::Vector3d(p.x, p.y, p.z) + opt_t_.back();
        p.x = static_cast<float>(pw.x());
        p.y = static_cast<float>(pw.y());
        p.z = static_cast<float>(pw.z());
        world.push_back(p);
    }
    body_clouds_.push_back(job.body);
    STDesc::down_sampling_voxel(world, config_.ds_size_);
    temp_cloud_->points.insert(temp_cloud_->points.end(), world.points.begin(), world.points.end());
    temp_cloud_->width = temp_cloud_->points.size();
    temp_cloud_->height = 1;
    temp_cloud_->is_dense = false;

    AddOdometryFactor(id);
    bool loop_closed = false;
    if (id != 0 && id % sub_frame_num_ == 0)
        loop_closed = TryLoop(id);
    UpdateAndPublish(loop_closed);
}

void OnlineLoop::AddOdometryFactor(int id)
{
    const gtsam::Rot3 R_opt(opt_R_[id]);
    const gtsam::Point3 t_opt(opt_t_[id]);
    const gtsam::Pose3 pose_opt(R_opt, t_opt);
    initial_.insert(id, pose_opt);
    if (id == 0)
    {
        graph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, pose_opt, prior_noise_));
        return;
    }

    const Eigen::Matrix3d R_ab = lio_R_[id - 1].transpose() * lio_R_[id];
    const Eigen::Vector3d t_ab = lio_R_[id - 1].transpose() * (lio_t_[id] - lio_t_[id - 1]);
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
        id - 1, id, gtsam::Pose3(gtsam::Rot3(R_ab), gtsam::Point3(t_ab)), odom_noise_));

    if (use_height_factor_)
    {
        graph_.add(boost::make_shared<gtsam::HeightFactor>(id, lio_t_[0].z(), height_noise_));
    }
}

bool OnlineLoop::TryLoop(int id)
{
    if (temp_cloud_->empty())
    {
        LOG(WARNING) << get_time_us() << "[WARN][OnlineLoop::" << __FUNCTION__ << "] "
                     << "temp cloud empty, skip descriptor at kf " << id;
        return false;
    }

    std::vector<STDesc::STDesc> stds_vec;
    std_manager_->GenerateSTDescs(temp_cloud_, stds_vec);

    std::pair<int, double> search_result(-1, 0);
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
    loop_transform.first.setZero();
    loop_transform.second.setIdentity();
    std::vector<std::pair<STDesc::STDesc, STDesc::STDesc>> loop_std_pair;
    if (desc_cloud_ind_ > skip_near_num_)
        std_manager_->SearchLoop(stds_vec, search_result, loop_transform, loop_std_pair);

    bool loop_closed = false;
    if (search_result.first >= 0)
    {
        loop_closed = true;
        const int curr_std = static_cast<int>(std_manager_->current_frame_id_);
        const int match_frame = search_result.first;
        LOG(INFO) << get_time_us() << "[INFO][OnlineLoop::" << __FUNCTION__ << "] "
                  << "[Loop Detection] cur_id=" << curr_std
                  << " his_id=" << match_frame
                  << " score=" << search_result.second;
        desc_cloud_ind_ = 0;
    }

    std_manager_->AddSTDescs(stds_vec);

    if (loop_closed)
    {
        const int match_frame = search_result.first;
        if (match_frame >= 0 &&
            match_frame < static_cast<int>(std_manager_->plane_cloud_vec_.size()))
        {
            std_manager_->PlaneGeomrtricIcp(std_manager_->plane_cloud_vec_.back(),
                                            std_manager_->plane_cloud_vec_[match_frame], loop_transform);
        }

        LoopEdge edge;
        bool have_edge = false;
        for (int j = 1; j <= sub_frame_num_; ++j)
        {
            const int src_frame = id + j - sub_frame_num_;
            const int tar_frame = match_frame * sub_frame_num_ + j;
            if (src_frame < 0 || tar_frame < 0 || src_frame >= static_cast<int>(opt_t_.size()) ||
                tar_frame >= static_cast<int>(opt_t_.size()))
                continue;

            const Eigen::Matrix3d src_R = loop_transform.second * opt_R_[src_frame];
            const Eigen::Vector3d src_t = loop_transform.second * opt_t_[src_frame] + loop_transform.first;
            const Eigen::Matrix3d &tar_R = opt_R_[tar_frame];
            const Eigen::Vector3d &tar_t = opt_t_[tar_frame];
            const Eigen::Vector3d t_rel = tar_R.transpose() * (src_t - tar_t);
            const Eigen::Matrix3d R_rel = tar_R.transpose() * src_R;
            graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                tar_frame, src_frame, gtsam::Pose3(gtsam::Rot3(R_rel), gtsam::Point3(t_rel)), loop_noise_));
            // 同一回环的子帧测量相同，只留当前关键帧；当前帧越界时留最后一条。
            if (!have_edge || src_frame == id)
            {
                edge.src = src_frame;
                edge.tar = tar_frame;
                edge.score = search_result.second;
                edge.t_rel = t_rel;
                edge.R_rel = R_rel;
                have_edge = true;
            }
        }
        if (have_edge)
            loop_edges_.push_back(edge);
    }

    temp_cloud_->clear();
    desc_cloud_ind_++;
    return loop_closed;
}

void OnlineLoop::UpdateAndPublish(bool loop_closed)
{
    if (graph_.empty() && initial_.empty())
        return;

    const auto t_pgo_begin = std::chrono::steady_clock::now();
    isam_.update(graph_, initial_);
    if (loop_closed)
    {
        for (int i = 0; i < 5; ++i)
            isam_.update();
    }
    graph_.resize(0);
    initial_.clear();

    const gtsam::Values result = isam_.calculateEstimate();
    const double d_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pgo_begin).count();
    if (d_time > pgo_update_warn_sec_)
    {
        LOG(WARNING) << get_time_us() << "[WARN][OnlineLoop::" << __FUNCTION__ << "] "
                     << "[Pgo Update Time " << std::setfill('0') << std::setw(5) << result.size()
                     << "]: " << std::setprecision(3) << d_time << "s";
    }
    path_.poses.clear();
    path_.header.stamp = rclcpp::Clock().now();
    // Only the published copy returns to the frontend's local "world" frame.
    // Keep optimized keyframe poses in ENU for map/voxelmap reconstruction.
    Eigen::Isometry3d T_local_from_opt = Eigen::Isometry3d::Identity();
    if (use_enu_results_)
        T_local_from_opt = (origin_rot_ * diff_).inverse();
    for (size_t i = 0; i < result.size() && i < kfs_.size(); ++i)
    {
        const gtsam::Pose3 pose = result.at<gtsam::Pose3>(i);
        const Eigen::Vector3d t = pose.translation();
        const Eigen::Matrix3d R = pose.rotation().matrix();
        if (i < opt_t_.size())
        {
            opt_t_[i] = t;
            opt_R_[i] = R;
        }
        kfs_[i]->SetOptPose(SE3(Eigen::Quaterniond(R), t));

        const Eigen::Vector3d t_pub = T_local_from_opt * t;
        const Eigen::Matrix3d R_pub = T_local_from_opt.linear() * R;
        geometry_msgs::msg::PoseStamped stamped;
        stamped.header.frame_id = "world";
        stamped.header.stamp = path_.header.stamp;
        stamped.pose.position.x = t_pub.x();
        stamped.pose.position.y = t_pub.y();
        stamped.pose.position.z = t_pub.z();
        const Eigen::Quaterniond q(R_pub);
        stamped.pose.orientation.x = q.x();
        stamped.pose.orientation.y = q.y();
        stamped.pose.orientation.z = q.z();
        stamped.pose.orientation.w = q.w();
        path_.poses.push_back(stamped);
        if (i + 1 == kfs_.size())
        {
            odom_.header.stamp = path_.header.stamp;
            odom_.pose.pose = stamped.pose;
        }
    }
    pub_path_->publish(path_);
    pub_odom_->publish(odom_);
}

void OnlineLoop::LogLoopResiduals() const
{
    if (loop_edges_.empty())
    {
        LOG(INFO) << get_time_us() << "[INFO][OnlineLoop::" << __FUNCTION__ << "] "
                  << "[Loop Residual] count=0";
        return;
    }

    double sum_trans_cm = 0.0;
    double sum_rot_deg = 0.0;
    double max_trans_cm = 0.0;
    double max_rot_deg = 0.0;
    int count = 0;
    for (const LoopEdge &edge : loop_edges_)
    {
        if (edge.src < 0 || edge.tar < 0 || edge.src >= static_cast<int>(opt_t_.size()) ||
            edge.tar >= static_cast<int>(opt_t_.size()))
            continue;

        const Eigen::Matrix3d R_opt = opt_R_[edge.tar].transpose() * opt_R_[edge.src];
        const Eigen::Vector3d t_opt =
            opt_R_[edge.tar].transpose() * (opt_t_[edge.src] - opt_t_[edge.tar]);
        const Eigen::Matrix3d R_res = edge.R_rel.transpose() * R_opt;
        const Eigen::Vector3d t_res = edge.R_rel.transpose() * (t_opt - edge.t_rel);
        const double trans_cm = t_res.norm() * 100.0;
        const double rot_deg = Eigen::AngleAxisd(R_res).angle() * 180.0 / M_PI;

        LOG(INFO) << get_time_us() << "[INFO][OnlineLoop::" << __FUNCTION__ << "] "
                  << "[Loop Residual] cur_kf=" << edge.src << " his_kf=" << edge.tar << std::fixed
                  << std::setprecision(1) << " trans_cm=" << trans_cm << std::setprecision(2)
                  << " rot_deg=" << rot_deg << std::setprecision(3) << " score=" << edge.score;

        sum_trans_cm += trans_cm;
        sum_rot_deg += rot_deg;
        max_trans_cm = std::max(max_trans_cm, trans_cm);
        max_rot_deg = std::max(max_rot_deg, rot_deg);
        ++count;
    }

    if (count == 0)
    {
        LOG(INFO) << get_time_us() << "[INFO][OnlineLoop::" << __FUNCTION__ << "] "
                  << "[Loop Residual] count=0";
        return;
    }
    LOG(INFO) << get_time_us() << "[INFO][OnlineLoop::" << __FUNCTION__ << "] "
              << "[Loop Residual] count=" << count << std::fixed << std::setprecision(1)
              << " trans_cm mean=" << (sum_trans_cm / count) << " max=" << max_trans_cm
              << std::setprecision(2) << " rot_deg mean=" << (sum_rot_deg / count)
              << " max=" << max_rot_deg;
}

void OnlineLoop::SaveMap()
{
    if (body_clouds_.empty())
        return;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    for (size_t i = 0; i < body_clouds_.size(); ++i)
    {
        const SE3 T = kfs_[i]->GetOptPose();
        const Eigen::Matrix3d R = T.rotationMatrix();
        const Eigen::Vector3d t = T.translation();
        for (const auto &pt : body_clouds_[i]->points)
        {
            const Eigen::Vector3d pw = R * Eigen::Vector3d(pt.x, pt.y, pt.z) + t;
            pcl::PointXYZI p = pt;
            p.x = static_cast<float>(pw.x());
            p.y = static_cast<float>(pw.y());
            p.z = static_cast<float>(pw.z());
            cloud->push_back(p);
        }
    }
    if (cloud->empty())
    {
        LOG(INFO) << get_time_us() << "[INFO][OnlineLoop::" << __FUNCTION__ << "] "
                  << "Map Size: 0 Input pointcloud is empty!";
        return;
    }

    const size_t size_before = cloud->size();
    LOG(INFO) << get_time_us() << "[INFO][OnlineLoop::" << __FUNCTION__ << "] " << "Finish Update Graph!";
    LOG(INFO) << get_time_us() << "[INFO][OnlineLoop::" << __FUNCTION__ << "] "
              << "size of original pgomap:  " << size_before;

    pcl::PointCloud<pcl::PointXYZI>::Ptr voxelmap_cloud(new pcl::PointCloud<pcl::PointXYZI>(*cloud));

    const float leaf = static_cast<float>(std::max(0.05, filter_size_map_));
    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.setInputCloud(cloud);
    voxel.filter(*cloud);

    LOG(INFO) << get_time_us() << "[INFO][OnlineLoop::" << __FUNCTION__ << "] "
              << "Voxel filter ratio: " << cloud->size() << " / " << size_before;

    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path pcd_file(save_path_);
    if (pcd_file.has_parent_path())
        fs::create_directories(pcd_file.parent_path(), ec);
    bool saved = false;
    bool threw = false;
    if (!ec)
    {
        try
        {
            saved = pcl::io::savePCDFileBinary(save_path_, *cloud) == 0;
        }
        catch (const pcl::IOException &ex)
        {
            threw = true;
            LOG(ERROR) << get_time_us() << "[ERROR][OnlineLoop::" << __FUNCTION__ << "] "
                       << "failed to save loop map: " << save_path_ << " " << ex.what();
        }
    }
    if (saved)
    {
        LOG(INFO) << get_time_us() << "[INFO][OnlineLoop::" << __FUNCTION__ << "] "
                  << "Save Map Successfully! Map Size: \n"
                  << cloud->size() << "\nMap Path: " << save_path_;
    }
    else if (ec)
    {
        LOG(ERROR) << get_time_us() << "[ERROR][OnlineLoop::" << __FUNCTION__ << "] "
                   << "failed to create " << pcd_file.parent_path() << ": " << ec.message();
    }
    else if (!threw)
    {
        LOG(ERROR) << get_time_us() << "[ERROR][OnlineLoop::" << __FUNCTION__ << "] "
                   << "failed to save loop map: " << save_path_;
    }

    SaveVoxelMap(voxelmap_cloud);
}

void OnlineLoop::SaveVoxelMap(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud)
{
    if (!cloud || cloud->empty() || voxelmap_root_.empty())
    {
        LOG(WARNING) << get_time_us() << "[WARN][OnlineLoop::" << __FUNCTION__ << "] "
                     << "skip 3DBBS voxelmap, empty cloud or map_path";
        return;
    }

    namespace fs = std::filesystem;
    std::error_code ec;
    std::string save_voxelmap_path = voxelmap_root_ + "VoxelMap";
    fs::create_directories(save_voxelmap_path, ec);
    if (ec)
    {
        LOG(ERROR) << get_time_us() << "[ERROR][OnlineLoop::" << __FUNCTION__ << "] "
                   << "failed to create " << save_voxelmap_path << ": " << ec.message();
        return;
    }

    VoxelMapSaver voxelmapsaver(cloud, save_voxelmap_path, bbs_min_level_res_, bbs_max_level_);
}

void OnlineLoop::Finish()
{
    if (!started_)
        return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (finished_)
            return;
        finished_ = true;
        exit_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable())
        thread_.join();
    if (!pending_origin_.empty())
    {
        LOG(WARNING) << get_time_us() << "[WARN][OnlineLoop::" << __FUNCTION__ << "] "
                     << "origin missing, flush " << pending_origin_.size()
                     << " keyframes in LIO frame";
        origin_ready_ = true;
        FlushPendingOrigin();
    }
    UpdateAndPublish(false);
    LogLoopResiduals();
    SaveMap();
    delete std_manager_;
    std_manager_ = nullptr;
}

}  // namespace lightning
