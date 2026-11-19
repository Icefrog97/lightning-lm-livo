#include "run_cpu.h"
#include "cpu_bbs3d/bbs3d.hpp"

#include <pcl/filters/approximate_voxel_grid.h>
#include <glog/logging.h>
#include <chrono>

#include "common/timer.h"

namespace lightning::cpu {

BBS3DRunner::BBS3DRunner(const Options &options)
    : options_(options), bbs3d_(std::make_unique<BBS3D>()), map_ready_(false) {}

BBS3DRunner::~BBS3DRunner() = default;

Eigen::Matrix3d BBS3DRunner::OrthogonalizeRotation(const Eigen::Matrix3d &R) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R_ortho = U * V.transpose();
    if (R_ortho.determinant() < 0.0) {
        U.col(2) *= -1.0;
        R_ortho = U * V.transpose();
    }
    return R_ortho;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr BBS3DRunner::CloudToXYZ(const CloudPtr &cloud, float leaf_size) const {
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz(new pcl::PointCloud<pcl::PointXYZ>());
    if (!cloud || cloud->empty())
        return xyz;

    CloudPtr used_cloud(new PointCloudType);
    if (leaf_size > 0.0f) {
        pcl::ApproximateVoxelGrid<PointType> voxel;
        voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
        voxel.setInputCloud(cloud);
        voxel.filter(*used_cloud);
    } else {
        *used_cloud = *cloud;
    }

    xyz->points.resize(used_cloud->size());
    for (size_t i = 0; i < used_cloud->size(); ++i) {
        xyz->points[i].x = used_cloud->points[i].x;
        xyz->points[i].y = used_cloud->points[i].y;
        xyz->points[i].z = used_cloud->points[i].z;
    }
    xyz->width = static_cast<uint32_t>(xyz->size());
    xyz->height = 1;
    xyz->is_dense = false;
    return xyz;
}

std::vector<Eigen::Vector3d> BBS3DRunner::CloudToEigenPoints(const CloudPtr &cloud, float leaf_size) const {
    if (!cloud || cloud->empty())
        return {};

    CloudPtr used_cloud(new PointCloudType);
    if (leaf_size > 0.0f) {
        pcl::ApproximateVoxelGrid<PointType> voxel;
        voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
        voxel.setInputCloud(cloud);
        voxel.filter(*used_cloud);
    } else {
        *used_cloud = *cloud;
    }

    std::vector<Eigen::Vector3d> points;
    points.reserve(used_cloud->size());
    for (const auto &pt : used_cloud->points)
        points.emplace_back(pt.x, pt.y, pt.z);
    return points;
}

CloudPtr BBS3DRunner::RangeFilter(const CloudPtr &cloud, double min_range, double max_range) const {
    CloudPtr filtered(new PointCloudType);
    if (!cloud || cloud->empty())
        return filtered;
    filtered->points.reserve(cloud->points.size());
    for (const auto &pt : cloud->points) {
        const double nx = static_cast<double>(pt.x);
        const double ny = static_cast<double>(pt.y);
        const double nz = static_cast<double>(pt.z);
        const double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (norm >= min_range && norm <= max_range)
            filtered->points.push_back(pt);
    }
    filtered->width = static_cast<uint32_t>(filtered->points.size());
    filtered->height = 1;
    filtered->is_dense = cloud->is_dense;
    return filtered;
}

void BBS3DRunner::ApplySearchConfig() {
    bbs3d_->set_angular_search_range(options_.min_rpy, options_.max_rpy);
    bbs3d_->set_score_threshold_percentage(options_.score_threshold_percentage);
    if (options_.timeout_msec > 0) {
        bbs3d_->enable_timeout();
        bbs3d_->set_timeout_duration_in_msec(options_.timeout_msec);
    } else {
        bbs3d_->disable_timeout();
    }
}

bool BBS3DRunner::BuildTargetVoxelMap(const CloudPtr &target_cloud) {
    const auto target_points = CloudToEigenPoints(target_cloud, options_.map_leaf_size);
    if (target_points.empty()) {
        map_ready_ = false;
        return false;
    }

    map_ready_ = false;
    std::lock_guard<std::mutex> lock(mutex_);
    bbs3d_->set_tar_points(target_points, options_.min_level_res, options_.max_level);
    bbs3d_->set_trans_search_range(target_points);
    bbs3d_->set_z_search_range(options_.min_z, options_.max_z);
    ApplySearchConfig();

    if (options_.use_gicp) {
        auto tar_xyz = CloudToXYZ(target_cloud, options_.map_leaf_size);
        gicp_ptr_ = std::make_unique<pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>>();
        gicp_ptr_->setInputTarget(tar_xyz);
        LOG(INFO) << lightning::get_time_us() << "[INFO][cpu::BBS3DRunner::" << __FUNCTION__ << "] "
                  << "GICP target set, points: " << tar_xyz->size();
    }
    map_ready_ = true;
    return true;
}

bool BBS3DRunner::BuildTargetVoxelMap(const std::string &tar_path) {
    map_ready_ = false;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!bbs3d_->set_voxelmaps_coords(tar_path)) {
        LOG(ERROR) << lightning::get_time_us() << "[ERROR][cpu::BBS3DRunner::" << __FUNCTION__ << "] "
                   << "Building 3D-BBS target voxel map failed, from path: " << tar_path;
        return false;
    }
    bbs3d_->set_z_search_range(options_.min_z, options_.max_z);
    ApplySearchConfig();
    map_ready_ = true;
    return true;
}

void BBS3DRunner::BuildGicpTarget(const CloudPtr &target_cloud) {
    if (!options_.use_gicp || !target_cloud || target_cloud->empty())
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto tar_xyz = CloudToXYZ(target_cloud, options_.map_leaf_size);
    gicp_ptr_ = std::make_unique<pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>>();
    gicp_ptr_->setInputTarget(tar_xyz);
    LOG(INFO) << lightning::get_time_us() << "[INFO][cpu::BBS3DRunner::" << __FUNCTION__ << "] "
              << "GICP target set, points: " << tar_xyz->size();
}

bool BBS3DRunner::EstimatePose(const CloudPtr &source_cloud, SE3 &estimated_pose, double *score, double *elapsed_ms) {
    if (!map_ready_)
        return false;

    std::lock_guard<std::mutex> lock(mutex_);
    const auto filtered_cloud = RangeFilter(source_cloud, options_.min_scan_range, options_.max_scan_range);
    const auto source_points = CloudToEigenPoints(filtered_cloud, options_.src_leaf_size);
    if (source_points.empty())
        return false;

    bbs3d_->set_src_points(source_points);
    bbs3d_->localize();
    const double bbs_time = bbs3d_->get_elapsed_time();
    if (score != nullptr)
        *score = bbs3d_->get_best_score_percentage();
    if (!bbs3d_->has_localized()) {
        if (elapsed_ms != nullptr)
            *elapsed_ms = bbs_time;
        return false;
    }

    Eigen::Matrix4d bbs_global_pose = bbs3d_->get_global_pose();
    if (options_.use_gicp && gicp_ptr_) {
        auto src_xyz = CloudToXYZ(filtered_cloud, options_.src_leaf_size);
        gicp_ptr_->setInputSource(src_xyz);
        pcl::PointCloud<pcl::PointXYZ> gicp_output;
        const auto gicp_t1 = std::chrono::steady_clock::now();
        gicp_ptr_->align(gicp_output, bbs_global_pose.cast<float>());
        const auto gicp_t2 = std::chrono::steady_clock::now();
        const double gicp_time_ms =
            std::chrono::duration_cast<std::chrono::nanoseconds>(gicp_t2 - gicp_t1).count() / 1e6;
        if (gicp_ptr_->hasConverged()) {
            const Eigen::Matrix4d gicp_T = gicp_ptr_->final_transformation_.cast<double>();
            const Eigen::Matrix3d R_ortho = OrthogonalizeRotation(gicp_T.block<3, 3>(0, 0));
            const Eigen::Vector3d t = gicp_T.block<3, 1>(0, 3);
            estimated_pose = SE3(Quatd(R_ortho), t);
            if (elapsed_ms != nullptr)
                *elapsed_ms = bbs_time + gicp_time_ms;
            return true;
        }
        LOG(WARNING) << lightning::get_time_us() << "[WARN][cpu::BBS3DRunner::" << __FUNCTION__ << "] "
                     << "GICP did not converge, fallback to bbs pose, gicp_time(ms): " << gicp_time_ms;
    }

    const Eigen::Matrix3d rot = OrthogonalizeRotation(bbs_global_pose.block<3, 3>(0, 0));
    const Eigen::Vector3d trans = bbs_global_pose.block<3, 1>(0, 3);
    estimated_pose = SE3(Quatd(rot), trans);
    if (elapsed_ms != nullptr)
        *elapsed_ms = bbs_time;
    return true;
}

}  // namespace lightning::cpu
