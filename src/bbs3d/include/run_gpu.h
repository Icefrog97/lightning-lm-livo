#pragma once

#include <cmath>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <memory>
#include <vector>
#include <mutex>
#include <atomic>

#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>

#include "common/eigen_types.h"
#include "common/point_def.h"

namespace lightning::gpu {

class BbsSearcher;

class BBS3DRunner {
public:
  struct Options {
    double min_level_res;
    int max_level;
    float map_leaf_size;
    float src_leaf_size;
    Eigen::Vector3d min_rpy;
    Eigen::Vector3d max_rpy;
    double score_threshold_percentage;
    int timeout_msec;
    double min_scan_range;
    double max_scan_range;
    double min_z;
    double max_z;
    bool use_gicp;

    Options() 
      : min_level_res(1.0),
        max_level(6),
        map_leaf_size(0.2f),
        src_leaf_size(0.2f),
        min_rpy(-0.02, -0.02, 0.0),
        max_rpy(0.02, 0.02, 2.0 * M_PI),
        score_threshold_percentage(0.7),
        max_scan_range(100.0),
        min_scan_range(0.0),
        min_z(-2.0),
        max_z(2.0),
        use_gicp(false),
        timeout_msec(0) {}
  };

  explicit BBS3DRunner(const Options& options = Options());
  ~BBS3DRunner();

  bool BuildTargetVoxelMap(const CloudPtr& target_cloud);
  bool BuildTargetVoxelMap(const std::string& tar_path);
  void BuildGicpTarget(const CloudPtr& target_cloud);

  bool EstimatePose(const CloudPtr& source_cloud, SE3& estimated_pose, double* score = nullptr, double* elapsed_ms = nullptr);

  bool IsReady() const { return map_ready_; }

private:
  std::vector<Eigen::Vector3f> CloudToEigenPoints(const CloudPtr& cloud, float leaf_size) const;
  CloudPtr RangeFilter(const CloudPtr& cloud, double min_range, double max_range) const;
  pcl::PointCloud<pcl::PointXYZ>::Ptr CloudToXYZ(const CloudPtr& cloud, float leaf_size) const;
  static Eigen::Matrix3d OrthogonalizeRotation(const Eigen::Matrix3d& R);

private:
  Options options_;
  std::unique_ptr<BbsSearcher> bbs3d_;
  std::atomic<bool> map_ready_{false};
  std::mutex mutex_;
  std::unique_ptr<pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>> gicp_ptr_;
};

}  // namespace lightning::gpu
