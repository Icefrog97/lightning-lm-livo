#pragma once

#include <algorithm>
#include <iostream>
#include <queue>
#include <chrono>
#include <Eigen/Dense>

#include <discrete_transformation/discrete_transformation.hpp>
// thrust
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace lightning::gpu {
class VoxelMaps;

struct AngularInfo {
  Eigen::Vector3i num_division;
  Eigen::Vector3f rpy_res;
  Eigen::Vector3f min_rpy;
};

struct AngularInfoGpu {
  float rpy_res_x;
  float rpy_res_y;
  float rpy_res_z;
  float min_rpy_x;
  float min_rpy_y;
  float min_rpy_z;
};

struct Point3fGpu {
  float x;
  float y;
  float z;
};

class BBS3D {
public:
  BBS3D();
  ~BBS3D();

  void set_tar_points(const std::vector<Eigen::Vector3f>& points, float min_level_res, int max_level);

  void set_src_points(const std::vector<Eigen::Vector3f>& points);

  void set_trans_search_range(const std::vector<Eigen::Vector3f>& points);

  void set_trans_search_range(const Eigen::Vector3f& min_xyz, const Eigen::Vector3f& max_xyz);

  void set_angular_search_range(const Eigen::Vector3f& min_rpy, const Eigen::Vector3f& max_rpy) {
    min_rpy_ = min_rpy;
    max_rpy_ = max_rpy;
  }

  /// 地图坐标系下传感器原点的高度带。max_z > min_z 时，每一层丢掉与该带不相交的平移节点。
  void set_z_search_range(float min_z, float max_z) {
    min_z_ = min_z;
    max_z_ = max_z;
    use_z_limit_ = max_z > min_z;
  }

  void set_voxel_expantion_rate(const float rate) {
    v_rate_ = rate;
    inv_v_rate_ = 1.0f / rate;
  }

  void set_branch_copy_size(int size) { branch_copy_size_ = size; }

  void set_score_threshold_percentage(float percentage) { score_threshold_percentage_ = percentage; }

  void enable_timeout() { use_timeout_ = true; }

  void disable_timeout() { use_timeout_ = false; }

  void set_timeout_duration_in_msec(const int msec);

  std::vector<Eigen::Vector3f> get_src_points() const { return src_points_; }

  bool set_voxelmaps_coords(const std::string& folder_path);

  std::pair<Eigen::Vector3f, Eigen::Vector3f> get_trans_search_range() const {
    return std::pair<Eigen::Vector3f, Eigen::Vector3f>{min_xyz_, max_xyz_};
  }

  std::vector<Eigen::Vector3f> get_angular_search_range() const { return std::vector<Eigen::Vector3f>{min_rpy_, max_rpy_}; }

  Eigen::Matrix4f get_global_pose() const { return global_pose_; }

  int get_best_score() const { return best_score_; }

  double get_elapsed_time() const { return elapsed_time_; }

  float get_best_score_percentage() {
    if (src_points_.size() == 0)
      return 0.0f;
    else
      return static_cast<float>(best_score_) / src_points_.size();
  };

  bool has_timed_out() { return has_timed_out_; }

  bool has_localized() { return has_localized_; }

  void localize();

private:
  void calc_angular_info(std::vector<AngularInfo>& ang_info_vec);

  std::vector<DiscreteTransformation<float>> create_init_transset(const AngularInfo& init_ang_info);

  bool z_in_range(int level, int z_index) const;

  std::vector<DiscreteTransformation<float>> calc_scores(
    const std::vector<DiscreteTransformation<float>>& h_transset,
    thrust::device_vector<AngularInfoGpu>& d_ang_info_vec);

  // pcd iof
  bool load_voxel_params(const std::string& voxelmaps_folder_path);

  std::vector<std::vector<Eigen::Vector4i>> set_multi_buckets(const std::string& voxelmaps_folder_path);

private:
  Eigen::Matrix4f global_pose_;
  bool has_timed_out_, has_localized_;
  double elapsed_time_;

  cudaStream_t stream;

  std::vector<Eigen::Vector3f> src_points_;

  thrust::device_vector<Point3fGpu> d_src_points_;

  std::unique_ptr<VoxelMaps> voxelmaps_ptr_;
  std::string voxelmaps_folder_name_;

  float v_rate_;  // voxel expansion rate
  float inv_v_rate_;

  int branch_copy_size_, best_score_;
  double score_threshold_percentage_;
  bool use_timeout_;
  std::chrono::milliseconds timeout_duration_;
  Eigen::Vector3f min_xyz_;
  Eigen::Vector3f max_xyz_;
  Eigen::Vector3f min_rpy_;
  Eigen::Vector3f max_rpy_;
  float min_z_{0.f};
  float max_z_{0.f};
  bool use_z_limit_{false};
  std::pair<int, int> init_tx_range_;
  std::pair<int, int> init_ty_range_;
  std::pair<int, int> init_tz_range_;
};
}  // namespace gpu