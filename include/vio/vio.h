/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VIO_H_
#define VIO_H_

#include "lio/voxel_map.h"
#include "vio/fast_livo_types.h"
#include "vio/feature.h"
#include "vio/frame.h"
#include "vio/visual_point.h"
#include <Eigen/Dense>
#include <fstream>
#include <list>
#include <memory>
#include <opencv2/imgproc/imgproc_c.h>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>
#include <vikit/pinhole_camera.h>

using lightning::Feature;
using lightning::Frame;
using lightning::FramePtr;
using lightning::VisualPoint;
using lightning::VoxelOctoTree;
using lightning::VoxelPlane;

struct SubSparseMap
{
  std::vector<float> propa_errors;
  std::vector<float> errors;
  std::vector<std::vector<float>> warp_patch;
  std::vector<int> search_levels;
  std::vector<VisualPoint *> voxel_points;
  std::vector<double> inv_expo_list;
  std::vector<pointWithVar> add_from_voxel_map;

  SubSparseMap()
  {
    propa_errors.reserve(SIZE_LARGE);
    errors.reserve(SIZE_LARGE);
    warp_patch.reserve(SIZE_LARGE);
    search_levels.reserve(SIZE_LARGE);
    voxel_points.reserve(SIZE_LARGE);
    inv_expo_list.reserve(SIZE_LARGE);
    add_from_voxel_map.reserve(SIZE_SMALL);
  };

  void reset()
  {
    propa_errors.clear();
    errors.clear();
    warp_patch.clear();
    search_levels.clear();
    voxel_points.clear();
    inv_expo_list.clear();
    add_from_voxel_map.clear();
  }
};

class Warp
{
public:
  Eigen::Matrix2d A_cur_ref;
  int search_level;
  Warp(int level, Eigen::Matrix2d warp_matrix) : search_level(level), A_cur_ref(warp_matrix) {}
  ~Warp() {}
};

class VOXEL_POINTS
{
public:
  std::vector<VisualPoint *> voxel_points;
  int count;
  VOXEL_POINTS(int num) : count(num) {}
  ~VOXEL_POINTS() 
  { 
    for (VisualPoint* vp : voxel_points) 
    {
      if (vp != nullptr) { delete vp; vp = nullptr; }
    }
  }
};

class VIOManager
{
public:
  int grid_size;
  vk::AbstractCamera *cam;
  vk::PinholeCamera *pinhole_cam;
  StatesGroup *state;
  StatesGroup *state_propagat;
  lightning::NavState *nav_state = nullptr;
  lightning::NavState *nav_state_propagat = nullptr;
  Eigen::Matrix<double, lightning::NavState::dim, lightning::NavState::dim> *nav_P = nullptr;
  M3D Rli, Rci, Rcl, Rcw, Jdphi_dR, Jdp_dt, Jdp_dR;
  V3D Pli, Pci, Pcl, Pcw;
  std::vector<int> grid_num;
  std::vector<int> map_index;
  std::vector<int> border_flag;
  std::vector<int> update_flag;
  std::vector<float> map_dist;
  std::vector<float> scan_value;
  std::vector<float> patch_buffer;
  bool normal_en, inverse_composition_en, exposure_estimate_en, raycast_en, has_ref_patch_cache;
  bool ncc_en = false, colmap_output_en = false;

  int width, height, grid_n_width, grid_n_height, length;
  double image_resize_factor;
  double fx, fy, cx, cy;
  int patch_pyrimid_level, patch_size, patch_size_total, patch_size_half, border, warp_len;
  int max_iterations, total_points;

  double img_point_cov, outlier_threshold, ncc_thre;
  
  SubSparseMap *visual_submap;
  std::vector<std::vector<V3D>> rays_with_sample_points;

  double compute_jacobian_time, update_ekf_time;
  double ave_total = 0;
  // double ave_build_residual_time = 0;
  // double ave_ekf_time = 0;

  int frame_count = 0;
  bool plot_flag;

  Eigen::Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
  Eigen::Matrix<double, lightning::NavState::dim, lightning::NavState::dim> G_nav;
  Eigen::MatrixXd K, H_sub_inv;

  std::ofstream fout_camera, fout_colmap;
  std::unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map;
  std::list<VOXEL_LOCATION> feat_lru_list_;
  std::unordered_map<VOXEL_LOCATION, typename std::list<VOXEL_LOCATION>::iterator> feat_lru_iters_;
  std::unordered_set<VOXEL_LOCATION> feat_pinned_;
  /// Visual voxel LRU capacity. 0 = unlimited (no eviction).
  std::size_t feat_lru_capacity_ = 100000;
  std::size_t feat_lru_evicted_ = 0;
  /// Max VisualPoints stored in one 0.5 m feat_map voxel. 0 = unlimited.
  std::size_t feat_max_points_per_voxel_ = 10;
  std::size_t feat_points_evicted_ = 0;
  /// Points retrieved this frame; FIFO will not delete them until the frame ends.
  std::unordered_set<VisualPoint *> feat_frame_points_;
  std::unordered_map<VOXEL_LOCATION, int> sub_feat_map;
  std::unordered_map<int, Warp *> warp_map;
  std::vector<VisualPoint *> retrieve_voxel_points;
  std::vector<pointWithVar> append_voxel_points;
  FramePtr new_frame_;
  cv::Mat img_cp, img_rgb, img_test;

  enum CellType
  {
    TYPE_MAP = 1,
    TYPE_POINTCLOUD,
    TYPE_UNKNOWN
  };

  VIOManager();
  ~VIOManager();
  void resetFeatMap();
  void setFeatLruCapacity(std::size_t capacity) { feat_lru_capacity_ = capacity; }
  void setFeatMaxPointsPerVoxel(std::size_t capacity) { feat_max_points_per_voxel_ = capacity; }
  std::size_t FeatMapSize() const { return feat_map.size(); }
  std::size_t FeatLruEvicted() const { return feat_lru_evicted_; }
  std::size_t FeatPointsEvicted() const { return feat_points_evicted_; }
  void updateStateInverse(cv::Mat img, int level);
  void updateState(cv::Mat img, int level);
  void processFrame(cv::Mat &img, std::vector<pointWithVar> &pg, const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time);
  void retrieveFromVisualSparseMap(cv::Mat img, std::vector<pointWithVar> &pg, const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void generateVisualMapPoints(cv::Mat img, std::vector<pointWithVar> &pg);
  void setImuToLidarExtrinsic(const V3D &transl, const M3D &rot);
  void setLidarToCameraExtrinsic(std::vector<double> &R, std::vector<double> &P);
  void initializeVIO();
  void getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level);
  void computeProjectionJacobian(V3D p, MD(2, 3) & J);
  void computeJacobianAndUpdateEKF(cv::Mat img);
  void SyncStateFromNav();
  bool ApplyNavVisualUpdate(const Eigen::MatrixXd &H_sub, const Eigen::VectorXd &z, int h_cols);
  void resetGrid();
  void updateVisualMapPoints(cv::Mat img);
  void getWarpMatrixAffine(const vk::AbstractCamera &cam, const Eigen::Vector2d &px_ref, const Eigen::Vector3d &f_ref, const double depth_ref, const SE3 &T_cur_ref,
                           const int level_ref, 
                           const int pyramid_level, const int halfpatch_size, Eigen::Matrix2d &A_cur_ref);
  void getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref,
                                     const V3D &xyz_ref, const V3D &normal_ref, const SE3 &T_cur_ref, const int level_ref, Eigen::Matrix2d &A_cur_ref);
  void warpAffine(const Eigen::Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Eigen::Vector2d &px_ref, const int level_ref, const int search_level,
                  const int pyramid_level, const int halfpatch_size, float *patch);
  void insertPointIntoVoxelMap(VisualPoint *pt_new);
  void plotTrackedPoints();
  void updateFrameState(StatesGroup state);
  void projectPatchFromRefToCur(const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void updateReferencePatch(const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void precomputeReferencePatches(int level);
  void dumpDataForColmap();
  double calculateNCC(float *ref_patch, float *cur_patch, int patch_size);
  int getBestSearchLevel(const Eigen::Matrix2d &A_cur_ref, const int max_level);
  V3F getInterpolatedPixel(cv::Mat img, V2D pc);

 private:
  VOXEL_LOCATION VisualVoxelIndex(const V3D &pt_w) const;
  void TouchFeatVoxel(const VOXEL_LOCATION &position);
  void PinFeatVoxel(const VOXEL_LOCATION &position);
  void TrimFeatVoxelPoints(VOXEL_POINTS *voxel);
  void EvictFeatIfNeeded();
  void PinVisualSubmapVoxels();
  void ClearFeatPins();
  
  // void resetRvizDisplay();
  // deque<VisualPoint *> map_cur_frame;
  // deque<VisualPoint *> sub_map_ray;
  // deque<VisualPoint *> sub_map_ray_fov;
  // deque<VisualPoint *> visual_sub_map_cur;
  // deque<VisualPoint *> visual_converged_point;
  // std::vector<std::vector<V3D>> sample_points;

  // PointCloudXYZI::Ptr pg_down;
  // pcl::VoxelGrid<PointType> downSizeFilter;
};
typedef std::shared_ptr<VIOManager> VIOManagerPtr;

#endif // VIO_H_