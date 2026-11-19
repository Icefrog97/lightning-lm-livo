#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "common/eigen_types.h"
#include "common/nav_state.h"
#include "common/point_def.h"
#include "vio/point_with_var.h"
#include "vio/voxel_location.h"

namespace lightning {

struct VoxelMapConfig {
    double max_voxel_size = 0.5;
    int max_layer = 2;
    int max_points_num = 50;
    std::vector<int> layer_init_num{5, 5, 5, 5, 5};
    double planner_threshold = 0.0025;
    double beam_err = 0.05;
    double dept_err = 0.02;
    double sigma_num = 3.0;
    int omp_num = 2;
    /// Root voxel LRU capacity. 0 = unlimited (no eviction).
    std::size_t lru_capacity = 100000;
};

struct PointToPlane {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3d point_b = Vec3d::Zero();
    Vec3d point_w = Vec3d::Zero();
    Vec3d normal = Vec3d::Zero();
    Vec3d center = Vec3d::Zero();
    Eigen::Matrix<double, 6, 6> plane_var = Eigen::Matrix<double, 6, 6>::Zero();
    Mat3d body_cov = Mat3d::Zero();
    int layer = 0;
    double d = 0.0;
    double eigen_value = 0.0;
    bool is_valid = false;
    float dis_to_plane = 0.0;
};

struct VoxelPlane {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3d center = Vec3d::Zero();
    Vec3d normal = Vec3d::Zero();
    Vec3d y_normal = Vec3d::Zero();
    Vec3d x_normal = Vec3d::Zero();
    Mat3d covariance = Mat3d::Zero();
    Eigen::Matrix<double, 6, 6> plane_var = Eigen::Matrix<double, 6, 6>::Zero();
    float radius = 0.0f;
    float min_eigen_value = 1.0f;
    float mid_eigen_value = 1.0f;
    float max_eigen_value = 1.0f;
    float d = 0.0f;
    int points_size = 0;
    bool is_plane = false;
    bool is_init = false;
    int id = 0;
    bool is_update = false;
};

class VoxelOctoTree {
   public:
    std::vector<PointWithVar> temp_points;
    VoxelPlane* plane_ptr = nullptr;
    int layer = 0;
    int octo_state = 0;
    VoxelOctoTree* leaves[8] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    double voxel_center[3] = {0.0, 0.0, 0.0};
    std::vector<int> layer_init_num;
    float quater_length = 0.0f;
    float planer_threshold = 0.01f;
    int points_size_threshold = 5;
    int update_size_threshold = 5;
    int max_points_num = 50;
    int max_layer = 2;
    int new_points = 0;
    bool init_octo = false;
    bool update_enable = true;

    VoxelOctoTree(int max_layer_in, int layer_in, int points_size_threshold_in, int max_points_num_in,
                  float planer_threshold_in);
    ~VoxelOctoTree();

    void InitPlane(const std::vector<PointWithVar>& points, VoxelPlane* plane);
    void InitOctoTree();
    void CutOctoTree();
    void UpdateOctoTree(const PointWithVar& pv);
    VoxelOctoTree* FindCorrespond(const Vec3d& pw);
    VoxelOctoTree* find_correspond(const Vec3d& pw) { return FindCorrespond(pw); }
};

class VoxelMap {
   public:
    using CovType = Eigen::Matrix<double, NavState::dim, NavState::dim>;

    explicit VoxelMap(const VoxelMapConfig& config);
    ~VoxelMap();

    void SetExtrinsic(const Mat3d& R_imu_lidar, const Vec3d& t_imu_lidar);
    void Clear();

    void BuildVoxelMap(const CloudPtr& body, const CloudPtr& world, const NavState& state, const CovType& P);
    void UpdateVoxelMap(const std::vector<PointWithVar>& input_points);
    void BuildPvList(const CloudPtr& body, const NavState& state, const CovType& P);
    void BuildResidualListOMP();
    void RefreshWorldAndVar(const NavState& state, const CovType& P);

    const std::vector<PointWithVar>& pv_list() const { return pv_list_; }
    std::vector<PointWithVar>& pv_list() { return pv_list_; }
    const std::vector<PointToPlane>& ptpl_list() const { return ptpl_list_; }
    const std::unordered_map<VoxelLocation, VoxelOctoTree*>& voxel_map() const { return voxel_map_; }
    std::size_t NumRootVoxels() const { return voxel_map_.size(); }
    std::size_t LruEvicted() const { return lru_evicted_; }
    int effect_feat_num() const { return static_cast<int>(ptpl_list_.size()); }

   private:
    VoxelLocation LocationFor(const Vec3d& point_w) const;
    void CalcBodyCov(Vec3d& pb, Mat3d& cov) const;
    void BuildSingleResidual(PointWithVar& pv, const VoxelOctoTree* current_octo, int current_layer, bool& is_sucess,
                             double& prob, PointToPlane& single_ptpl);
    void Touch(const VoxelLocation& loc);
    void EvictIfNeeded();
    VoxelOctoTree* InsertRoot(const VoxelLocation& position, VoxelOctoTree* octo_tree);

    VoxelMapConfig config_;
    Mat3d ext_R_ = Mat3d::Identity();
    Vec3d ext_t_ = Vec3d::Zero();
    std::unordered_map<VoxelLocation, VoxelOctoTree*> voxel_map_;
    std::list<VoxelLocation> lru_list_;
    std::unordered_map<VoxelLocation, typename std::list<VoxelLocation>::iterator> lru_iters_;
    std::size_t lru_evicted_ = 0;
    std::vector<PointWithVar> pv_list_;
    std::vector<PointToPlane> ptpl_list_;
};

}  // namespace lightning
