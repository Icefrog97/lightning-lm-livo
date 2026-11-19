#include "lio/voxel_map.h"

#include "common/constant.h"
#include "lightning_math.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lightning {

int g_voxel_plane_id = 0;

// FAST StatesGroup: rot @ (0,0), pos @ (3,3). locros NavState: pos @ (0,0), rot @ (3,3).
static Mat3d FastRotVar(const VoxelMap::CovType& P) { return P.block<3, 3>(3, 3); }
static Mat3d FastPosVar(const VoxelMap::CovType& P) { return P.block<3, 3>(0, 0); }

VoxelOctoTree::VoxelOctoTree(int max_layer_in, int layer_in, int points_size_threshold_in, int max_points_num_in,
                             float planer_threshold_in)
    : max_layer(max_layer_in),
      layer(layer_in),
      points_size_threshold(points_size_threshold_in),
      max_points_num(max_points_num_in),
      planer_threshold(planer_threshold_in) {
    temp_points.clear();
    octo_state = 0;
    new_points = 0;
    update_size_threshold = 5;
    init_octo = false;
    update_enable = true;
    for (int i = 0; i < 8; i++) {
        leaves[i] = nullptr;
    }
    plane_ptr = new VoxelPlane();
}

VoxelOctoTree::~VoxelOctoTree() {
    for (int i = 0; i < 8; i++) {
        delete leaves[i];
    }
    delete plane_ptr;
}

void VoxelOctoTree::InitPlane(const std::vector<PointWithVar>& points, VoxelPlane* plane) {
    plane->plane_var = Eigen::Matrix<double, 6, 6>::Zero();
    plane->covariance = Mat3d::Zero();
    plane->center = Vec3d::Zero();
    plane->normal = Vec3d::Zero();
    plane->points_size = static_cast<int>(points.size());
    plane->radius = 0;
    for (auto pv : points) {
        plane->covariance += pv.point_w * pv.point_w.transpose();
        plane->center += pv.point_w;
    }
    plane->center = plane->center / plane->points_size;
    plane->covariance = plane->covariance / plane->points_size - plane->center * plane->center.transpose();
    Eigen::EigenSolver<Mat3d> es(plane->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Vec3d evalsReal = evals.real();
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    Mat3d J_Q;
    J_Q << 1.0 / plane->points_size, 0, 0, 0, 1.0 / plane->points_size, 0, 0, 0, 1.0 / plane->points_size;
    if (evalsReal(evalsMin) < planer_threshold) {
        for (int i = 0; i < static_cast<int>(points.size()); i++) {
            Eigen::Matrix<double, 6, 3> J;
            Mat3d F;
            for (int m = 0; m < 3; m++) {
                if (m != static_cast<int>(evalsMin)) {
                    Eigen::Matrix<double, 1, 3> F_m =
                        (points[static_cast<std::size_t>(i)].point_w - plane->center).transpose() /
                            (plane->points_size * (evalsReal[evalsMin] - evalsReal[m])) *
                        (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() +
                         evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
                    F.row(m) = F_m;
                } else {
                    Eigen::Matrix<double, 1, 3> F_m;
                    F_m << 0, 0, 0;
                    F.row(m) = F_m;
                }
            }
            J.block<3, 3>(0, 0) = evecs.real() * F;
            J.block<3, 3>(3, 0) = J_Q;
            plane->plane_var += J * points[static_cast<std::size_t>(i)].var * J.transpose();
        }

        plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
        plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
        plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
        plane->min_eigen_value = static_cast<float>(evalsReal(evalsMin));
        plane->mid_eigen_value = static_cast<float>(evalsReal(evalsMid));
        plane->max_eigen_value = static_cast<float>(evalsReal(evalsMax));
        plane->radius = static_cast<float>(std::sqrt(evalsReal(evalsMax)));
        plane->d = static_cast<float>(-(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) +
                                        plane->normal(2) * plane->center(2)));
        plane->is_plane = true;
        plane->is_update = true;
        if (!plane->is_init) {
            plane->id = g_voxel_plane_id;
            g_voxel_plane_id++;
            plane->is_init = true;
        }
    } else {
        plane->is_update = true;
        plane->is_plane = false;
    }
}

void VoxelOctoTree::InitOctoTree() {
    if (temp_points.size() > static_cast<std::size_t>(points_size_threshold)) {
        InitPlane(temp_points, plane_ptr);
        if (plane_ptr->is_plane == true) {
            octo_state = 0;
            if (temp_points.size() > static_cast<std::size_t>(max_points_num)) {
                update_enable = false;
                std::vector<PointWithVar>().swap(temp_points);
                new_points = 0;
            }
        } else {
            octo_state = 1;
            CutOctoTree();
        }
        init_octo = true;
        new_points = 0;
    }
}

void VoxelOctoTree::CutOctoTree() {
    if (layer >= max_layer) {
        octo_state = 0;
        return;
    }
    for (size_t i = 0; i < temp_points.size(); i++) {
        int xyz[3] = {0, 0, 0};
        if (temp_points[i].point_w[0] > voxel_center[0]) {
            xyz[0] = 1;
        }
        if (temp_points[i].point_w[1] > voxel_center[1]) {
            xyz[1] = 1;
        }
        if (temp_points[i].point_w[2] > voxel_center[2]) {
            xyz[2] = 1;
        }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves[leafnum] == nullptr) {
            leaves[leafnum] =
                new VoxelOctoTree(max_layer, layer + 1, layer_init_num[layer + 1], max_points_num, planer_threshold);
            leaves[leafnum]->layer_init_num = layer_init_num;
            leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
            leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
            leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
            leaves[leafnum]->quater_length = quater_length / 2;
        }
        leaves[leafnum]->temp_points.push_back(temp_points[i]);
        leaves[leafnum]->new_points++;
    }
    for (unsigned i = 0; i < 8; i++) {
        if (leaves[i] != nullptr) {
            if (leaves[i]->temp_points.size() > static_cast<std::size_t>(leaves[i]->points_size_threshold)) {
                InitPlane(leaves[i]->temp_points, leaves[i]->plane_ptr);
                if (leaves[i]->plane_ptr->is_plane) {
                    leaves[i]->octo_state = 0;
                    if (leaves[i]->temp_points.size() > static_cast<std::size_t>(leaves[i]->max_points_num)) {
                        leaves[i]->update_enable = false;
                        std::vector<PointWithVar>().swap(leaves[i]->temp_points);
                        new_points = 0;
                    }
                } else {
                    leaves[i]->octo_state = 1;
                    leaves[i]->CutOctoTree();
                }
                leaves[i]->init_octo = true;
                leaves[i]->new_points = 0;
            }
        }
    }
}

void VoxelOctoTree::UpdateOctoTree(const PointWithVar& pv) {
    if (!init_octo) {
        new_points++;
        temp_points.push_back(pv);
        if (temp_points.size() > static_cast<std::size_t>(points_size_threshold)) {
            InitOctoTree();
        }
    } else {
        if (plane_ptr->is_plane) {
            if (update_enable) {
                new_points++;
                temp_points.push_back(pv);
                if (new_points > update_size_threshold) {
                    InitPlane(temp_points, plane_ptr);
                    new_points = 0;
                }
                if (temp_points.size() >= static_cast<std::size_t>(max_points_num)) {
                    update_enable = false;
                    std::vector<PointWithVar>().swap(temp_points);
                    new_points = 0;
                }
            }
        } else {
            if (layer < max_layer) {
                int xyz[3] = {0, 0, 0};
                if (pv.point_w[0] > voxel_center[0]) {
                    xyz[0] = 1;
                }
                if (pv.point_w[1] > voxel_center[1]) {
                    xyz[1] = 1;
                }
                if (pv.point_w[2] > voxel_center[2]) {
                    xyz[2] = 1;
                }
                int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
                if (leaves[leafnum] != nullptr) {
                    leaves[leafnum]->UpdateOctoTree(pv);
                } else {
                    leaves[leafnum] = new VoxelOctoTree(max_layer, layer + 1, layer_init_num[layer + 1], max_points_num,
                                                        planer_threshold);
                    leaves[leafnum]->layer_init_num = layer_init_num;
                    leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
                    leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
                    leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
                    leaves[leafnum]->quater_length = quater_length / 2;
                    leaves[leafnum]->UpdateOctoTree(pv);
                }
            } else {
                if (update_enable) {
                    new_points++;
                    temp_points.push_back(pv);
                    if (new_points > update_size_threshold) {
                        InitPlane(temp_points, plane_ptr);
                        new_points = 0;
                    }
                    if (temp_points.size() > static_cast<std::size_t>(max_points_num)) {
                        update_enable = false;
                        std::vector<PointWithVar>().swap(temp_points);
                        new_points = 0;
                    }
                }
            }
        }
    }
}

VoxelOctoTree* VoxelOctoTree::FindCorrespond(const Vec3d& pw) {
    if (!init_octo || plane_ptr->is_plane || (layer >= max_layer)) {
        return this;
    }
    int xyz[3] = {0, 0, 0};
    xyz[0] = pw[0] > voxel_center[0] ? 1 : 0;
    xyz[1] = pw[1] > voxel_center[1] ? 1 : 0;
    xyz[2] = pw[2] > voxel_center[2] ? 1 : 0;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    return (leaves[leafnum] != nullptr) ? leaves[leafnum]->FindCorrespond(pw) : this;
}

VoxelMap::VoxelMap(const VoxelMapConfig& config) : config_(config) {
    if (config_.layer_init_num.empty()) {
        config_.layer_init_num = {5, 5, 5, 5, 5};
    }
}

VoxelMap::~VoxelMap() { Clear(); }

void VoxelMap::SetExtrinsic(const Mat3d& R_imu_lidar, const Vec3d& t_imu_lidar) {
    ext_R_ = R_imu_lidar;
    ext_t_ = t_imu_lidar;
}

void VoxelMap::Clear() {
    for (auto& kv : voxel_map_) {
        delete kv.second;
    }
    voxel_map_.clear();
    lru_list_.clear();
    lru_iters_.clear();
    lru_evicted_ = 0;
    pv_list_.clear();
    ptpl_list_.clear();
}

void VoxelMap::Touch(const VoxelLocation& loc) {
    auto it = lru_iters_.find(loc);
    if (it == lru_iters_.end()) {
        return;
    }
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
    it->second = lru_list_.begin();
}

void VoxelMap::EvictIfNeeded() {
    if (config_.lru_capacity == 0) {
        return;
    }
    while (lru_list_.size() > config_.lru_capacity) {
        const VoxelLocation back = lru_list_.back();
        auto map_it = voxel_map_.find(back);
        if (map_it != voxel_map_.end()) {
            delete map_it->second;
            voxel_map_.erase(map_it);
        }
        lru_iters_.erase(back);
        lru_list_.pop_back();
        ++lru_evicted_;
    }
}

VoxelOctoTree* VoxelMap::InsertRoot(const VoxelLocation& position, VoxelOctoTree* octo_tree) {
    voxel_map_[position] = octo_tree;
    lru_list_.push_front(position);
    lru_iters_[position] = lru_list_.begin();
    return octo_tree;
}

VoxelLocation VoxelMap::LocationFor(const Vec3d& point_w) const {
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
        loc_xyz[j] = static_cast<float>(point_w[j] / config_.max_voxel_size);
        if (loc_xyz[j] < 0) {
            loc_xyz[j] -= 1.0f;
        }
    }
    return VoxelLocation(static_cast<int64_t>(loc_xyz[0]), static_cast<int64_t>(loc_xyz[1]),
                         static_cast<int64_t>(loc_xyz[2]));
}

void VoxelMap::CalcBodyCov(Vec3d& pb, Mat3d& cov) const {
    if (pb[2] == 0) {
        pb[2] = 0.0001;
    }
    const float range = static_cast<float>(std::sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]));
    const float range_var = static_cast<float>(config_.dept_err * config_.dept_err);
    Eigen::Matrix2d direction_var;
    const double s = std::sin(config_.beam_err * constant::kDEG2RAD);
    direction_var << s * s, 0, 0, s * s;
    Vec3d direction(pb);
    direction.normalize();
    Mat3d direction_hat;
    direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
    Vec3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
    base_vector1.normalize();
    Vec3d base_vector2 = base_vector1.cross(direction);
    base_vector2.normalize();
    Eigen::Matrix<double, 3, 2> N;
    N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
    Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
    cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
}

void VoxelMap::BuildPvList(const CloudPtr& body, const NavState& state, const CovType& P) {
    // FAST StateEstimation: body_cov + R_wi * body (no extR) + [p_imu]_x * P_rot + P_pos
    const Mat3d R_wl = (state.rot_ * state.offset_R_lidar_).matrix();
    const Vec3d t_wl = state.rot_ * state.offset_t_lidar_ + state.pos_;
    const Mat3d rot_var = FastRotVar(P);
    const Mat3d t_var = FastPosVar(P);
    pv_list_.resize(body->size());
    for (size_t i = 0; i < body->size(); i++) {
        PointWithVar& pv = pv_list_[i];
        pv = PointWithVar();
        Vec3d point_this(body->points[i].x, body->points[i].y, body->points[i].z);
        if (point_this[2] == 0) {
            point_this[2] = 0.001;
        }
        Mat3d var;
        CalcBodyCov(point_this, var);
        pv.body_var = var;
        pv.point_b = point_this;
        point_this = ext_R_ * point_this + ext_t_;
        pv.point_i = point_this;
        pv.point_crossmat = math::SKEW_SYM_MATRIX(point_this);
        pv.point_w = R_wl * pv.point_b + t_wl;
        pv.var = state.rot_.matrix() * pv.body_var * state.rot_.matrix().transpose() +
                 (-pv.point_crossmat) * rot_var * (-pv.point_crossmat.transpose()) + t_var;
    }
}

void VoxelMap::RefreshWorldAndVar(const NavState& state, const CovType& P) {
    // FAST handleLIO after StateEstimation
    const Mat3d R_wl = (state.rot_ * state.offset_R_lidar_).matrix();
    const Vec3d t_wl = state.rot_ * state.offset_t_lidar_ + state.pos_;
    const Mat3d rot_var = FastRotVar(P);
    const Mat3d t_var = FastPosVar(P);
    for (auto& pv : pv_list_) {
        pv.point_w = R_wl * pv.point_b + t_wl;
        Mat3d var = pv.body_var;
        // FAST handleLIO after StateEstimation：Rwi*Rli * body_var，再交给 processFrame
        var = (state.rot_.matrix() * ext_R_) * var * (state.rot_.matrix() * ext_R_).transpose() +
              (-pv.point_crossmat) * rot_var * (-pv.point_crossmat.transpose()) + t_var;
        pv.var = var;
    }
}

void VoxelMap::BuildVoxelMap(const CloudPtr& body, const CloudPtr& world, const NavState& state, const CovType& P) {
    Clear();
    const float voxel_size = static_cast<float>(config_.max_voxel_size);
    const float planer_threshold = static_cast<float>(config_.planner_threshold);
    const int max_layer = config_.max_layer;
    const int max_points_num = config_.max_points_num;
    std::vector<int> layer_init_num = config_.layer_init_num;
    const Mat3d rot_var = FastRotVar(P);
    const Mat3d t_var = FastPosVar(P);

    std::vector<PointWithVar> input_points;
    const size_t n = std::min(body->size(), world->size());
    for (size_t i = 0; i < n; i++) {
        PointWithVar pv;
        pv.point_w << world->points[i].x, world->points[i].y, world->points[i].z;
        Vec3d point_this(body->points[i].x, body->points[i].y, body->points[i].z);
        Mat3d var;
        CalcBodyCov(point_this, var);
        pv.body_var = var;
        pv.point_b = point_this;
        Mat3d point_crossmat = math::SKEW_SYM_MATRIX(point_this);
        var = (state.rot_.matrix() * ext_R_) * var * (state.rot_.matrix() * ext_R_).transpose() +
              (-point_crossmat) * rot_var * (-point_crossmat.transpose()) + t_var;
        pv.var = var;
        input_points.push_back(pv);
    }

    const unsigned plsize = static_cast<unsigned>(input_points.size());
    for (unsigned i = 0; i < plsize; i++) {
        const PointWithVar p_v = input_points[i];
        const VoxelLocation position = LocationFor(p_v.point_w);
        auto iter = voxel_map_.find(position);
        if (iter != voxel_map_.end()) {
            iter->second->temp_points.push_back(p_v);
            iter->second->new_points++;
            Touch(position);
        } else {
            VoxelOctoTree* octo_tree =
                new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
            octo_tree->quater_length = voxel_size / 4;
            octo_tree->voxel_center[0] = (0.5 + position.x) * voxel_size;
            octo_tree->voxel_center[1] = (0.5 + position.y) * voxel_size;
            octo_tree->voxel_center[2] = (0.5 + position.z) * voxel_size;
            octo_tree->temp_points.push_back(p_v);
            octo_tree->new_points++;
            octo_tree->layer_init_num = layer_init_num;
            InsertRoot(position, octo_tree);
        }
    }
    EvictIfNeeded();
    for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); ++iter) {
        iter->second->InitOctoTree();
    }
    pv_list_ = std::move(input_points);
}

void VoxelMap::UpdateVoxelMap(const std::vector<PointWithVar>& input_points) {
    const float voxel_size = static_cast<float>(config_.max_voxel_size);
    const float planer_threshold = static_cast<float>(config_.planner_threshold);
    const int max_layer = config_.max_layer;
    const int max_points_num = config_.max_points_num;
    std::vector<int> layer_init_num = config_.layer_init_num;
    const unsigned plsize = static_cast<unsigned>(input_points.size());
    for (unsigned i = 0; i < plsize; i++) {
        const PointWithVar p_v = input_points[i];
        const VoxelLocation position = LocationFor(p_v.point_w);
        auto iter = voxel_map_.find(position);
        if (iter != voxel_map_.end()) {
            iter->second->UpdateOctoTree(p_v);
            Touch(position);
        } else {
            VoxelOctoTree* octo_tree =
                new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
            octo_tree->layer_init_num = layer_init_num;
            octo_tree->quater_length = voxel_size / 4;
            octo_tree->voxel_center[0] = (0.5 + position.x) * voxel_size;
            octo_tree->voxel_center[1] = (0.5 + position.y) * voxel_size;
            octo_tree->voxel_center[2] = (0.5 + position.z) * voxel_size;
            octo_tree->UpdateOctoTree(p_v);
            InsertRoot(position, octo_tree);
        }
    }
    EvictIfNeeded();
}

void VoxelMap::BuildSingleResidual(PointWithVar& pv, const VoxelOctoTree* current_octo, int current_layer,
                                   bool& is_sucess, double& prob, PointToPlane& single_ptpl) {
    const int max_layer = config_.max_layer;
    const double sigma_num = config_.sigma_num;
    const double radius_k = 3;
    Vec3d p_w = pv.point_w;
    if (current_octo->plane_ptr->is_plane) {
        VoxelPlane& plane = *current_octo->plane_ptr;
        const float dis_to_plane = std::fabs(plane.normal(0) * p_w(0) + plane.normal(1) * p_w(1) +
                                             plane.normal(2) * p_w(2) + plane.d);
        const float dis_to_center = (plane.center(0) - p_w(0)) * (plane.center(0) - p_w(0)) +
                                    (plane.center(1) - p_w(1)) * (plane.center(1) - p_w(1)) +
                                    (plane.center(2) - p_w(2)) * (plane.center(2) - p_w(2));
        const float range_dis = std::sqrt(dis_to_center - dis_to_plane * dis_to_plane);

        if (range_dis <= radius_k * plane.radius) {
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = p_w - plane.center;
            J_nq.block<1, 3>(0, 3) = -plane.normal;
            double sigma_l = J_nq * plane.plane_var * J_nq.transpose();
            sigma_l += plane.normal.transpose() * pv.var * plane.normal;
            if (dis_to_plane < sigma_num * std::sqrt(sigma_l)) {
                is_sucess = true;
                const double this_prob =
                    1.0 / (std::sqrt(sigma_l)) * std::exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
                if (this_prob > prob) {
                    prob = this_prob;
                    pv.normal = plane.normal;
                    single_ptpl.body_cov = pv.body_var;
                    single_ptpl.point_b = pv.point_b;
                    single_ptpl.point_w = pv.point_w;
                    single_ptpl.plane_var = plane.plane_var;
                    single_ptpl.normal = plane.normal;
                    single_ptpl.center = plane.center;
                    single_ptpl.d = plane.d;
                    single_ptpl.layer = current_layer;
                    single_ptpl.dis_to_plane =
                        static_cast<float>(plane.normal(0) * p_w(0) + plane.normal(1) * p_w(1) +
                                           plane.normal(2) * p_w(2) + plane.d);
                }
                return;
            } else {
                return;
            }
        } else {
            return;
        }
    } else {
        if (current_layer < max_layer) {
            for (size_t leafnum = 0; leafnum < 8; leafnum++) {
                if (current_octo->leaves[leafnum] != nullptr) {
                    VoxelOctoTree* leaf_octo = current_octo->leaves[leafnum];
                    BuildSingleResidual(pv, leaf_octo, current_layer + 1, is_sucess, prob, single_ptpl);
                }
            }
            return;
        } else {
            return;
        }
    }
}

void VoxelMap::BuildResidualListOMP() {
    const double voxel_size = config_.max_voxel_size;
    std::mutex mylock;
    ptpl_list_.clear();
    std::vector<PointToPlane> all_ptpl_list(pv_list_.size());
    std::vector<bool> useful_ptpl(pv_list_.size());
    std::vector<VoxelLocation> touched;
    touched.reserve(pv_list_.size() * 2);
    std::vector<size_t> index(pv_list_.size());
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
        useful_ptpl[i] = false;
    }
#ifdef _OPENMP
    omp_set_num_threads(std::max(1, config_.omp_num));
#pragma omp parallel for
#endif
    for (int i = 0; i < static_cast<int>(index.size()); i++) {
        PointWithVar& pv = pv_list_[static_cast<std::size_t>(i)];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = static_cast<float>(pv.point_w[j] / voxel_size);
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0f;
            }
        }
        VoxelLocation position(static_cast<int64_t>(loc_xyz[0]), static_cast<int64_t>(loc_xyz[1]),
                               static_cast<int64_t>(loc_xyz[2]));
        auto iter = voxel_map_.find(position);
        if (iter != voxel_map_.end()) {
            VoxelOctoTree* current_octo = iter->second;
            PointToPlane single_ptpl;
            bool is_sucess = false;
            double prob = 0;
            VoxelLocation used_loc = position;
            BuildSingleResidual(pv, current_octo, 0, is_sucess, prob, single_ptpl);
            if (!is_sucess) {
                VoxelLocation near_position = position;
                if (loc_xyz[0] > (current_octo->voxel_center[0] + current_octo->quater_length)) {
                    near_position.x = near_position.x + 1;
                } else if (loc_xyz[0] < (current_octo->voxel_center[0] - current_octo->quater_length)) {
                    near_position.x = near_position.x - 1;
                }
                if (loc_xyz[1] > (current_octo->voxel_center[1] + current_octo->quater_length)) {
                    near_position.y = near_position.y + 1;
                } else if (loc_xyz[1] < (current_octo->voxel_center[1] - current_octo->quater_length)) {
                    near_position.y = near_position.y - 1;
                }
                if (loc_xyz[2] > (current_octo->voxel_center[2] + current_octo->quater_length)) {
                    near_position.z = near_position.z + 1;
                } else if (loc_xyz[2] < (current_octo->voxel_center[2] - current_octo->quater_length)) {
                    near_position.z = near_position.z - 1;
                }
                auto iter_near = voxel_map_.find(near_position);
                if (iter_near != voxel_map_.end()) {
                    BuildSingleResidual(pv, iter_near->second, 0, is_sucess, prob, single_ptpl);
                    if (is_sucess) {
                        used_loc = near_position;
                    }
                }
            }
            if (is_sucess) {
                mylock.lock();
                useful_ptpl[static_cast<std::size_t>(i)] = true;
                all_ptpl_list[static_cast<std::size_t>(i)] = single_ptpl;
                touched.push_back(used_loc);
                mylock.unlock();
            } else {
                mylock.lock();
                useful_ptpl[static_cast<std::size_t>(i)] = false;
                mylock.unlock();
            }
        }
    }
    for (size_t i = 0; i < useful_ptpl.size(); i++) {
        if (useful_ptpl[i]) {
            ptpl_list_.push_back(all_ptpl_list[i]);
        }
    }
    for (const VoxelLocation& loc : touched) {
        Touch(loc);
    }
}

}  // namespace lightning
