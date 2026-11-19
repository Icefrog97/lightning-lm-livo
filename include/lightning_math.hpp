//
// Created by xiang on 25-3-12.
// Refactored for ROS 1 by Gemini
//

#ifndef LIGHTNING_MATH_HPP
#define LIGHTNING_MATH_HPP

#pragma once

#include <glog/logging.h>
#include <pcl/filters/voxel_grid.h>
#include <boost/array.hpp>
#include <boost/math/tools/precision.hpp>
#include <cmath>
#include <numeric>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include "ros/ros2_types.h"  

#include "common/eigen_types.h"
#include "common/options.h"
#include "common/point_def.h"
#include "common/pose_rpy.h"
#include "common/timer.h"

namespace lightning::math {

template <typename T>
inline Eigen::Matrix<T, 3, 3> SKEW_SYM_MATRIX(const Eigen::Matrix<T, 3, 1>& v) {
    Eigen::Matrix<T, 3, 3> m;
    m << 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0;
    return m;
}

template <typename T>
inline Eigen::Matrix<T, 3, 3> SKEW_SYM_MATRIX(const T& v1, const T& v2, const T& v3) {
    Eigen::Matrix<T, 3, 3> m;
    m << 0.0, -v3, v2, v3, 0.0, -v1, -v2, v1, 0.0;
    return m;
}

template <typename T>
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1>&& ang) {
    T ang_norm = ang.norm();
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
    if (ang_norm > 0.0000001) {
        Eigen::Matrix<T, 3, 1> r_axis = ang / ang_norm;
        Eigen::Matrix<T, 3, 3> K;
        K = SKEW_SYM_MATRIX(r_axis);

        return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
    } else {
        return Eye3;
    }
}

template <typename T, typename Ts>
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1>& ang_vel, const Ts& dt) {
    T ang_vel_norm = ang_vel.norm();
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();

    if (ang_vel_norm > 0.0000001) {
        Eigen::Matrix<T, 3, 1> r_axis = ang_vel / ang_vel_norm;
        Eigen::Matrix<T, 3, 3> K;

        K = SKEW_SYM_MATRIX(r_axis);

        T r_ang = ang_vel_norm * dt;

        /// Roderigous Tranformation
        return Eye3 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
    } else {
        return Eye3;
    }
}

template <typename T>
Eigen::Matrix<T, 3, 3> Exp(const T& v1, const T& v2, const T& v3) {
    T&& norm = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
    if (norm > 0.00001) {
        Eigen::Matrix<T, 3, 3> K;
        K = SKEW_SYM_MATRIX(v1 / norm, v2 / norm, v3 / norm);

        /// Roderigous Transformation
        return Eye3 + std::sin(norm) * K + (1.0 - std::cos(norm)) * K * K;
    } else {
        return Eye3;
    }
}

/* Logrithm of a Rotation Matrix */
template <typename T>
Eigen::Matrix<T, 3, 1> Log(const Eigen::Matrix<T, 3, 3>& R) {
    T theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
    Eigen::Matrix<T, 3, 1> K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
    return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

template <typename T>
Eigen::Matrix<T, 3, 1> RotMtoEuler(const Eigen::Matrix<T, 3, 3>& rot) {
    T sy = sqrt(rot(0, 0) * rot(0, 0) + rot(1, 0) * rot(1, 0));
    bool singular = sy < 1e-6;
    T x, y, z;
    if (!singular) {
        x = atan2(rot(2, 1), rot(2, 2));
        y = atan2(-rot(2, 0), sy);
        z = atan2(rot(1, 0), rot(0, 0));
    } else {
        x = atan2(-rot(1, 2), rot(1, 1));
        y = atan2(-rot(2, 0), sy);
        z = 0;
    }
    Eigen::Matrix<T, 3, 1> ang(x, y, z);
    return ang;
}

/// xt: 经验证，此转换与ros中tf::createQuaternionFromRPY结果一致
template <typename T>
Eigen::Matrix<T, 3, 3> RpyToRotM2(const T r, const T p, const T y) {
    using AA = Eigen::AngleAxis<T>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    return Eigen::Matrix<T, 3, 3>(AA(y, Vec3::UnitZ()) * AA(p, Vec3::UnitY()) * AA(r, Vec3::UnitX()));
}

template <typename S>
inline Eigen::Matrix<S, 3, 1> VecFromArray(const std::vector<double>& v) {
    return Eigen::Matrix<S, 3, 1>(v[0], v[1], v[2]);
}

template <typename S>
inline Eigen::Matrix<S, 3, 1> VecFromArray(const boost::array<S, 3>& v) {
    return Eigen::Matrix<S, 3, 1>(v[0], v[1], v[2]);
}

template <typename S>
inline Eigen::Matrix<S, 3, 3> MatFromArray(const std::vector<double>& v) {
    Eigen::Matrix<S, 3, 3> m;
    m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
    return m;
}

template <typename S>
inline Eigen::Matrix<S, 3, 3> MatFromArray(const boost::array<S, 9>& v) {
    Eigen::Matrix<S, 3, 3> m;
    m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
    return m;
}

template <typename T>
T rad2deg(const T& radians) {
    return radians * 180.0 / M_PI;
}

template <typename T>
T deg2rad(const T& degrees) {
    return degrees * M_PI / 180.0;
}

template <typename T, typename T2>
void limit_in_range(T&& num, T2&& min_limit, T2&& max_limit) {
    if (num < min_limit) {
        num = min_limit;
    }
    if (num >= max_limit) {
        num = max_limit;
    }
}

template <typename C, typename D, typename Getter>
inline void ComputeMeanAndCovDiag(const C& data, D& mean, D& cov_diag, Getter&& getter) {
    size_t len = data.size();
    if (len == 1) {
        mean = getter(data[0]);
        cov_diag = D::Zero().eval();
    } else {
        mean = std::accumulate(data.begin(), data.end(), D::Zero().eval(),
                               [&getter](const D& sum, const auto& data) -> D { return sum + getter(data); }) / len;
        cov_diag = std::accumulate(data.begin(), data.end(), D::Zero().eval(),
                                   [&mean, &getter](const D& sum, const auto& data) -> D {
                                     return sum + (getter(data) - mean).cwiseAbs2();
                                   }) / (len);
    }
}

inline void HistoryMeanAndVar(size_t hist_n, float hist_mean, float hist_var2, size_t curr_n, float curr_mean,
                              float curr_var2, float& new_mean, float& new_var2) {
    new_mean = (hist_n * hist_mean + curr_n * curr_mean) / (hist_n + curr_n);
    new_var2 = (hist_n * (hist_var2 + (new_mean - hist_mean) * (new_mean - hist_mean)) +
                curr_n * (curr_var2 + (new_mean - curr_mean) * (new_mean - curr_mean))) /
               (hist_n + curr_n);
}

template <class scalar>
inline std::pair<scalar, scalar> cos_sinc_sqrt(const scalar& x2) {
    using std::cos;
    using std::sin;
    using std::sqrt;
    static scalar const taylor_0_bound = boost::math::tools::epsilon<scalar>();
    static scalar const taylor_2_bound = sqrt(taylor_0_bound);
    static scalar const taylor_n_bound = sqrt(taylor_2_bound);

    assert(x2 >= 0 && "argument must be non-negative");

    if (x2 >= taylor_n_bound) {
        scalar x = sqrt(x2);
        return std::make_pair(cos(x), sin(x) / x);
    }

    static scalar const inv[] = {1 / 3., 1 / 4., 1 / 5., 1 / 6., 1 / 7., 1 / 8., 1 / 9.};
    scalar cosi = 1., sinc = 1;
    scalar term = -1 / 2. * x2;
    for (int i = 0; i < 3; ++i) {
        cosi += term;
        term *= inv[2 * i];
        sinc += term;
        term *= -inv[2 * i + 1] * x2;
    }

    return std::make_pair(cosi, sinc);
}

inline SO3 exp(const Vec3d& vec, const double& scale = 1) {
    double norm2 = vec.squaredNorm();
    std::pair<double, double> cos_sinc = cos_sinc_sqrt(scale * scale * norm2);
    double mult = cos_sinc.second * scale;
    Vec3d result = mult * vec;
    return SO3(Quatd(cos_sinc.first, result[0], result[1], result[2]));
}

inline Eigen::Matrix<double, 2, 3> PseudoInverse(const Eigen::Matrix<double, 3, 2>& X) {
    Eigen::JacobiSVD<Eigen::Matrix<double, 3, 2>> svd(X, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Vec2d sv = svd.singularValues();
    Eigen::Matrix<double, 3, 2> U = svd.matrixU().block<3, 2>(0, 0);
    Eigen::Matrix<double, 2, 2> V = svd.matrixV();
    Eigen::Matrix<double, 2, 3> U_adjoint = U.adjoint();
    double tolerance = std::numeric_limits<double>::epsilon() * 3 * std::abs(sv(0, 0));
    sv(0, 0) = std::abs(sv(0, 0)) > tolerance ? 1.0 / sv(0, 0) : 0;
    sv(1, 0) = std::abs(sv(1, 0)) > tolerance ? 1.0 / sv(1, 0) : 0;

    return V * sv.asDiagonal() * U_adjoint;
}

inline Eigen::Matrix<double, 3, 3> A_matrix(const Vec3d& v) {
    Eigen::Matrix<double, 3, 3> res;
    double squaredNorm = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    double norm = std::sqrt(squaredNorm);
    if (norm < 1e-5) {
        res = Eigen::Matrix<double, 3, 3>::Identity();
    } else {
        res = Eigen::Matrix<double, 3, 3>::Identity() + (1 - std::cos(norm)) / squaredNorm * SO3::hat(v) +
              (1 - std::sin(norm) / norm) / squaredNorm * SO3::hat(v) * SO3::hat(v);
    }
    return res;
}

inline Eigen::Matrix<double, 3, 3> A_inv(const Vec3d& v) {
    Eigen::Matrix<double, 3, 3> res;
    if (v.norm() > 1e-5) {
        res = Eigen::Matrix<double, 3, 3>::Identity() - 0.5 * SKEW_SYM_MATRIX(v) +
              (1 - v.norm() * std::cos(v.norm() / 2) / 2 / std::sin(v.norm() / 2)) * SKEW_SYM_MATRIX(v) *
                  SKEW_SYM_MATRIX(v) / v.squaredNorm();

    } else {
        res = Eigen::Matrix<double, 3, 3>::Identity() - 0.5 * SKEW_SYM_MATRIX(v) +
              ((1. / 12) + v.squaredNorm() * 1. / 720) * SKEW_SYM_MATRIX(v) * SKEW_SYM_MATRIX(v);
    }

    return res;
}

template <int N>
struct hash_vec {
    inline size_t operator()(const Eigen::Matrix<int, N, 1>& v) const;
};

template <>
inline size_t hash_vec<2>::operator()(const Eigen::Matrix<int, 2, 1>& v) const {
    return size_t(((v[0]) * 73856093) ^ ((v[1]) * 471943)) % 10000000;
}

template <>
inline size_t hash_vec<3>::operator()(const Eigen::Matrix<int, 3, 1>& v) const {
    return size_t(((v[0]) * 73856093) ^ ((v[1]) * 471943) ^ ((v[2]) * 83492791)) % 10000000;
}

template <typename T>
inline bool esti_plane(Eigen::Matrix<T, 4, 1>& pca_result, const PointVector& point, const T& threshold = 0.1f) {
    if (point.size() < fasterlio::MIN_NUM_MATCH_POINTS) {
        return false;
    }

    Eigen::Matrix<T, 3, 1> normvec;

    if (point.size() == fasterlio::NUM_MATCH_POINTS) {
        Eigen::Matrix<T, fasterlio::NUM_MATCH_POINTS, 3> A;
        Eigen::Matrix<T, fasterlio::NUM_MATCH_POINTS, 1> b;

        A.setZero();
        b.setOnes();
        b *= -1.0f;

        for (int j = 0; j < fasterlio::NUM_MATCH_POINTS; j++) {
            A(j, 0) = point[j].x;
            A(j, 1) = point[j].y;
            A(j, 2) = point[j].z;
        }

        normvec = A.colPivHouseholderQr().solve(b);
    } else {
        Eigen::MatrixXd A(point.size(), 3);
        Eigen::VectorXd b(point.size(), 1);

        A.setZero();
        b.setOnes();
        b *= -1.0f;

        for (int j = 0; j < point.size(); j++) {
            A(j, 0) = point[j].x;
            A(j, 1) = point[j].y;
            A(j, 2) = point[j].z;
        }

        Eigen::MatrixXd n = A.colPivHouseholderQr().solve(b);
        normvec(0, 0) = n(0, 0);
        normvec(1, 0) = n(1, 0);
        normvec(2, 0) = n(2, 0);
    }

    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (const auto& p : point) {
        Eigen::Matrix<T, 4, 1> temp = p.getVector4fMap();
        temp[3] = 1.0;
        if (fabs(pca_result.dot(temp)) > threshold) {
            return false;
        }
    }
    return true;
}

inline CloudPtr VoxelGrid(CloudPtr cloud, float voxel_size = 0.05) {
    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel.setInputCloud(cloud);

    CloudPtr output(new PointCloudType);
    voxel.filter(*output);

    return output;
}

inline double ToSec(uint64_t t) { return double(t) * 1e-9; }

template <typename T, int dim, typename PointType>
inline Eigen::Matrix<T, dim, 1> ToEigen(const PointType& pt) {
    return Eigen::Matrix<T, dim, 1>(pt.x, pt.y, pt.z);
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZ>(const pcl::PointXYZ& pt) {
    return pt.getVector3fMap();
}

template <>
inline Eigen::Matrix<float, 2, 1> ToEigen<float, 2, PointXYZIT>(const PointXYZIT& pt) {
    return Vec2f(pt.x, pt.y);
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZI>(const pcl::PointXYZI& pt) {
    return pt.getVector3fMap();
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZINormal>(const pcl::PointXYZINormal& pt) {
    return pt.getVector3fMap();
}

inline float calc_dist(const PointType& p1, const PointType& p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
}

inline float calc_dist(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2) { return (p1 - p2).squaredNorm(); }

template <typename PointT>
inline double distance2(const PointT& pt1, const PointT& pt2) {
    Eigen::Vector3f d = pt1.getVector3fMap() - pt2.getVector3fMap();
    return d.squaredNorm();
}

template <typename S>
inline Eigen::Matrix<S, 4, 4> Mat4FromArray(const std::vector<double>& v) {
    Eigen::Matrix<S, 4, 4> m;
    for (int i = 0; i < m.size(); ++i) m(i / 4, i % 4) = v[i];
    return m;
}

template <int N>
struct less_vec {
    inline bool operator()(const Eigen::Matrix<int, N, 1>& v1, const Eigen::Matrix<int, N, 1>& v2) const;
};

template <>
inline bool less_vec<2>::operator()(const Eigen::Matrix<int, 2, 1>& v1, const Eigen::Matrix<int, 2, 1>& v2) const {
    return v1[0] < v2[0] || (v1[0] == v2[0] && v1[1] < v2[1]);
}

template <>
inline bool less_vec<3>::operator()(const Eigen::Matrix<int, 3, 1>& v1, const Eigen::Matrix<int, 3, 1>& v2) const {
    return v1[0] < v2[0] || (v1[0] == v2[0] && v1[1] < v2[1]) || (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] < v2[2]);
}

template <typename C, int dim, typename S, typename Getter>
void ComputeMeanAndCov(const C& data, Eigen::Matrix<S, dim, 1>& mean, Eigen::Matrix<S, dim, dim>& cov,
                       Getter&& getter) {
    using D = Eigen::Matrix<S, dim, 1>;
    using E = Eigen::Matrix<S, dim, dim>;
    size_t len = data.size();
    assert(len > 1);

    mean = std::accumulate(data.begin(), data.end(), Eigen::Matrix<S, dim, 1>::Zero().eval(),
                           [&getter](const D& sum, const auto& data) -> D { return sum + getter(data); }) / len;
    cov = std::accumulate(data.begin(), data.end(), E::Zero().eval(),
                          [&mean, &getter](const E& sum, const auto& data) -> E {
                            D v = getter(data) - mean;
                            return sum + v * v.transpose();
                          }) / (len - 1);
}

template <typename S, int D>
void UpdateMeanAndCov(int hist_m, int curr_n, const Eigen::Matrix<S, D, 1>& hist_mean,
                      const Eigen::Matrix<S, D, D>& hist_var, const Eigen::Matrix<S, D, 1>& curr_mean,
                      const Eigen::Matrix<S, D, D>& curr_var, Eigen::Matrix<S, D, 1>& new_mean,
                      Eigen::Matrix<S, D, D>& new_var) {
    assert(hist_m > 0);
    assert(curr_n > 0);
    new_mean = (hist_m * hist_mean + curr_n * curr_mean) / (hist_m + curr_n);
    new_var = (hist_m * (hist_var + (hist_mean - new_mean) * (hist_mean - new_mean).template transpose()) +
               curr_n * (curr_var + (curr_mean - new_mean) * (curr_mean - new_mean).template transpose())) /
              (hist_m + curr_n);
}

inline void KeepAngleInPI(double& angle) {
    while (angle < -M_PI) {
        angle = angle + 2 * M_PI;
    }
    while (angle > M_PI) {
        angle = angle - 2 * M_PI;
    }
}

inline void KeepAngleIn2PI(double& angle) {
    while (angle <= 0) {
        angle = angle + 2 * M_PI;
    }
    while (angle > 2 * M_PI) {
        angle = angle - 2 * M_PI;
    }
}

inline builtin_interfaces::msg::Time FromSec(double t) {
    return StampFromSec(t);
}

inline PoseRPYD SE3ToRollPitchYaw(const SE3& pose) {
    auto rot = pose.rotationMatrix();
    tf2::Matrix3x3 temp_tf_matrix(rot(0, 0), rot(0, 1), rot(0, 2), rot(1, 0), rot(1, 1), rot(1, 2), rot(2, 0),
                                  rot(2, 1), rot(2, 2));
    PoseRPYD output;
    output.x = pose.translation()[0];
    output.y = pose.translation()[1];
    output.z = pose.translation()[2];

    temp_tf_matrix.getRPY(output.roll, output.pitch, output.yaw);
    return output;
}

inline SE3 XYZRPYToSE3(const PoseRPYD& pose) {
    using AA = Eigen::AngleAxisd;
    return SE3((AA(pose.yaw, Vec3d::UnitZ()) * AA(pose.pitch, Vec3d::UnitY()) * AA(pose.roll, Vec3d::UnitX())),
               Vec3d(pose.x, pose.y, pose.z));
}

template <typename T, typename C, typename FT, typename FP>
inline bool PoseInterp(double query_time, C&& data, FT&& take_time_func, FP&& take_pose_func, SE3& result,
                       T& best_match, float time_th = 0.5, bool verbose = false) {
    if (data.size() <= 1) {
        if (verbose) {
            LOG(INFO) << get_time_us() << "[INFO][math::" << __FUNCTION__ << "]" << "data size is too small for interp: " << data.size();
        }
        return false;
    }

    double last_time = take_time_func(*data.rbegin());
    double first_time = take_time_func(*data.begin());
    if (query_time > last_time) {
        if (verbose) {
            LOG(WARNING) << get_time_us() << "[WARN][math::" << __FUNCTION__ << "]" << "query time is later than last time.";
        }

        if (query_time < (last_time + time_th)) {
            if (data.size() == 1) {
                result = take_pose_func(*data.rbegin());
                best_match = *data.rbegin();
                return true;
            } else {
                auto second = data.rbegin();
                auto first = second;
                ++first;
                result = take_pose_func(*second);
                double time_first = take_time_func(*first);
                double time_second = take_time_func(*second);

                if (time_second <= time_first) {
                    result = take_pose_func(*data.rbegin());
                    best_match = *data.rbegin();
                    return true;
                }

                while ((time_second - time_first) < 0.05) {
                    first++;
                    if (first == data.rend()) {
                        result = take_pose_func(*data.rbegin());
                        best_match = *data.rbegin();
                        return true;
                    }
                    time_first = take_time_func(*first);
                }

                SE3 pose_first = take_pose_func(*first);
                SE3 pose_second = take_pose_func(*second);
                Vec6d xi = (pose_first.inverse() * pose_second).log();
                double s = (query_time - time_second) / (time_second - time_first);
                result = result * SE3::exp(s * xi);
                best_match = *data.rbegin();
                return true;
            }
        }

        if (verbose) {
            LOG(WARNING) << get_time_us() << "[WARN][math::" << __FUNCTION__ << "]" << "pose interp failed: " << query_time << ", " << (last_time + time_th);
        }
        return false;
    } else if (query_time < first_time) {
        if (query_time < (first_time - time_th)) {
            if (verbose) {
                LOG(WARNING) << get_time_us() << "[WARN][math::" << __FUNCTION__ << "]" << "query time too early: " << ", " << query_time << ", " << (first_time - time_th);
            }
            return false;
        } else {
            result = take_pose_func(*data.begin());
            best_match = *data.begin();
            return true;
        }
    }

    auto match_iter = data.begin();
    for (auto right = data.rbegin(); right != data.rend(); ++right) {
        auto left = right;
        ++left;

        if (left == data.rend()) {
            match_iter = std::prev(right.base());
            break;
        }

        if (take_time_func(*left) < query_time && take_time_func(*right) >= query_time) {
            match_iter = std::prev(left.base());
            break;
        }
    }

    auto match_iter_n = match_iter;
    match_iter_n++;
    if (match_iter_n == data.end() || take_time_func(*match_iter) >= query_time) {
        best_match = *match_iter;
        result = take_pose_func(*match_iter);
        return true;
    }

    double dt = take_time_func(*match_iter_n) - take_time_func(*match_iter);
    double s = (query_time - take_time_func(*match_iter)) / dt;
    if (fabs(dt) < 1e-6) {
        best_match = *match_iter;
        result = take_pose_func(*match_iter);
        return true;
    }

    SE3 pose_first = take_pose_func(*match_iter);
    SE3 pose_next = take_pose_func(*match_iter_n);
    result = {pose_first.unit_quaternion().slerp(s, pose_next.unit_quaternion()),
              pose_first.translation() * (1 - s) + pose_next.translation() * s};
    best_match = s < time_th ? *match_iter : *match_iter_n;
    return true;
}

}  // namespace lightning::math

#endif  // LIGHTNING_MATH_HPP