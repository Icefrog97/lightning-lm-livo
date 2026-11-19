#pragma once

#include <Eigen/Core>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "Sophus/se3.hpp"
#include "vio/point_with_var.h"
#include "vio/voxel_location.h"

using V2D = Eigen::Vector2d;
using V3D = Eigen::Vector3d;
using V3F = Eigen::Vector3f;
using M3D = Eigen::Matrix3d;
using SE3 = Sophus::SE3d;

// yaml 外参和 R *= Exp(dθ) 都会让 R 略不正交；Sophus SO3(R)/SE3(R,t) 会 abort。
inline Eigen::Quaterniond QuatFromRotation(const M3D& R) {
    Eigen::Quaterniond q(R);
    q.normalize();
    return q;
}

inline SE3 Se3FromRt(const M3D& R, const V3D& t) { return SE3(QuatFromRotation(R), t); }

#define DIM_STATE (19)
#define INIT_COV (0.01)
#define SIZE_LARGE (500)
#define SIZE_SMALL (100)
#define VEC_FROM_ARRAY(v) (v)[0], (v)[1], (v)[2]
#define MAT_FROM_ARRAY(v) (v)[0], (v)[1], (v)[2], (v)[3], (v)[4], (v)[5], (v)[6], (v)[7], (v)[8]
#define MD(a, b) Eigen::Matrix<double, (a), (b)>
#define VD(a) Eigen::Matrix<double, (a), 1>
#define SKEW_SYM_MATRX(v) 0.0, -(v)[2], (v)[1], (v)[2], 0.0, -(v)[0], -(v)[1], (v)[0], 0.0
#ifndef ROOT_DIR
#define ROOT_DIR "/tmp/"
#endif
#define DEBUG_FILE_DIR(name) (std::string(ROOT_DIR) + (name))

using pointWithVar = lightning::PointWithVar;
using VOXEL_LOCATION = lightning::VoxelLocation;

template <typename T>
Eigen::Matrix<T, 3, 3> Exp(const T& v1, const T& v2, const T& v3) {
    const T norm = std::sqrt(v1 * v1 + v2 * v2 + v3 * v3);
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
    if (norm > 0.00001) {
        const T r_ang[3] = {v1 / norm, v2 / norm, v3 / norm};
        Eigen::Matrix<T, 3, 3> K;
        K << SKEW_SYM_MATRX(r_ang);
        return Eye3 + std::sin(norm) * K + (1.0 - std::cos(norm)) * K * K;
    }
    return Eye3;
}

template <typename T>
Eigen::Matrix<T, 3, 1> Log(const Eigen::Matrix<T, 3, 3>& R) {
    const T theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
    Eigen::Matrix<T, 3, 1> K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
    return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

struct StatesGroup {
    StatesGroup() {
        rot_end = M3D::Identity();
        pos_end = V3D::Zero();
        vel_end = V3D::Zero();
        bias_g = V3D::Zero();
        bias_a = V3D::Zero();
        gravity = V3D::Zero();
        inv_expo_time = 1.0;
        cov = MD(DIM_STATE, DIM_STATE)::Identity() * INIT_COV;
        cov(6, 6) = 0.00001;
        cov.block<9, 9>(10, 10) = MD(9, 9)::Identity() * 0.00001;
    }

    StatesGroup(const StatesGroup& b) { *this = b; }

    StatesGroup& operator=(const StatesGroup& b) {
        rot_end = b.rot_end;
        pos_end = b.pos_end;
        vel_end = b.vel_end;
        bias_g = b.bias_g;
        bias_a = b.bias_a;
        gravity = b.gravity;
        inv_expo_time = b.inv_expo_time;
        cov = b.cov;
        return *this;
    }

    StatesGroup operator+(const Eigen::Matrix<double, DIM_STATE, 1>& state_add) {
        StatesGroup a;
        a.rot_end = rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
        a.pos_end = pos_end + state_add.block<3, 1>(3, 0);
        a.inv_expo_time = inv_expo_time + state_add(6, 0);
        a.vel_end = vel_end + state_add.block<3, 1>(7, 0);
        a.bias_g = bias_g + state_add.block<3, 1>(10, 0);
        a.bias_a = bias_a + state_add.block<3, 1>(13, 0);
        a.gravity = gravity + state_add.block<3, 1>(16, 0);
        a.cov = cov;
        return a;
    }

    StatesGroup& operator+=(const Eigen::Matrix<double, DIM_STATE, 1>& state_add) {
        rot_end = rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
        pos_end += state_add.block<3, 1>(3, 0);
        inv_expo_time += state_add(6, 0);
        vel_end += state_add.block<3, 1>(7, 0);
        bias_g += state_add.block<3, 1>(10, 0);
        bias_a += state_add.block<3, 1>(13, 0);
        gravity += state_add.block<3, 1>(16, 0);
        return *this;
    }

    Eigen::Matrix<double, DIM_STATE, 1> operator-(const StatesGroup& b) {
        Eigen::Matrix<double, DIM_STATE, 1> a;
        const M3D rotd(b.rot_end.transpose() * rot_end);
        a.block<3, 1>(0, 0) = Log(rotd);
        a.block<3, 1>(3, 0) = pos_end - b.pos_end;
        a(6, 0) = inv_expo_time - b.inv_expo_time;
        a.block<3, 1>(7, 0) = vel_end - b.vel_end;
        a.block<3, 1>(10, 0) = bias_g - b.bias_g;
        a.block<3, 1>(13, 0) = bias_a - b.bias_a;
        a.block<3, 1>(16, 0) = gravity - b.gravity;
        return a;
    }

    M3D rot_end = M3D::Identity();
    V3D pos_end = V3D::Zero();
    V3D vel_end = V3D::Zero();
    double inv_expo_time = 1.0;
    V3D bias_g = V3D::Zero();
    V3D bias_a = V3D::Zero();
    V3D gravity = V3D::Zero();
    Eigen::Matrix<double, DIM_STATE, DIM_STATE> cov = Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity() * INIT_COV;
};
