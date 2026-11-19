#pragma once

#include "common/eigen_types.h"

namespace lightning {

struct PointWithVar {
    Vec3d point_w = Vec3d::Zero();
    Vec3d point_b = Vec3d::Zero();
    Vec3d point_i = Vec3d::Zero();
    Vec3d normal = Vec3d::Zero();
    Mat3d var = Mat3d::Identity();
    Mat3d var_nostate = Mat3d::Zero();
    Mat3d body_var = Mat3d::Zero();
    Mat3d point_crossmat = Mat3d::Zero();
};

}  // namespace lightning
