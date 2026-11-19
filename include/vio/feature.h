#pragma once

#include <opencv2/core.hpp>

#include "common/eigen_types.h"

namespace lightning {

class VisualPoint;

struct Feature {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum FeatureType {
        CORNER,
        EDGELET,
    };

    int id_ = 0;
    FeatureType type_ = CORNER;
    cv::Mat img_;
    Vec2d px_ = Vec2d::Zero();
    Vec3d f_ = Vec3d::Zero();
    int level_ = 0;
    VisualPoint* point_ = nullptr;
    Vec2d grad_ = Vec2d::Zero();
    SE3 T_f_w_ = SE3();
    float* patch_ = nullptr;
    float score_ = 0.0f;
    float mean_ = 0.0f;
    double inv_expo_time_ = 1.0;
    int cam_id_ = 0;

    Feature(VisualPoint* point, float* patch, const Vec2d& px, const Vec3d& f, const SE3& T_f_w, int level)
        : type_(CORNER), px_(px), f_(f), level_(level), point_(point), T_f_w_(T_f_w), patch_(patch) {}

    Vec3d pos() const { return T_f_w_.inverse().translation(); }

    ~Feature() { delete[] patch_; }
};

}  // namespace lightning
