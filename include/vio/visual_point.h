#pragma once

#include <boost/noncopyable.hpp>

#include <list>

#include "common/eigen_types.h"

namespace lightning {

struct Feature;

class VisualPoint : boost::noncopyable {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Vec3d pos_ = Vec3d::Zero();
    Vec3d normal_ = Vec3d::Zero();
    Matrix3 normal_information_ = Matrix3::Zero();
    Vec3d previous_normal_ = Vec3d::Zero();
    std::list<Feature*> obs_;
    Mat3d covariance_ = Mat3d::Zero();
    bool is_converged_ = false;
    bool is_normal_initialized_ = false;
    bool has_ref_patch_ = false;
    Feature* ref_patch = nullptr;

    explicit VisualPoint(const Vec3d& pos);
    ~VisualPoint();

    void findMinScoreFeature(const Vec3d& framepos, Feature*& ftr) const;
    void deleteNonRefPatchFeatures();
    void deleteFeatureRef(Feature* ftr);
    void addFrameRef(Feature* ftr);
    bool getCloseViewObs(const Vec3d& pos, Feature*& obs, const Vec2d& cur_px) const;
};

}  // namespace lightning
