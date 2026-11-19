#pragma once

#include <boost/noncopyable.hpp>
#include <opencv2/core.hpp>
#include <vikit/abstract_camera.h>

#include <list>
#include <memory>
#include <vector>

#include "common/eigen_types.h"

namespace lightning {

class VisualPoint;
struct Feature;

using Features = std::list<Feature*>;
using ImgPyr = std::vector<cv::Mat>;

class Frame : boost::noncopyable {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static int frame_counter_;

    int id_ = 0;
    vk::AbstractCamera* cam_ = nullptr;
    SE3 T_f_w_ = SE3();
    SE3 T_f_w_prior_ = SE3();
    cv::Mat img_;
    Features fts_;

    Frame(vk::AbstractCamera* cam, const cv::Mat& img);
    ~Frame();

    void initFrame(const cv::Mat& img);

    size_t nObs() const { return fts_.size(); }

    Vec2d w2c(const Vec3d& xyz_w) const { return cam_->world2cam(T_f_w_ * xyz_w); }
    Vec2d w2c_prior(const Vec3d& xyz_w) const { return cam_->world2cam(T_f_w_prior_ * xyz_w); }
    Vec3d c2f(const Vec2d& px) const { return cam_->cam2world(px[0], px[1]); }
    Vec3d c2f(double x, double y) const { return cam_->cam2world(x, y); }
    Vec3d w2f(const Vec3d& xyz_w) const { return T_f_w_ * xyz_w; }
    Vec3d f2w(const Vec3d& f) const { return T_f_w_.inverse() * f; }
    Vec2d f2c(const Vec3d& f) const { return cam_->world2cam(f); }
    Vec3d pos() const { return T_f_w_.inverse().translation(); }
};

using FramePtr = std::unique_ptr<Frame>;

namespace frame_utils {
void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr);
}  // namespace frame_utils

}  // namespace lightning
