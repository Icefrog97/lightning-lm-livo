#include "vio/frame.h"

#include <algorithm>
#include <stdexcept>

#include <vikit/vision.h>

#include "vio/feature.h"

namespace lightning {

int Frame::frame_counter_ = 0;

Frame::Frame(vk::AbstractCamera* cam, const cv::Mat& img) : id_(frame_counter_++), cam_(cam) { initFrame(img); }

Frame::~Frame() {
    std::for_each(fts_.begin(), fts_.end(), [](Feature* feature) { delete feature; });
}

void Frame::initFrame(const cv::Mat& img) {
    if (img.empty()) {
        throw std::runtime_error("Frame: provided image is empty");
    }

    if (cam_ == nullptr) {
        throw std::runtime_error("Frame: camera model is null");
    }

    if (img.cols != static_cast<int>(cam_->width()) || img.rows != static_cast<int>(cam_->height())) {
        throw std::runtime_error("Frame: provided image has not the same size as the camera model");
    }

    if (img.type() != CV_8UC1) {
        throw std::runtime_error("Frame: provided image is not grayscale");
    }

    img_ = img;
}

namespace frame_utils {

void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr) {
    pyr.resize(n_levels);
    pyr[0] = img_level_0;
    for (int i = 1; i < n_levels; ++i) {
        pyr[i] = cv::Mat(pyr[i - 1].rows / 2, pyr[i - 1].cols / 2, CV_8U);
        vk::halfSample(pyr[i - 1], pyr[i]);
    }
}

}  // namespace frame_utils

}  // namespace lightning
