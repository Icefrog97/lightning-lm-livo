#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace lightning {
namespace perception {

struct Detection {
    cv::Rect box;
    float confidence = 0.f;
    int class_id = 0;
};

}  // namespace perception
}  // namespace lightning
