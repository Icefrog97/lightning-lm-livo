#pragma once

#include <string>
#include <vector>

#include "common/eigen_types.h"

namespace lightning {

struct CameraSlot {
    int cam_id = 0;
    std::string topic;
    std::string model = "Pinhole";
    int width = 0;
    int height = 0;
    double scale = 1.0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    std::vector<double> distortion;
    double undistort_scale_x = 1.0;
    double undistort_scale_y = 1.0;
    double undistort_shift_x = 0.0;
    double undistort_shift_y = 0.0;
    std::string camera_loader_ns;
    SE3 T_cam_imu = SE3();
};

}  // namespace lightning
