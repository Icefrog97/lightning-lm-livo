#pragma once

#include "common/eigen_types.h"
#include "common/point_def.h"
#include "perception/detection.h"
#include "perception/yolo_trt_detector.h"

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include <vikit/abstract_camera.h>

namespace lightning {
namespace perception {

struct DynCameraModel {
    int cam_id = 0;
    std::string model = "Pinhole";
    double fx = 0, fy = 0, cx = 0, cy = 0;
    int width = 0, height = 0;
    double scale = 1.0;
    /// OpenCV pinhole distortion [k1,k2,p1,p2,k3] (same as VIO CameraSlot)
    std::vector<double> distortion;
    std::string camera_loader_ns;
    /// p_cam = T_cam_lidar * p_lidar
    SE3 T_cam_lidar = SE3();
};

/// YOLO mask → drop lidar points that project into dynamic boxes.
class DynamicFilter {
public:
    struct Options {
        bool enable = false;
        YoloTrtDetector::Options yolo;
        int dilate_px = 5;
        double max_range = 60.0;
        bool publish_removed_cloud = true;
        std::string removed_body_topic = "/dynamic_filter/removed_body";
        std::string removed_world_topic = "/dynamic_filter/removed";
    };

    DynamicFilter() = default;

    bool Init(const Options &options, const std::vector<DynCameraModel> &cameras);
    bool Enabled() const { return options_.enable && ready_; }
    bool PublishRemovedCloud() const { return options_.publish_removed_cloud; }
    const std::string &RemovedBodyTopic() const { return options_.removed_body_topic; }
    const std::string &RemovedWorldTopic() const { return options_.removed_world_topic; }

    /// images: BGR (raw / distorted), same order as cameras_.
    /// cloud: lidar-frame motion-undistorted scan (in/out).
    /// Projection uses vikit world2cam (same geometric distortion as VIO).
    /// Returns number of removed points; -1 if skipped.
    int FilterCloud(CloudPtr &cloud_lidar,
                    const std::vector<cv::Mat> &images,
                    double stamp);

    const std::vector<Detection> &LastDetections() const { return last_dets_; }
    /// Draw cam0 boxes onto a VIO-scale image (pixels = original * cam->scale()).
    void OverlayOnScaledImage(cv::Mat &img) const;
    /// Lidar-frame undistorted points dropped by the last FilterCloud (empty if skipped).
    const CloudPtr &LastRemovedCloud() const { return last_removed_; }

private:
    std::vector<cv::Rect> BuildDilatedBoxes(const cv::Size &size,
                                            const std::vector<Detection> &dets) const;
    bool BuildCameraModels(const std::vector<DynCameraModel> &cameras);

    Options options_;
    bool ready_ = false;
    std::vector<DynCameraModel> cameras_;
    std::vector<std::unique_ptr<vk::AbstractCamera>> cam_models_;
    std::unique_ptr<YoloTrtDetector> detector_;
    std::vector<Detection> last_dets_;
    std::vector<Detection> last_cam0_dets_;
    CloudPtr last_removed_{new PointCloudType()};
};

}  // namespace perception
}  // namespace lightning
