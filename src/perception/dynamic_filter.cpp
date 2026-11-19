#include "perception/dynamic_filter.h"

#include "common/timer.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <yaml_loader.h>

namespace lightning {
namespace perception {

namespace {

double DistortionAt(const DynCameraModel &camera, size_t idx) {
    return idx < camera.distortion.size() ? camera.distortion[idx] : 0.0;
}

std::string Lowercase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace

bool DynamicFilter::BuildCameraModels(const std::vector<DynCameraModel> &cameras) {
    cam_models_.clear();
    cam_models_.reserve(cameras.size());

    for (const auto &camera : cameras) {
        if (camera.width <= 0 || camera.height <= 0 || camera.fx <= 0.0 || camera.fy <= 0.0) {
            LOG(WARNING) << "[DynamicFilter] invalid camera " << camera.cam_id
                         << " (width/height/fx/fy); projection disabled for this cam";
            cam_models_.emplace_back(nullptr);
            continue;
        }

        const std::string model = Lowercase(camera.model);
        YAML::Node node;
        if (model == "fisheye" || model == "equidistant" || model == "equidistantcamera") {
            node["cam_model"] = "EquidistantCamera";
            node["k1"] = DistortionAt(camera, 0);
            node["k2"] = DistortionAt(camera, 1);
            node["k3"] = DistortionAt(camera, 2);
            node["k4"] = DistortionAt(camera, 3);
        } else if (model == "pinhole" || model.empty()) {
            node["cam_model"] = "Pinhole";
            node["cam_d0"] = DistortionAt(camera, 0);
            node["cam_d1"] = DistortionAt(camera, 1);
            node["cam_d2"] = DistortionAt(camera, 2);
            node["cam_d3"] = DistortionAt(camera, 3);
            node["cam_d4"] = DistortionAt(camera, 4);
        } else {
            node["cam_model"] = camera.model;
        }
        node["cam_width"] = camera.width;
        node["cam_height"] = camera.height;
        node["scale"] = camera.scale;
        node["cam_fx"] = camera.fx;
        node["cam_fy"] = camera.fy;
        node["cam_cx"] = camera.cx;
        node["cam_cy"] = camera.cy;

        vk::AbstractCamera *loaded = nullptr;
        if (!vk::camera_loader::loadFromYaml(node, loaded) || loaded == nullptr) {
            LOG(WARNING) << "[DynamicFilter] unsupported model '" << camera.model
                         << "' for cam " << camera.cam_id;
            cam_models_.emplace_back(nullptr);
            continue;
        }
        cam_models_.emplace_back(loaded);
        LOG(INFO) << "[DynamicFilter] cam " << camera.cam_id << " loaded via yaml_loader model="
                  << node["cam_model"].as<std::string>();
    }

    return std::any_of(cam_models_.begin(), cam_models_.end(),
                       [](const std::unique_ptr<vk::AbstractCamera> &c) { return c != nullptr; });
}

bool DynamicFilter::Init(const Options &options, const std::vector<DynCameraModel> &cameras) {
    cameras_ = cameras;
    options_ = options;

    if (!options_.enable) {
        LOG(INFO) << "[DynamicFilter] disabled";
        return true;
    }
    if (cameras_.empty()) {
        LOG(WARNING) << "[DynamicFilter] enable=true but no camera model; disabled";
        options_.enable = false;
        return false;
    }

    if (!BuildCameraModels(cameras_)) {
        LOG(WARNING) << "[DynamicFilter] no valid projection camera model; disabled";
        options_.enable = false;
        return false;
    }

    detector_ = std::make_unique<YoloTrtDetector>(options_.yolo);
    ready_ = detector_ && detector_->Ready();
    if (!ready_) {
        LOG(WARNING) << "[DynamicFilter] detector not ready (check TensorRT build / model_path)";
        options_.enable = false;
        return false;
    }

    LOG(INFO) << get_time_us() << "[INFO][DynamicFilter] ready, model=" << options_.yolo.model_path
              << ", cameras=" << cameras_.size()
              << ", keep_class_ids=" << options_.yolo.keep_class_ids.size();
    return true;
}

std::vector<cv::Rect> DynamicFilter::BuildDilatedBoxes(const cv::Size &size,
                                                       const std::vector<Detection> &dets) const {
    std::vector<cv::Rect> boxes;
    boxes.reserve(dets.size());
    const cv::Rect img(0, 0, size.width, size.height);
    const int pad = std::max(0, options_.dilate_px);
    for (const auto &d : dets) {
        cv::Rect r = d.box;
        r.x -= pad;
        r.y -= pad;
        r.width += 2 * pad;
        r.height += 2 * pad;
        r &= img;
        if (r.area() > 0)
            boxes.push_back(r);
    }
    return boxes;
}

void DynamicFilter::OverlayOnScaledImage(cv::Mat &img) const {
    if (img.empty() || last_cam0_dets_.empty())
        return;
    const double s = (cam_models_.empty() || cam_models_[0] == nullptr) ? 1.0 : cam_models_[0]->scale();
    const cv::Rect img_roi(0, 0, img.cols, img.rows);
    for (const auto &d : last_cam0_dets_) {
        cv::Rect r(static_cast<int>(std::lround(d.box.x * s)),
                   static_cast<int>(std::lround(d.box.y * s)),
                   static_cast<int>(std::lround(d.box.width * s)),
                   static_cast<int>(std::lround(d.box.height * s)));
        r &= img_roi;
        if (r.area() <= 0)
            continue;
        cv::rectangle(img, r, cv::Scalar(0, 0, 255), 2);
        const std::string text = "c" + std::to_string(d.class_id) + " " +
                                 cv::format("%.2f", d.confidence);
        cv::putText(img, text, r.tl() + cv::Point(0, -4), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 255), 1);
    }
}

int DynamicFilter::FilterCloud(CloudPtr &cloud_lidar,
                               const std::vector<cv::Mat> &images,
                               double stamp) {
    last_removed_.reset(new PointCloudType);
    if (!Enabled() || !cloud_lidar || cloud_lidar->empty() || images.empty())
        return -1;
    (void)stamp;

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    const size_t n_cam = std::min({cameras_.size(), images.size(), cam_models_.size()});
    std::vector<std::vector<cv::Rect>> dyn_boxes(n_cam);
    last_dets_.clear();
    last_cam0_dets_.clear();
    double pre_ms = 0, trt_ms = 0, post_ms = 0, mask_ms = 0;

    for (size_t i = 0; i < n_cam; ++i) {
        if (images[i].empty() || cam_models_[i] == nullptr)
            continue;
        auto dets = detector_->Infer(images[i]);
        const auto t = detector_->LastTiming();
        pre_ms += t.pre_ms;
        trt_ms += t.trt_ms;
        post_ms += t.post_ms;
        last_dets_.insert(last_dets_.end(), dets.begin(), dets.end());
        if (i == 0)
            last_cam0_dets_ = dets;
        const auto t_mask0 = Clock::now();
        // YOLO boxes are in the raw image; world2cam is in vikit scale (e.g. 0.5).
        const double s = cam_models_[i]->scale();
        std::vector<Detection> scaled = dets;
        for (auto &d : scaled) {
            d.box.x = static_cast<int>(std::lround(d.box.x * s));
            d.box.y = static_cast<int>(std::lround(d.box.y * s));
            d.box.width = static_cast<int>(std::lround(d.box.width * s));
            d.box.height = static_cast<int>(std::lround(d.box.height * s));
        }
        dyn_boxes[i] = BuildDilatedBoxes(
            cv::Size(cam_models_[i]->width(), cam_models_[i]->height()), scaled);
        mask_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_mask0).count();
    }

    const auto t_infer = Clock::now();

    CloudPtr kept(new PointCloudType);
    kept->reserve(cloud_lidar->size());
    last_removed_->reserve(cloud_lidar->size() / 8);
    int removed = 0;
    const float max_r2 = static_cast<float>(options_.max_range * options_.max_range);

    for (const auto &pt : cloud_lidar->points) {
        const float r2 = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
        if (r2 > max_r2) {
            kept->push_back(pt);
            continue;
        }

        bool dynamic = false;
        const Vec3d p_lidar(pt.x, pt.y, pt.z);
        for (size_t i = 0; i < n_cam && !dynamic; ++i) {
            if (dyn_boxes[i].empty() || cam_models_[i] == nullptr)
                continue;
            const Vec3d p_cam = cameras_[i].T_cam_lidar * p_lidar;
            if (p_cam.z() <= 0.1)
                continue;

            // Same as VIO: world2cam is in scaled camera pixels (match dyn_boxes).
            const Eigen::Vector2d uv = cam_models_[i]->world2cam(p_cam);
            const cv::Point px(static_cast<int>(std::lround(uv.x())),
                               static_cast<int>(std::lround(uv.y())));
            for (const auto &box : dyn_boxes[i]) {
                if (box.contains(px)) {
                    dynamic = true;
                    break;
                }
            }
        }

        if (dynamic) {
            last_removed_->push_back(pt);
            ++removed;
        } else
            kept->push_back(pt);
    }

    kept->width = kept->size();
    kept->height = 1;
    kept->is_dense = false;
    last_removed_->width = last_removed_->size();
    last_removed_->height = 1;
    last_removed_->is_dense = false;
    cloud_lidar.swap(kept);

    const auto t1 = Clock::now();
    const double infer_ms =
        std::chrono::duration<double, std::milli>(t_infer - t0).count();
    const double proj_ms =
        std::chrono::duration<double, std::milli>(t1 - t_infer).count();
    const double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    LOG(INFO) << get_time_us() << "[INFO][DynamicFilter] removed=" << removed
              << ", kept=" << cloud_lidar->size()
              << ", dets=" << last_dets_.size()
              << ", pre=" << pre_ms << "ms"
              << ", trt=" << trt_ms << "ms"
              << ", post=" << post_ms << "ms"
              << ", mask=" << mask_ms << "ms"
              << ", infer=" << infer_ms << "ms"
              << ", proj=" << proj_ms << "ms"
              << ", total=" << total_ms << "ms";
    return removed;
}

}  // namespace perception
}  // namespace lightning
