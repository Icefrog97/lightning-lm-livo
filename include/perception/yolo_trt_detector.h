#pragma once

#include "perception/detection.h"

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace lightning {
namespace perception {

/// YOLO26 end-to-end TensorRT detector (output0: 1xNx6, no NMS). From IQIS3.3.
class YoloTrtDetector {
public:
    struct Options {
        std::string model_path;  // .onnx or .engine (YOLO26 E2E export)
        int imgsz = 640;
        double conf_thres = 0.4;
        double iou_thres = 0.45;  // unused (E2E has no NMS); kept for yaml compat
        int num_classes = 80;     // unused for decode; kept for yaml compat
        /// If empty, keep all classes. Otherwise only these class ids (COCO: 0 person, 2 car, ...)
        std::vector<int> keep_class_ids;
    };

    struct Timing {
        double pre_ms = 0;   // letterbox + RGB + /255 + NCHW
        double trt_ms = 0;   // H2D + enqueue + D2H + sync
        double post_ms = 0;  // decode boxes
    };

    explicit YoloTrtDetector(Options options);
    ~YoloTrtDetector();

    bool Ready() const;
    std::vector<Detection> Infer(const cv::Mat &bgr);
    Timing LastTiming() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace perception
}  // namespace lightning
