// Jetson / JetPack TensorRT 8.x: legacy Binding API (getNbBindings / executeV2).
// Selected by CMake option YOLO_TRT_JETSON=ON. Do not compile together with
// yolo_trt_detector.cpp (PC TensorRT 10 IO-tensor API).
#include "perception/yolo_trt_detector.h"

#include <glog/logging.h>

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstring>
#include <fstream>
#include <unistd.h>

#if defined(LIDAR_SLAM_USE_TENSORRT)
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#endif

namespace lightning {
namespace perception {
namespace {

cv::Mat Letterbox(const cv::Mat &src, int target, double &scale, int &pad_w, int &pad_h) {
    const int ow = src.cols;
    const int oh = src.rows;
    scale = std::min(static_cast<double>(target) / ow, static_cast<double>(target) / oh);
    const int nw = static_cast<int>(ow * scale);
    const int nh = static_cast<int>(oh * scale);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh));
    pad_w = (target - nw) / 2;
    pad_h = (target - nh) / 2;
    cv::Mat out;
    cv::copyMakeBorder(resized, out, pad_h, target - nh - pad_h, pad_w, target - nw - pad_w,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

bool KeepClass(const std::vector<int> &keep, int class_id) {
    if (keep.empty())
        return true;
    for (int id : keep)
        if (id == class_id)
            return true;
    return false;
}

}  // namespace

#if defined(LIDAR_SLAM_USE_TENSORRT)

struct TrtDestroyer {
    template <typename T>
    void operator()(T *obj) const {
        delete obj;
    }
};

struct YoloTrtDetector::Impl {
    Options opt;
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char *msg) noexcept override {
            if (severity <= Severity::kWARNING)
                LOG(WARNING) << "[TensorRT] " << msg;
        }
    } logger;

    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::shared_ptr<nvinfer1::IExecutionContext> context;
    std::vector<void *> bindings;
    std::vector<size_t> binding_sizes;
    int input_index = -1;
    int output_index = -1;
    int num_candidates = 0;  // N in 1xNx6
    Timing last_timing;

    bool Init() {
        if (opt.model_path.empty()) {
            LOG(ERROR) << "[YoloTrt] empty model_path";
            return false;
        }
        std::string engine_path = opt.model_path;
        const bool is_engine =
            engine_path.size() >= 7 && engine_path.substr(engine_path.size() - 7) == ".engine";
        if (!is_engine) {
            // Prefer sibling .engine: best.onnx → best.engine (not best.onnx.engine).
            std::string sibling = engine_path;
            const auto dot = sibling.find_last_of('.');
            if (dot != std::string::npos)
                sibling = sibling.substr(0, dot) + ".engine";
            else
                sibling += ".engine";
            if (access(sibling.c_str(), R_OK) == 0)
                engine_path = sibling;
            else if (access((opt.model_path + ".engine").c_str(), R_OK) == 0)
                engine_path = opt.model_path + ".engine";
            else if (!LoadFromOnnx(opt.model_path))
                return false;
            else
                engine_path = opt.model_path + ".engine";
        }
        if (!engine && !LoadEngine(engine_path)) {
            LOG(ERROR) << "[YoloTrt] failed to load engine: " << engine_path;
            return false;
        }

        context = std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext(), TrtDestroyer());
        if (!context)
            return false;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        const int nb = engine->getNbBindings();
        bindings.assign(nb, nullptr);
        binding_sizes.assign(nb, 0);
        for (int i = 0; i < nb; ++i) {
            const nvinfer1::Dims dims = engine->getBindingDimensions(i);
            size_t n = 1;
            for (int d = 0; d < dims.nbDims; ++d)
                n *= static_cast<size_t>(dims.d[d] > 0 ? dims.d[d] : 1);
            binding_sizes[i] = n * sizeof(float);
            cudaMalloc(&bindings[i], binding_sizes[i]);
            const char *name = engine->getBindingName(i);
            if (engine->bindingIsInput(i)) {
                input_index = i;
            } else if (std::string(name) == "output0" || output_index < 0) {
                output_index = i;
                // YOLO26 E2E: [1, N, 6] = xyxy + score + cls
                if (dims.nbDims >= 3 && dims.d[2] == 6) {
                    num_candidates = dims.d[1] > 0 ? dims.d[1] : static_cast<int>(n / 6);
                } else if (dims.nbDims == 2 && dims.d[1] == 6) {
                    num_candidates = dims.d[0] > 0 ? dims.d[0] : static_cast<int>(n / 6);
                }
            }
            LOG(INFO) << "[YoloTrt] binding " << i << " " << name << " elems=" << n
                      << " nbDims=" << dims.nbDims;
        }
#pragma GCC diagnostic pop

        if (input_index < 0 || output_index < 0) {
            LOG(ERROR) << "[YoloTrt] missing input/output binding";
            return false;
        }
        if (num_candidates <= 0) {
            LOG(ERROR) << "[YoloTrt] expect YOLO26 E2E output shape [1,N,6] (last dim == 6); "
                       << "re-export with end-to-end head";
            return false;
        }

        // One-shot warmup (same as IQIS): prime CUDA kernels; not done every frame.
        {
            cv::Mat dummy = cv::Mat::zeros(opt.imgsz, opt.imgsz, CV_8UC3);
            (void)Infer(dummy);
        }
        LOG(INFO) << "[YoloTrt] ready (YOLO26 E2E, Jetson TRT8), candidates=" << num_candidates;
        return true;
    }

    bool LoadEngine(const std::string &path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.good())
            return false;
        const auto size = file.tellg();
        file.seekg(0);
        std::vector<char> buf(static_cast<size_t>(size));
        if (!file.read(buf.data(), size))
            return false;
        auto runtime = std::unique_ptr<nvinfer1::IRuntime, TrtDestroyer>(nvinfer1::createInferRuntime(logger));
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(buf.data(), buf.size()), TrtDestroyer());
        return engine != nullptr;
    }

    bool LoadFromOnnx(const std::string &onnx) {
        if (access(onnx.c_str(), R_OK) != 0) {
            LOG(ERROR) << "[YoloTrt] ONNX not found or unreadable: " << onnx
                       << " (put YOLO26 E2E .onnx there, or set dynamic_filter/model_path in launch)";
            return false;
        }
        auto builder = std::unique_ptr<nvinfer1::IBuilder, TrtDestroyer>(nvinfer1::createInferBuilder(logger));
        if (!builder)
            return false;
        const auto flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network =
            std::unique_ptr<nvinfer1::INetworkDefinition, TrtDestroyer>(builder->createNetworkV2(flags));
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig, TrtDestroyer>(builder->createBuilderConfig());
        auto parser =
            std::unique_ptr<nvonnxparser::IParser, TrtDestroyer>(nvonnxparser::createParser(*network, logger));
        if (!parser->parseFromFile(onnx.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            LOG(ERROR) << "[YoloTrt] parse ONNX failed: " << onnx;
            return false;
        }
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256ULL << 20);
        if (builder->platformHasFastFp16())
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        auto plan = std::unique_ptr<nvinfer1::IHostMemory, TrtDestroyer>(
            builder->buildSerializedNetwork(*network, *config));
        if (!plan)
            return false;
        {
            std::ofstream out(onnx + ".engine", std::ios::binary);
            out.write(static_cast<const char *>(plan->data()), plan->size());
        }
        auto runtime = std::unique_ptr<nvinfer1::IRuntime, TrtDestroyer>(nvinfer1::createInferRuntime(logger));
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(plan->data(), plan->size()), TrtDestroyer());
        return engine != nullptr;
    }

    std::vector<Detection> Infer(const cv::Mat &bgr) {
        if (!context || bgr.empty())
            return {};
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();

        double scale = 1.0;
        int pad_w = 0, pad_h = 0;
        cv::Mat letter = Letterbox(bgr, opt.imgsz, scale, pad_w, pad_h);
        cv::cvtColor(letter, letter, cv::COLOR_BGR2RGB);
        letter.convertTo(letter, CV_32FC3, 1.0 / 255.0);
        std::vector<cv::Mat> ch(3);
        cv::split(letter, ch);
        std::vector<float> input(3 * opt.imgsz * opt.imgsz);
        for (int c = 0; c < 3; ++c)
            std::memcpy(input.data() + c * opt.imgsz * opt.imgsz, ch[c].data,
                        opt.imgsz * opt.imgsz * sizeof(float));

        const auto t1 = Clock::now();

        cudaMemcpy(bindings[input_index], input.data(), binding_sizes[input_index], cudaMemcpyHostToDevice);
        context->executeV2(bindings.data());

        std::vector<float> output(binding_sizes[output_index] / sizeof(float));
        cudaMemcpy(output.data(), bindings[output_index], binding_sizes[output_index], cudaMemcpyDeviceToHost);

        const auto t2 = Clock::now();

        // YOLO26 E2E: row-major [N, 6] = x1,y1,x2,y2,score,cls (letterbox coords)
        const int n = std::min(num_candidates, static_cast<int>(output.size() / 6));
        std::vector<Detection> dets;
        dets.reserve(64);
        const cv::Rect img_roi(0, 0, bgr.cols, bgr.rows);
        for (int i = 0; i < n; ++i) {
            const float *row = output.data() + i * 6;
            const float score = row[4];
            if (score < static_cast<float>(opt.conf_thres))
                continue;
            const int cls = static_cast<int>(row[5]);
            if (!KeepClass(opt.keep_class_ids, cls))
                continue;
            const int xmin = static_cast<int>((row[0] - pad_w) / scale);
            const int ymin = static_cast<int>((row[1] - pad_h) / scale);
            const int xmax = static_cast<int>((row[2] - pad_w) / scale);
            const int ymax = static_cast<int>((row[3] - pad_h) / scale);
            Detection d;
            d.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin) & img_roi;
            d.confidence = score;
            d.class_id = cls;
            if (d.box.area() > 0)
                dets.push_back(d);
        }
        const auto t3 = Clock::now();
        last_timing.pre_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        last_timing.trt_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        last_timing.post_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        return dets;
    }

    ~Impl() {
        for (void *p : bindings)
            if (p)
                cudaFree(p);
    }
};

YoloTrtDetector::YoloTrtDetector(Options options) : impl_(std::make_unique<Impl>()) {
    impl_->opt = std::move(options);
    if (!impl_->Init())
        impl_.reset();
}

YoloTrtDetector::~YoloTrtDetector() = default;

bool YoloTrtDetector::Ready() const { return impl_ != nullptr; }

std::vector<Detection> YoloTrtDetector::Infer(const cv::Mat &bgr) {
    if (!impl_)
        return {};
    return impl_->Infer(bgr);
}

YoloTrtDetector::Timing YoloTrtDetector::LastTiming() const {
    return impl_ ? impl_->last_timing : Timing{};
}

#else  // !LIDAR_SLAM_USE_TENSORRT

struct YoloTrtDetector::Impl {};

YoloTrtDetector::YoloTrtDetector(Options) {
    LOG(WARNING) << "[YoloTrt] built without TensorRT (LIDAR_SLAM_USE_TENSORRT=0); detector is stub";
}

YoloTrtDetector::~YoloTrtDetector() = default;

bool YoloTrtDetector::Ready() const { return false; }

std::vector<Detection> YoloTrtDetector::Infer(const cv::Mat &) { return {}; }

YoloTrtDetector::Timing YoloTrtDetector::LastTiming() const { return {}; }

#endif

}  // namespace perception
}  // namespace lightning
