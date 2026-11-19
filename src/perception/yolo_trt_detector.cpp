// PC TensorRT 10+: IO-tensor API (getNbIOTensors / enqueueV3).
// Selected by CMake option YOLO_TRT_JETSON=OFF (default). Do not compile
// together with yolo_trt_detector_jetson.cpp (Jetson TensorRT 8 Binding API).
#include "perception/yolo_trt_detector.h"

#include <glog/logging.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <unistd.h>

#if defined(LIDAR_SLAM_USE_TENSORRT)
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "perception/yolo_letterbox.h"
#endif

namespace lightning {
namespace perception {
namespace {

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
    std::vector<std::string> tensor_names;
    std::string input_name;
    std::string output_name;
    int input_index = -1;
    int output_index = -1;
    int num_candidates = 0;  // N in 1xNx6
    cudaStream_t stream = nullptr;
    cudaGraphExec_t graph_exec = nullptr;  // 启动时录一次，每帧 cudaGraphLaunch
    float *host_output = nullptr;  // pinned，D2H 目的
    uint8_t *bgr_dev = nullptr;    // 原图 uint8，按帧上传
    size_t bgr_dev_bytes = 0;
    cudaEvent_t pre_start = nullptr;
    cudaEvent_t pre_end = nullptr;
    cudaEvent_t trt_end = nullptr;
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

        // TensorRT 10+: IO-tensor API (legacy getNbBindings / executeV2 removed).
        const int nb = engine->getNbIOTensors();
        bindings.assign(nb, nullptr);
        binding_sizes.assign(nb, 0);
        tensor_names.resize(static_cast<size_t>(nb));
        for (int i = 0; i < nb; ++i) {
            const char *name = engine->getIOTensorName(i);
            tensor_names[static_cast<size_t>(i)] = name;
            const nvinfer1::Dims dims = engine->getTensorShape(name);
            size_t n = 1;
            for (int d = 0; d < dims.nbDims; ++d)
                n *= static_cast<size_t>(dims.d[d] > 0 ? dims.d[d] : 1);
            binding_sizes[static_cast<size_t>(i)] = n * sizeof(float);
            cudaMalloc(&bindings[static_cast<size_t>(i)], binding_sizes[static_cast<size_t>(i)]);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                input_index = i;
                input_name = name;
            } else if (std::string(name) == "output0" || output_index < 0) {
                output_index = i;
                output_name = name;
                // YOLO26 E2E: [1, N, 6] = xyxy + score + cls
                if (dims.nbDims >= 3 && dims.d[2] == 6) {
                    num_candidates = dims.d[1] > 0 ? dims.d[1] : static_cast<int>(n / 6);
                } else if (dims.nbDims == 2 && dims.d[1] == 6) {
                    num_candidates = dims.d[0] > 0 ? dims.d[0] : static_cast<int>(n / 6);
                }
            }
            LOG(INFO) << "[YoloTrt] tensor " << i << " " << name << " elems=" << n
                      << " nbDims=" << dims.nbDims;
        }

        if (input_index < 0 || output_index < 0) {
            LOG(ERROR) << "[YoloTrt] missing input/output tensor";
            return false;
        }
        if (num_candidates <= 0) {
            LOG(ERROR) << "[YoloTrt] expect YOLO26 E2E output shape [1,N,6] (last dim == 6); "
                       << "re-export with end-to-end head";
            return false;
        }

        if (cudaStreamCreate(&stream) != cudaSuccess) {
            LOG(ERROR) << "[YoloTrt] cudaStreamCreate failed";
            return false;
        }

        const size_t in_bytes = binding_sizes[static_cast<size_t>(input_index)];
        const size_t out_bytes = binding_sizes[static_cast<size_t>(output_index)];
        if (in_bytes != static_cast<size_t>(3 * opt.imgsz * opt.imgsz) * sizeof(float)) {
            LOG(ERROR) << "[YoloTrt] input bytes " << in_bytes << " != imgsz " << opt.imgsz;
            return false;
        }
        void *pinned_out = nullptr;
        if (cudaHostAlloc(&pinned_out, out_bytes, cudaHostAllocDefault) != cudaSuccess) {
            LOG(ERROR) << "[YoloTrt] cudaHostAlloc failed";
            return false;
        }
        host_output = static_cast<float *>(pinned_out);
        if (cudaEventCreate(&pre_start) != cudaSuccess || cudaEventCreate(&pre_end) != cudaSuccess ||
            cudaEventCreate(&trt_end) != cudaSuccess) {
            LOG(ERROR) << "[YoloTrt] cudaEventCreate failed";
            return false;
        }
        for (size_t i = 0; i < tensor_names.size(); ++i) {
            if (!context->setTensorAddress(tensor_names[i].c_str(), bindings[i])) {
                LOG(ERROR) << "[YoloTrt] setTensorAddress failed: " << tensor_names[i];
                return false;
            }
        }

        // One-shot warmup (same as IQIS): prime CUDA kernels; not done every frame.
        {
            cv::Mat dummy = cv::Mat::zeros(opt.imgsz, opt.imgsz, CV_8UC3);
            (void)Infer(dummy);
        }

        // 输入输出显存地址已固定。把 enqueueV3 的 kernel 提交录成图，之后每帧只重放。
        if (!CaptureCudaGraph())
            LOG(WARNING) << "[YoloTrt] CUDA graph unavailable, using enqueueV3 each frame";
        LOG(INFO) << "[YoloTrt] ready (YOLO26 E2E), candidates=" << num_candidates
                  << ", cuda_graph=" << (graph_exec ? 1 : 0);
        return true;
    }

    bool CaptureCudaGraph() {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            LOG(WARNING) << "[YoloTrt] sync before graph capture failed: " << cudaGetErrorString(err);
            return false;
        }
        err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        if (err != cudaSuccess) {
            LOG(WARNING) << "[YoloTrt] cudaStreamBeginCapture failed: " << cudaGetErrorString(err);
            return false;
        }
        const bool enqueued = context->enqueueV3(stream);
        cudaGraph_t graph = nullptr;
        err = cudaStreamEndCapture(stream, &graph);
        if (!enqueued || err != cudaSuccess || graph == nullptr) {
            if (graph)
                cudaGraphDestroy(graph);
            LOG(WARNING) << "[YoloTrt] CUDA graph capture failed: " << cudaGetErrorString(err);
            return false;
        }
        err = cudaGraphInstantiateWithFlags(&graph_exec, graph, 0);
        cudaGraphDestroy(graph);
        if (err != cudaSuccess || graph_exec == nullptr) {
            graph_exec = nullptr;
            LOG(WARNING) << "[YoloTrt] cudaGraphInstantiate failed: " << cudaGetErrorString(err);
            return false;
        }
        LOG(INFO) << "[YoloTrt] CUDA graph captured";
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
        // TRT 10: explicit batch is default; NetworkDefinitionCreationFlag::kEXPLICIT_BATCH removed/deprecated.
        auto network =
            std::unique_ptr<nvinfer1::INetworkDefinition, TrtDestroyer>(builder->createNetworkV2(0));
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig, TrtDestroyer>(builder->createBuilderConfig());
        auto parser =
            std::unique_ptr<nvonnxparser::IParser, TrtDestroyer>(nvonnxparser::createParser(*network, logger));
        if (!parser->parseFromFile(onnx.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            LOG(ERROR) << "[YoloTrt] parse ONNX failed: " << onnx;
            return false;
        }
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256ULL << 20);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        // kFP16 still works on TRT 10; strongly-typed path is optional for this detector.
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
#pragma GCC diagnostic pop
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
        if (!context || !stream || !host_output || bgr.empty())
            return {};
        if (bgr.type() != CV_8UC3) {
            LOG(WARNING) << "[YoloTrt] expect CV_8UC3, type=" << bgr.type();
            return {};
        }
        using Clock = std::chrono::steady_clock;

        const cv::Mat src = bgr.isContinuous() ? bgr : bgr.clone();
        const int ow = src.cols;
        const int oh = src.rows;
        const double scale = std::min(static_cast<double>(opt.imgsz) / ow, static_cast<double>(opt.imgsz) / oh);
        const int nw = static_cast<int>(ow * scale);
        const int nh = static_cast<int>(oh * scale);
        const int pad_w = (opt.imgsz - nw) / 2;
        const int pad_h = (opt.imgsz - nh) / 2;
        const size_t src_bytes = static_cast<size_t>(ow) * static_cast<size_t>(oh) * 3;
        if (src_bytes > bgr_dev_bytes) {
            if (bgr_dev)
                cudaFree(bgr_dev);
            void *dev = nullptr;
            if (cudaMalloc(&dev, src_bytes) != cudaSuccess) {
                bgr_dev = nullptr;
                bgr_dev_bytes = 0;
                LOG(ERROR) << "[YoloTrt] cudaMalloc bgr failed, bytes=" << src_bytes;
                return {};
            }
            bgr_dev = static_cast<uint8_t *>(dev);
            bgr_dev_bytes = src_bytes;
        }

        cudaEventRecord(pre_start, stream);
        cudaMemcpyAsync(bgr_dev, src.data, src_bytes, cudaMemcpyHostToDevice, stream);
        LaunchYoloLetterbox(bgr_dev, ow, oh, static_cast<float *>(bindings[static_cast<size_t>(input_index)]),
                            opt.imgsz, pad_w, pad_h, nw, nh, stream);
        cudaEventRecord(pre_end, stream);

        if (graph_exec) {
            if (cudaGraphLaunch(graph_exec, stream) != cudaSuccess) {
                LOG(WARNING) << "[YoloTrt] cudaGraphLaunch failed";
                return {};
            }
        } else if (!context->enqueueV3(stream)) {
            LOG(WARNING) << "[YoloTrt] enqueueV3 failed";
            return {};
        }

        cudaMemcpyAsync(host_output, bindings[static_cast<size_t>(output_index)],
                        binding_sizes[static_cast<size_t>(output_index)], cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(trt_end, stream);
        cudaStreamSynchronize(stream);

        const auto t2 = Clock::now();

        // YOLO26 E2E: row-major [N, 6] = x1,y1,x2,y2,score,cls (letterbox coords)
        const int out_floats =
            static_cast<int>(binding_sizes[static_cast<size_t>(output_index)] / sizeof(float));
        const int n = std::min(num_candidates, out_floats / 6);
        std::vector<Detection> dets;
        dets.reserve(64);
        const cv::Rect img_roi(0, 0, bgr.cols, bgr.rows);
        for (int i = 0; i < n; ++i) {
            const float *row = host_output + i * 6;
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
        float pre_ms = 0.f, trt_ms = 0.f;
        cudaEventElapsedTime(&pre_ms, pre_start, pre_end);
        cudaEventElapsedTime(&trt_ms, pre_end, trt_end);
        last_timing.pre_ms = pre_ms;
        last_timing.trt_ms = trt_ms;
        last_timing.post_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        return dets;
    }

    ~Impl() {
        if (graph_exec)
            cudaGraphExecDestroy(graph_exec);
        for (void *p : bindings)
            if (p)
                cudaFree(p);
        if (bgr_dev)
            cudaFree(bgr_dev);
        if (host_output)
            cudaFreeHost(host_output);
        if (pre_start)
            cudaEventDestroy(pre_start);
        if (pre_end)
            cudaEventDestroy(pre_end);
        if (trt_end)
            cudaEventDestroy(trt_end);
        if (stream)
            cudaStreamDestroy(stream);
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
