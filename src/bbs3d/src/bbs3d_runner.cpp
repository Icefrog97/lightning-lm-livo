#include "bbs3d_runner.h"

#include "run_cpu.h"

#include <algorithm>
#include <cctype>
#include <glog/logging.h>

#include "common/timer.h"

#if defined(LIDAR_SLAM_USE_CUDA)
#include "run_gpu.h"
#include <cuda_runtime_api.h>
#endif

namespace lightning::bbs {
namespace {

std::string Lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

template <typename DstOpt>
DstOpt CopyCommonOptions(const BBS3DRunner::Options &src) {
    DstOpt dst;
    dst.min_level_res = src.min_level_res;
    dst.max_level = src.max_level;
    dst.map_leaf_size = src.map_leaf_size;
    dst.src_leaf_size = src.src_leaf_size;
    dst.min_rpy = src.min_rpy;
    dst.max_rpy = src.max_rpy;
    dst.score_threshold_percentage = src.score_threshold_percentage;
    dst.timeout_msec = src.timeout_msec;
    dst.min_scan_range = src.min_scan_range;
    dst.max_scan_range = src.max_scan_range;
    dst.min_z = src.min_z;
    dst.max_z = src.max_z;
    dst.use_gicp = src.use_gicp;
    return dst;
}

}  // namespace

class BBS3DRunner::Impl {
public:
    explicit Impl(const Options &options) : options_(options) {}

    bool UseGpu() const { return use_gpu_; }

    void SetUseGpu(bool use_gpu) {
        use_gpu_ = use_gpu;
        if (use_gpu_) {
#if defined(LIDAR_SLAM_USE_CUDA)
            gpu_ = std::make_unique<gpu::BBS3DRunner>(CopyCommonOptions<gpu::BBS3DRunner::Options>(options_));
#endif
        } else {
            cpu_ = std::make_unique<cpu::BBS3DRunner>(CopyCommonOptions<cpu::BBS3DRunner::Options>(options_));
        }
    }

    bool BuildTargetVoxelMap(const CloudPtr &target_cloud) {
#if defined(LIDAR_SLAM_USE_CUDA)
        if (use_gpu_)
            return gpu_->BuildTargetVoxelMap(target_cloud);
#endif
        return cpu_->BuildTargetVoxelMap(target_cloud);
    }

    bool BuildTargetVoxelMap(const std::string &tar_path) {
#if defined(LIDAR_SLAM_USE_CUDA)
        if (use_gpu_)
            return gpu_->BuildTargetVoxelMap(tar_path);
#endif
        return cpu_->BuildTargetVoxelMap(tar_path);
    }

    void BuildGicpTarget(const CloudPtr &target_cloud) {
#if defined(LIDAR_SLAM_USE_CUDA)
        if (use_gpu_) {
            gpu_->BuildGicpTarget(target_cloud);
            return;
        }
#endif
        cpu_->BuildGicpTarget(target_cloud);
    }

    bool EstimatePose(const CloudPtr &source_cloud, SE3 &estimated_pose, double *score, double *elapsed_ms) {
#if defined(LIDAR_SLAM_USE_CUDA)
        if (use_gpu_)
            return gpu_->EstimatePose(source_cloud, estimated_pose, score, elapsed_ms);
#endif
        return cpu_->EstimatePose(source_cloud, estimated_pose, score, elapsed_ms);
    }

    bool IsReady() const {
#if defined(LIDAR_SLAM_USE_CUDA)
        if (use_gpu_)
            return gpu_->IsReady();
#endif
        return cpu_->IsReady();
    }

private:
    Options options_;
    bool use_gpu_ = false;
#if defined(LIDAR_SLAM_USE_CUDA)
    std::unique_ptr<gpu::BBS3DRunner> gpu_;
#endif
    std::unique_ptr<cpu::BBS3DRunner> cpu_;
};

bool BBS3DRunner::CudaDeviceAvailable() {
#if defined(LIDAR_SLAM_USE_CUDA)
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
#else
    return false;
#endif
}

BBS3DRunner::BBS3DRunner(const Options &options) : impl_(std::make_unique<Impl>(options)) {
    const std::string want = Lower(options.backend);
    const bool cuda_ok = CudaDeviceAvailable();
    bool use_gpu = false;
    if (want == "cpu") {
        use_gpu = false;
    } else if (want == "gpu") {
        if (cuda_ok) {
            use_gpu = true;
        } else {
            LOG(WARNING) << get_time_us() << "[WARN][BBS3DRunner] bbs_backend=gpu but no CUDA device; fallback to cpu";
        }
    } else {
        use_gpu = cuda_ok;
        if (want != "auto")
            LOG(WARNING) << get_time_us() << "[WARN][BBS3DRunner] unknown bbs_backend='" << options.backend
                         << "', treating as auto";
    }
#if !defined(LIDAR_SLAM_USE_CUDA)
    if (use_gpu)
        LOG(WARNING) << get_time_us() << "[WARN][BBS3DRunner] GPU BBS was not compiled; using cpu";
    use_gpu = false;
#endif
    impl_->SetUseGpu(use_gpu);
    active_backend_ = use_gpu ? "gpu" : "cpu";
    LOG(INFO) << get_time_us() << "[INFO][BBS3DRunner] 3D-BBS backend=" << active_backend_
              << " (requested=" << options.backend << ", cuda_device=" << (cuda_ok ? "yes" : "no") << ")";
}

BBS3DRunner::~BBS3DRunner() = default;

bool BBS3DRunner::BuildTargetVoxelMap(const CloudPtr &target_cloud) {
    return impl_->BuildTargetVoxelMap(target_cloud);
}

bool BBS3DRunner::BuildTargetVoxelMap(const std::string &tar_path) {
    return impl_->BuildTargetVoxelMap(tar_path);
}

void BBS3DRunner::BuildGicpTarget(const CloudPtr &target_cloud) {
    impl_->BuildGicpTarget(target_cloud);
}

bool BBS3DRunner::EstimatePose(const CloudPtr &source_cloud, SE3 &estimated_pose, double *score, double *elapsed_ms) {
    return impl_->EstimatePose(source_cloud, estimated_pose, score, elapsed_ms);
}

bool BBS3DRunner::IsReady() const { return impl_->IsReady(); }

}  // namespace lightning::bbs
