#pragma once

#include <cmath>
#include <memory>
#include <string>

#include "common/eigen_types.h"
#include "common/point_def.h"

namespace lightning::bbs {

class BBS3DRunner {
public:
    struct Options {
        std::string backend;
        double min_level_res;
        int max_level;
        float map_leaf_size;
        float src_leaf_size;
        Eigen::Vector3d min_rpy;
        Eigen::Vector3d max_rpy;
        double score_threshold_percentage;
        int timeout_msec;
        double min_scan_range;
        double max_scan_range;
        double min_z;
        double max_z;
        bool use_gicp;

        Options()
            : backend("auto"),
              min_level_res(1.0),
              max_level(6),
              map_leaf_size(0.2f),
              src_leaf_size(0.2f),
              min_rpy(-0.02, -0.02, 0.0),
              max_rpy(0.02, 0.02, 2.0 * M_PI),
              score_threshold_percentage(0.7),
              timeout_msec(0),
              min_scan_range(0.0),
              max_scan_range(100.0),
              min_z(-2.0),
              max_z(2.0),
              use_gicp(false) {}
    };

    explicit BBS3DRunner(const Options &options = Options());
    ~BBS3DRunner();

    bool BuildTargetVoxelMap(const CloudPtr &target_cloud);
    bool BuildTargetVoxelMap(const std::string &tar_path);
    void BuildGicpTarget(const CloudPtr &target_cloud);
    bool EstimatePose(const CloudPtr &source_cloud, SE3 &estimated_pose, double *score = nullptr,
                      double *elapsed_ms = nullptr);
    bool IsReady() const;
    const std::string &ActiveBackend() const { return active_backend_; }

    static bool CudaDeviceAvailable();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    std::string active_backend_;
};

}  // namespace lightning::bbs
