#include <cpu_bbs3d/bbs3d.hpp>
#include <pointcloud_iof/pcd_loader_without_pcl.hpp>
#include <glog/logging.h>
#include <algorithm>
#include <iterator>
#include "common/timer.h"
namespace lightning
{
    class VoxelMapSaver
    {
    public:
        VoxelMapSaver(pcl::PointCloud<pcl::PointXYZI>::Ptr &target_point, std::string &target_pcd_folder,
                      double &min_level_res, int &max_level)
        {
            LOG(INFO) << get_time_us() << "[INFO][VoxelMapSaver::" << __FUNCTION__ << "] "
                      << "Loading target pcds...";

            std::vector<Eigen::Vector3d> tar_points_d;

            tar_points_d.reserve(target_point->size());

            for (const auto &point : target_point->points)
            {
                // 过滤掉可能存在的 NaN 或 Inf 无效点，确保数据安全
                if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
                {
                    continue;
                }

                // 一步到位，直接写成 double 类型
                tar_points_d.emplace_back(
                    static_cast<double>(point.x),
                    static_cast<double>(point.y),
                    static_cast<double>(point.z));
            }

            LOG(INFO) << get_time_us() << "[INFO][VoxelMapSaver::" << __FUNCTION__ << "] "
                      << "Creating hierarchical voxel map...";
            std::unique_ptr<cpu::BBS3D> bbs3d_ptr(new cpu::BBS3D);
            bbs3d_ptr->set_tar_points(tar_points_d, min_level_res, max_level);

            LOG(INFO) << get_time_us() << "[INFO][VoxelMapSaver::" << __FUNCTION__ << "] "
                      << "Saving voxel maps...";
            bbs3d_ptr->save_voxel_params(target_pcd_folder);
            bbs3d_ptr->save_voxelmaps_pcd(target_pcd_folder);
            LOG(INFO) << get_time_us() << "[INFO][VoxelMapSaver::" << __FUNCTION__ << "] "
                      << "Saving voxel maps success";
        }
    };
}