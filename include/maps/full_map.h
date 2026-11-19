//
// Created by yx-liang on 26-02-26.
//

#ifndef LIDARSLAM_FULL_MAP_H
#define LIDARSLAM_FULL_MAP_H

#include <set>
#include <map>
#include <unordered_map>

#include "common/functional_points.h"
#include "common/point_def.h"
#include "lightning_math.hpp"
#include "maps/tiled_map_chunk.h"

namespace lightning
{
    class FullMap
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using UL = std::unique_lock<std::mutex>;

        enum class DynamicCloudPolicy
        {
            /// 短时间保留，车辆移出该区域就会清空
            SHORT = 0,

            /// 长时间保留在内存，车辆移出区域后仍然保留，但不会存盘并在下次开机时使用
            LONG,

            /// 更长时间的保留，会存盘，下次运行也会读取，超过一定时间后才会清除
            PERSISTENT,
        };

        struct Options
        {
            Options() {}
            std::string map_path_ = "./data/maps/";                      // 地图存储路径
            std::string global_loop_map_name_ = "";
            float chunk_size_ = 100.0;                                   // 单个chunk大小
            float inv_chunk_size_ = 1.0 / 100.0;                         // 反向chunk size
            float voxel_size_in_chunk_ = 0.1;                            // chunk内部的分辨率
            const int load_nearby_chunks_ = 8;                           // 载入邻近范围内的点云
            bool save_dynamic_layer_ = false;                            // 是否将动态图层也存入地图文件夹
            size_t max_pts_in_dyn_chunk_ = 50000;                        // 每个动态区块中最多点数
            DynamicCloudPolicy policy_ = DynamicCloudPolicy::PERSISTENT; // 动态区域处理策略

            bool load_dyn_cloud_ = true;     // 初始化时是否加载动态点云
            bool save_dyn_when_quit_ = true; // 退出时是否保存动态点云
        };

        enum class NearbyType
        {
            CENTER,
            NEARBY6,
        };

        struct NdtMapOptions
        {
            NdtMapOptions() {}
            float static_cloud_weight_ = 1.0;             // 静态点云权重
            float dynamic_cloud_weight_ = 0.1;            // 动态点云权重
            NearbyType nearby_type_ = NearbyType::CENTER; // 邻近类型
            float voxel_size_ = 1.0;                      // voxel大小
            int max_iteration_ = 5;                       // 最大迭代次数
            double eps_ = 1e-2;                           // 收敛判定条件
            double res_outlier_th_ = 50.0;                // 异常值拒绝阈值
            float inv_voxel_size_ = 1.0;                  // 栅格尺寸倒数
            int min_pts_in_voxel_ = 3;                    // 每个栅格中最小点数
            int min_effective_pts_ = 500;                 // 最小有效点数
            size_t capacity_ = 100000;                    // 缓存的体素数量
            int max_pts_in_voxel_ = 50;                   // 每个栅格中最大点数

            bool verbose_ = false;
        };

        using KeyType = Eigen::Matrix<int, 3, 1>; // 体素的索引
        struct VoxelData
        {
            VoxelData() {}
            VoxelData(size_t id) { idx_.emplace_back(id); }
            VoxelData(const Vec3f &pt)
            {
                pts_.emplace_back(pt);
                num_pts_ = 1;
            }

            void AddPoint(const Vec3f &pt)
            {
                pts_.emplace_back(pt);
                if (!ndt_estimated_)
                {
                    num_pts_++;
                }
            }

            std::vector<Vec3f> pts_;          // 内部点，多于一定数量之后再估计均值和协方差
            std::vector<size_t> idx_;         // 点云中点的索引
            Vec3f mu_ = Vec3f::Zero();        // 均值
            Mat3f sigma_ = Mat3f::Identity(); // 协方差
            Mat3f info_ = Mat3f::Zero();      // 协方差之逆

            bool ndt_estimated_ = false; // NDT是否已经估计
            int num_pts_ = 0;            // 总共的点数，用于更新估计
        };

        using GridHashMap = std::unordered_map<KeyType, VoxelData, math::hash_vec<3>>;
        using ChunkHashMap = std::unordered_map<Vec2i, std::shared_ptr<MapChunk>, math::hash_vec<2>>;

        FullMap(Options options = Options()) : options_(options) {}

        /// 载入地图索引文件
        bool LoadMap();

        /// 获取拼接好的地图
        CloudPtr GetAllMap();

        /**
         * 使用最近配准之后的scan更新动态图层
         * @param cloud_world       世界坐标系下点云
         * @param remove_old_pts    是否会移除历史的数据
         */
        void UpdateDynamicCloud(CloudPtr cloud_world, bool remove_old_pts = true);

        /// accessors
        /// 获取地图原点
        Vec3d GetOrigin() const { return origin_; }

        /// 添加地图功能点
        void AddFP(const FunctionalPoint &fp) { func_points_.emplace_back(fp); }

        /// 获取所有功能点
        std::vector<FunctionalPoint> GetAllFP() const { return func_points_; }

        bool DynamicMapUpdated() const { return dynamic_map_updated_; }

        bool MapUpdated() const { return map_updated_; }

        void CleanMapUpdate()
        {
            dynamic_map_updated_ = false;
            map_updated_ = false;
        }

        /// 存储到二进制pcd文件
        void SaveToBin(bool only_dynamic = false);

        template <class T>
        void SetNewTargetForNDT(T ndt)
        {
            bool has_cloud = false;
            {
                UL lock(static_data_mutex_);

                if (!static_chunks_[Vec2i(0, 0)]->cloud_->empty())
                {
                    has_cloud = true;
                }
                ndt->AddTarget(static_chunks_[Vec2i(0, 0)]->cloud_);
            }

            {
                UL lock(dynamic_data_mutex_);

                if (dynamic_chunks_.find(Vec2i(0, 0)) != dynamic_chunks_.end())
                {
                    if (dynamic_chunks_[Vec2i(0, 0)]->cloud_ != nullptr)
                    {
                        has_cloud = true;
                        ndt->AddTarget(dynamic_chunks_[Vec2i(0, 0)]->cloud_);
                    }
                }
            }

            ndt->ComputeTargetGrids();
            if (has_cloud)
            {
                ndt->initCompute();
            }
        }

        bool map_updated_ = false;

    private:
        /// 注意这里有个边长
        inline Vec2i Pos2Grid(const Vec2f &pt) const
        {
            Vec2d p = ((pt.cast<double>() * options_.inv_chunk_size_) + Vec2d(0.5, 0.5));
            return Vec2i(floor(p[0]), floor(p[1]));
        }

        inline Vec2i Pos2Grid(const Vec2d &pt) const
        {
            Vec2d p = ((pt * options_.inv_chunk_size_) + Vec2d(0.5, 0.5));
            return Vec2i(floor(p[0]), floor(p[1]));
        }

        Options options_;
        NdtMapOptions ndt_map_options_;

        std::mutex static_data_mutex_;  // 静态数据锁
        std::mutex dynamic_data_mutex_; // 动态数据锁

        Vec3d origin_ = Vec3d::Zero();
        ChunkHashMap static_chunks_;      // 静态地图
        ChunkHashMap dynamic_chunks_;     // 动态地图
        std::map<int, Vec2i> id_to_grid_; // 从ID找grid

        Vec2i last_load_grid_ = Vec2i::Zero(); // 上次加载时的网格
        bool last_load_grid_set_ = false;      // 上次加载的flag

        bool dynamic_map_updated_ = false;

        std::vector<FunctionalPoint> func_points_; // 功能点

        bool flag_first_dynamic_scan_ = true; // 首帧动态点云特殊处理
        bool flag_first_static_scan_ = true;  // 首帧静态点云特殊处理
    };
}

#endif