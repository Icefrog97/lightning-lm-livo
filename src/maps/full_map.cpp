//
// Created by yx-liang on 26-02-26.
//

#include "maps/full_map.h"
#include <pcl/io/pcd_io.h>

namespace lightning
{
    // TODO: 添加从动态地图更新静态地图功能

    void FullMap::SaveToBin(bool only_dynamic)
    {
        LOG(INFO) << get_time_us() << "[INFO][FullMap::" << __FUNCTION__ << "] "
                  << "func_points_ is saved to " << options_.map_path_ << ", sz: " << func_points_.size();

        if (!only_dynamic)
        {
            std::ofstream fout(options_.map_path_ + "index.txt");

            // 保留切片地图写法，这里只使用单块全量地图
            auto it_static = static_chunks_.find(Vec2i(0, 0));
            if (it_static != static_chunks_.end() && !it_static->second->cloud_->empty())
            {
                std::string filename = options_.map_path_ + options_.global_loop_map_name_ + ".pcd";
                fout << "0 0 0 " << filename << std::endl;
            }

            fout << "# functional points" << std::endl;
            for (const auto &fp : func_points_)
            {
                auto t = fp.pose_.translation();
                auto q = fp.pose_.unit_quaternion();
                fout << fp.name_ << " " << t[0] << " " << t[1] << " " << t[2] << " "
                     << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
            }

            fout.close();
            return;
        }

        if (options_.policy_ == FullMap::DynamicCloudPolicy::PERSISTENT &&
            options_.save_dyn_when_quit_)
        {
            auto it_dyn = dynamic_chunks_.find(Vec2i(0, 0));
            if (it_dyn != dynamic_chunks_.end() && it_dyn->second->cloud_ != nullptr && !it_dyn->second->cloud_->empty())
            {
                std::string filename = options_.map_path_ + "dynamic_map.pcd";
                it_dyn->second->cloud_->width = it_dyn->second->cloud_->size();
                it_dyn->second->cloud_->height = 1;

                pcl::io::savePCDFileBinaryCompressed(filename,
                                                     *math::VoxelGrid(it_dyn->second->cloud_, options_.voxel_size_in_chunk_));

                LOG(INFO) << get_time_us() << "[INFO][FullMap::" << __FUNCTION__ << "] " << "Dynamic map saved to: " << filename;
            }
        }
    }

    bool FullMap::LoadMap()
    {
        std::string index_path = options_.map_path_ + "index.txt";
        std::string pcd_path = options_.map_path_ + options_.global_loop_map_name_ + ".pcd";

        std::ifstream fin(index_path);
        if (!fin)
        {
            LOG(ERROR) << get_time_us() << "[ERROR][FullMap::" << __FUNCTION__ << "] " << "cannot load map index from: " << index_path;
            return false;
        }

        LOG(INFO) << get_time_us() << "[INFO][FullMap::" << __FUNCTION__ << "] " << "loading maps";

        bool first_line = true;
        bool reading_fp = false;

        UL lock(static_data_mutex_);
        UL lock2(dynamic_data_mutex_);

        std::string line;
        while (std::getline(fin, line))
        {
            if (line.empty())
                continue;

            std::stringstream ss(line);

            if (line == "# functional points")
            {
                reading_fp = true;
                continue;
            }

            if (reading_fp)
            {
                /// 读功能点
                std::string name;
                double data[7];
                ss >> name;
                for (int i = 0; i < 7; ++i)
                    ss >> data[i];

                func_points_.emplace_back(name,
                                          SE3(Quatd(data[6], data[3], data[4], data[5]), Vec3d(data[0], data[1], data[2])));
            }
            else
            {
                // 保留加载瓦片地图的逻辑
                int id = 0;
                Vec2i grid(0, 0);

                auto ck = std::make_shared<MapChunk>(id, grid, pcd_path);
                ck->LoadCloud();
                static_chunks_.emplace(grid, ck);
                id_to_grid_.emplace(id, grid);

                if (options_.policy_ == DynamicCloudPolicy::PERSISTENT)
                {
                    if (options_.load_dyn_cloud_)
                    {
                    // 动态图层策略为永久时，从硬盘中读取动态区域信息
                    // 看是否有该栅格的动态图层
                    std::string dyn_filename = options_.map_path_ + "global_down_map_dyn.pcd";

                    LOG(INFO) << get_time_us() << "[INFO][FullMap::" << __FUNCTION__ << "] " << "load dynamic chunk: " << dyn_filename;
                    auto dyn_chunk = std::make_shared<MapChunk>(id, grid, dyn_filename);
                    dynamic_chunks_.emplace(grid, dyn_chunk);

                    
                        /// 从硬盘中读点云
                        dyn_chunk->LoadCloud();
                    }
                }
            }
        }

        LOG(INFO) << get_time_us() << "[INFO][FullMap::" << __FUNCTION__ << "] " << "loaded fps: " << func_points_.size();
        fin.close();

        return true;
    }

    CloudPtr FullMap::GetAllMap()
    {
        CloudPtr cloud_out(new PointCloudType);

        {
            UL lock_static(static_data_mutex_);
            auto iter = static_chunks_.find(Vec2i(0, 0));
            if (iter != static_chunks_.end() && iter->second->cloud_ != nullptr)
                *cloud_out += *(iter->second->cloud_);
        }

        {
            UL lock_dyn(dynamic_data_mutex_);
            // 在 FullMap 中，如果也是对应 (0,0) 瓦片
            auto iter_dyn = dynamic_chunks_.find(Vec2i(0, 0));
            if (iter_dyn != dynamic_chunks_.end() && iter_dyn->second->cloud_ != nullptr)
            {
                // 如果动态图层没加载，先加载
                if (!iter_dyn->second->loaded_)
                    iter_dyn->second->LoadCloud();

                *cloud_out += *(iter_dyn->second->cloud_);
            }
        }

        if (cloud_out->empty())
        {
            LOG(ERROR) << get_time_us() << "[ERROR][FullMap::" << __FUNCTION__ << "] " << "Global map (Static + Dynamic) is empty!";
            return nullptr;
        }

        return cloud_out;
    }

    // TODO: 添加动态物体剔除的功能后才能使用更新动态地图的功能。
    // 在 AddPoint 之前，判断新点与老点是否在同一体素内，如果是，则更新点的“寿命”；如果某个体素长时间没被观测到，则删除
    void FullMap::UpdateDynamicCloud(CloudPtr cloud_world, bool remove_old)
    {
        UL lock(dynamic_data_mutex_);

        Vec2i global_grid(0, 0);
        auto iter = dynamic_chunks_.find(global_grid);

        if (iter == dynamic_chunks_.end())
        {
            auto new_chunk = std::make_shared<MapChunk>(0, global_grid, options_.map_path_ + "dynamic_map.pcd");
            dynamic_chunks_.emplace(global_grid, new_chunk);
            iter = dynamic_chunks_.find(global_grid);
        }

        auto &dyn_chunk = iter->second;
        if (!dyn_chunk->loaded_)
            dyn_chunk->LoadCloud();
        // TODO: 添加地图点改成静态地图一样维护一个 voxel
        for (const auto &pt : cloud_world->points)
        {
            dyn_chunk->AddPoint(pt);
            dynamic_map_updated_ = true;
        }

        if (dyn_chunk->cloud_->size() > 1.5 * options_.max_pts_in_dyn_chunk_)
            dyn_chunk->cloud_ = math::VoxelGrid(dyn_chunk->cloud_, 0.5);

        size_t full_size = dyn_chunk->cloud_->size();
        if (remove_old && full_size > options_.max_pts_in_dyn_chunk_)
        {
            size_t erase_count = static_cast<size_t>(0.7 * full_size);
            dyn_chunk->cloud_->points.erase(
                dyn_chunk->cloud_->points.begin(),
                dyn_chunk->cloud_->points.begin() + erase_count);
            dyn_chunk->cloud_->width = dyn_chunk->cloud_->size();
            dyn_chunk->cloud_->points.shrink_to_fit();
        }
    }

    // TODO: 看下 ResetDynamicCloud 有没有用
    
}