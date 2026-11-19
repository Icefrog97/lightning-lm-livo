#include <algorithm>
#include <chrono>
#include <execution>

#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/pcl_base.h>
#include <pcl/registration/ndt.h>

#include "localization/pclomp/ndt_omp_impl.hpp"
#include "localization/pclomp/voxel_grid_covariance_omp_impl.hpp"

#include "localization/lidar_loc.h"
#include <glog/logging.h>
#include "pub/pub_ROS1.h"

namespace lightning::loc
{
    LidarLoc::LidarLoc(LidarLoc::Options options) : options_(options)
    {
        pcl_ndt_.reset(new NDTType());
        pcl_ndt_->setResolution(1.0);
        pcl_ndt_->setNeighborhoodSearchMethod(pclomp::DIRECT7);
        pcl_ndt_->setStepSize(0.1);
        pcl_ndt_->setOulierRatio(0.45);
        pcl_ndt_->setTransformationEpsilon(0.01);
        pcl_ndt_->setMaximumIterations(20);
        pcl_ndt_->setNumThreads(2);

        pcl_ndt_rough_.reset(new NDTType());
        pcl_ndt_rough_->setResolution(5.0);
        pcl_ndt_rough_->setNeighborhoodSearchMethod(pclomp::DIRECT7);
        pcl_ndt_rough_->setStepSize(0.1);
        pcl_ndt_rough_->setMaximumIterations(10);
        pcl_ndt_rough_->setNumThreads(2);

        pcl_icp_.reset(new ICPType());
        pcl_icp_->setMaximumIterations(4);
        pcl_icp_->setTransformationEpsilon(0.01);

        LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "match name is NDT_OMP"
                  << ", MaximumIterations is: " << pcl_ndt_->getMaximumIterations();
    }

    LidarLoc::~LidarLoc()
    {
        if (update_map_thread_.joinable())
        {
            update_map_quit_ = true;
            update_map_thread_.join();
        }

        recover_pose_out_.close();
    }

    bool LidarLoc::Init()
    {
        LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "min init confidence: " << options_.min_init_confidence_;

        map_ = std::make_shared<FullMap>(options_.map_option_);
        map_->LoadMap();

        if (options_.use_3d_bbs_)
        {
            lightning::bbs::BBS3DRunner::Options bbs_options;
            bbs_options.backend = options_.bbs_backend_;
            bbs_options.min_level_res = options_.bbs_min_level_res_;
            bbs_options.max_level = options_.bbs_max_level_;
            bbs_options.map_leaf_size = static_cast<float>(options_.bbs_map_leaf_size_);
            bbs_options.src_leaf_size = static_cast<float>(options_.bbs_src_leaf_size_);
            bbs_options.min_rpy = options_.bbs_min_rpy_;
            bbs_options.max_rpy = options_.bbs_max_rpy_;
            bbs_options.score_threshold_percentage = options_.bbs_score_threshold_percentage_;
            bbs_options.timeout_msec = options_.bbs_timeout_msec_;
            bbs_options.min_scan_range = options_.bbs_min_scan_range_;
            bbs_options.max_scan_range = options_.bbs_max_scan_range_;
            bbs_options.min_z = options_.bbs_min_z_;
            bbs_options.max_z = options_.bbs_max_z_;
            bbs_options.use_gicp = options_.bbs_use_gicp_;
            bbs3d_runner_ = std::make_unique<lightning::bbs::BBS3DRunner>(bbs_options);
        }

        /// load recover pose if exist
        if (options_.add_recover_to_index_)
        {
            std::ifstream fin(options_.recover_pose_path_);
            double data[7] = {0, 0, 0, 0, 0, 0, 1};
            for (int i = 0; i < 7; ++i)
                fin >> data[i];
            fin.close();

            SE3 pose(Quatd(data[6], data[3], data[4], data[5]), Vec3d(data[0], data[1], data[2]));
            FunctionalPoint fp_recover;
            fp_recover.name_ = "recover";
            fp_recover.pose_ = pose;
            // TODO: 这里添加后有没有剔除相近距离的点
            map_->AddFP(fp_recover);
        }

        auto fps = map_->GetAllFP();
        if (!fps.empty())
        {
            map_->map_updated_ = true;
            UpdateGlobalMap();
        }

        // update_map_thread_ = std::thread([this]()
        //                                  { LidarLoc::UpdateMapThread(); });

        return true;
    }

    bool LidarLoc::ProcessCloud(CloudPtr cloud_input, double timestamp)
    {
        assert(cloud_input != nullptr);
        auto t1 = std::chrono::steady_clock::now();
        if (cloud_input->empty() || cloud_input->size() < 50)
        {
            LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] " << "loc input is empty or invalid, sz: " << cloud_input->size();
            return false;
        }
        if (timestamp <= 0.0)
        {
            LOG(ERROR) << get_time_us() << "[ERROR][LidarLoc::" << __FUNCTION__ << "] "
                       << "invalid LIO end timestamp=" << timestamp << ", skip Align";
            return false;
        }

        CloudPtr cloud(new PointCloudType);
        pcl::VoxelGrid<PointType> voxel;

        float sz = 0.2;
        voxel.setLeafSize(sz, sz, sz);
        voxel.setInputCloud(cloud_input);
        voxel.filter(*cloud);

        current_scan_ = cloud;
        // current_scan_ = cloud_input;

        Align(current_scan_, timestamp);
        auto t2 = std::chrono::steady_clock::now();
        auto vo_time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        if (options_.enable_debug_log_)
            LOG(INFO) << get_time_us() << "[DEBUG][LidarLoc::" << __FUNCTION__ << "] "
                      << "Align use time: " << vo_time_used.count() << " s, "
                      << "cloud size: " << current_scan_->size();
        return true;
    }

    // TODO: 没有被调用
    NavState LidarLoc::GetState()
    {
        // TODO: 是不是可以把第一个锁单独放在一个域内
        UL lock_res(result_mutex_);
        NavState ns;
        ns.SetPose(current_abs_pose_);
        ns.timestamp_ = current_timestamp_;
        ns.confidence_ = current_score_;

        UL lock(lo_pose_mutex_);
        if (!lo_pose_queue_.empty())
        {
            auto s = lo_pose_queue_.back();
            ns.SetVel(s.GetVel());
        }

        return ns;
    }

    bool LidarLoc::ProcessDR(const NavState &state)
    {
        // 未初始化成功的数据不接收
        if (!state.pose_is_ok_)
            return false;

        // DR数据check
        UL lock(dr_pose_mutex_);
        if (!dr_pose_queue_.empty())
        {
            const double last_stamp = dr_pose_queue_.back().timestamp_;
            if (state.timestamp_ < last_stamp)
            {
                // LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] " << "当前相对定位的结果的时间戳应当比上一个时间戳数值大，实际相减得"
                //              << state.timestamp_ - last_stamp;
                return false;
            }
        }

        dr_pose_queue_.emplace_back(state);
        while (dr_pose_queue_.size() >= 200)
            dr_pose_queue_.pop_front();

        return true;
    }

    bool LidarLoc::ProcessLO(const NavState &state)
    {
        /// 理论上相对定位是按时间顺序到达的
        UL lock(lo_pose_mutex_);
        if (!lo_pose_queue_.empty())
        {
            const double last_stamp = lo_pose_queue_.back().timestamp_;
            if (state.timestamp_ < last_stamp)
            {
                LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] " << "当前相对定位的结果的时间戳应当比上一个时间戳数值大，实际相减得"
                             << state.timestamp_ - last_stamp;
                return false;
            }
        }

        lo_pose_queue_.emplace_back(state);

        while (lo_pose_queue_.size() >= 50)
            lo_pose_queue_.pop_front();

        // TODO: 添加 lidar_odom_reliable_ 的判定函数
        if (state.lidar_odom_reliable_ == false)
        {
            lo_reliable_ = false;
            lo_reliable_cnt_ = 10;
        }
        else
        {
            if (state.lidar_odom_reliable_ && lo_reliable_cnt_ > 0)
                lo_reliable_cnt_--;

            if (lo_reliable_cnt_ == 0)
                lo_reliable_ = true;
        }

        return true;
    }

    bool LidarLoc::YawSearch(SE3 &pose, double &confidence, CloudPtr input, CloudPtr output)
    {
        auto t1 = std::chrono::steady_clock::now();
        SE3 init_pose = pose;
        auto RPYXYZ = math::SE3ToRollPitchYaw(init_pose);
        double init_yaw = RPYXYZ.yaw;

        confidence = 0;
        bool yaw_search_success = false;

        int step = lidar_loc::grid_search_angle_step;
        double radius = lidar_loc::grid_search_angle_range * constant::kDEG2RAD;
        double angle_search_step = 2 * radius / step;

        std::vector<double> searched_yaw;
        std::vector<double> scores(step);
        std::vector<int> index;
        std::vector<SE3> pose_opti(step);

        for (int i = 0; i < step; ++i)
        {
            double search_yaw = init_yaw + i * angle_search_step - radius;
            searched_yaw.emplace_back(search_yaw);
            index.emplace_back(i);
        }

        if (options_.enable_debug_log_)
            LOG(INFO) << get_time_us() << "[DEBUG][LidarLoc::" << __FUNCTION__ << "] "
                      << "init yaw: " << init_yaw << ", p: " << RPYXYZ.pitch << ", ro: " << RPYXYZ.roll << ", x: "
                      << init_pose.translation().x() << ", y: " << init_pose.translation().y() << ", z: " << init_pose.translation().z();

        /// 粗分辨率
        std::for_each(index.begin(), index.end(), [&](int i)
                      {
        double fitness_score = 0;
        RPYXYZ.yaw = searched_yaw[i];
        SE3 pose_esti = math::XYZRPYToSE3(RPYXYZ);
        Localize(pose_esti, fitness_score, input, output, true);
        // Localize(pose_esti, fitness_score, input, output, false);
        scores[i] = fitness_score;
        pose_opti[i] = pose_esti; });

        // find best match
        auto best_score_idx = std::max_element(scores.begin(), scores.end()) - scores.begin();
        confidence = scores.at(best_score_idx);
        pose = pose_opti.at(best_score_idx);

        // Localize(pose, confidence, input, output, true);

        if (options_.enable_debug_log_)
            LOG(INFO) << get_time_us() << "[DEBUG][Localization::" << __FUNCTION__ << "] "
                      << "best score: " << confidence;

        /// 高分辨率
        if (confidence > options_.min_init_confidence_)
            Localize(pose, confidence, input, output, false);

        if (options_.enable_debug_log_)
            LOG(INFO) << get_time_us() << "[DEBUG][LidarLoc::" << __FUNCTION__ << "] "
                      << "t: " << pose.translation().transpose() << ", confidence: " << confidence << ", th: " << options_.min_init_confidence_;

        if (confidence > options_.min_init_confidence_)
        {
            Eigen::Vector3d suc_translation = pose.translation();
            Eigen::Matrix3d suc_rotation_matrix = pose.rotationMatrix();
            double suc_x = suc_translation.x();
            double suc_y = suc_translation.y();
            double suc_yaw = atan2(suc_rotation_matrix(1, 0), suc_rotation_matrix(0, 0));
            LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] "
                      << "localization init success, x: " << suc_x << ", y: " << suc_y << ", yaw: " << suc_yaw
                      << ", confidence: " << confidence << ", th = " << options_.min_init_confidence_;
            yaw_search_success = true;
        }
        auto t2 = std::chrono::steady_clock::now();
        auto all_time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        if (options_.enable_debug_log_)
            LOG(INFO) << get_time_us() << "[DEBUG][Localization::" << __FUNCTION__ << "] " << "time cost " << all_time_used.count();
        return yaw_search_success;
    }

    bool LidarLoc::InitWithFP(CloudPtr input, const SE3 &fp_pose)
    {
        assert(input != nullptr && !input->empty());

        // 使用功能点的位置进行定位初始化
        double fitness_score;
        SE3 pose_esti = fp_pose;
        CloudPtr output_cloud(new PointCloudType);
        loc_inited_ = YawSearch(pose_esti, fitness_score, input, output_cloud);
        // loc_inited_ = Localize(pose_esti, fitness_score, input, output_cloud);

        if (loc_inited_)
        {
            // current_timestamp_ already set by Align from LIO end time
            localization_result_.confidence_ = fitness_score;
            current_abs_pose_ = pose_esti;
            first_abs_pose_ = pose_esti;
            localization_result_.pose_ = pose_esti;
            localization_result_.timestamp_ = current_timestamp_;
            localization_result_.lidar_loc_valid_ = true;
            localization_result_.status_ = LocalizationStatus::GOOD;

            last_abs_pose_set_ = true;
            last_abs_pose_ = pose_esti;

            current_score_ = fitness_score;

            LOG(ERROR) << get_time_us() << "[ERROR][LidarLoc::" << __FUNCTION__ << "] "
                       << "Loc init success, init pose: " << last_abs_pose_.translation().transpose()
                       << ", fitness_score is: " << fitness_score;

            map_height_ = fp_pose.translation()[2];

            if (current_lo_pose_set_)
            {
                // 设置上一次的相对定位结果
                last_lo_pose_ = current_lo_pose_;
                last_lo_pose_set_ = true;

                last_dr_pose_ = current_dr_pose_;
                last_dr_pose_set_ = true;
            }

            //  定位成功，则清空失败记录
            fp_init_fail_pose_vec_.clear();
        }
        else
        {
            // 添加失败历史记录
            LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "init failed, score: " << fitness_score;
            fp_init_fail_pose_vec_.emplace_back(fp_pose);
            fp_last_tried_time_ = current_timestamp_;
        }
        return loc_inited_;
    }

    void LidarLoc::ResetLastPose(const SE3 &last_pose)
    {
        last_abs_pose_ = last_pose;

        // TODO：清空动态图层

        return;
    }

    bool LidarLoc::TryOtherSolution(CloudPtr input, SE3 &pose)
    {
        double fitness_score;
        SE3 pose_esti = pose;
        CloudPtr output_cloud(new PointCloudType);

        bool loc_success = Localize(pose_esti, fitness_score, input, output_cloud);

        if (loc_success)
        {
            // 激光重置逻辑
            float score_th = std::min(1.5 * current_score_, current_score_ + 0.3);
            if (fitness_score > score_th && fitness_score > 1.0)
            {
                // 显著好于现在的估计
                LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] "
                             << "alternate solution is significantly better: " << fitness_score << " " << current_score_;
                pose = pose_esti;
                localization_result_.lidar_loc_smooth_flag_ = false;
                return true;
            }
            else
            {
                LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "not using alternate solution: " << fitness_score << " " << current_score_;
                return false;
            }
        }
        return false;
    }

    bool LidarLoc::UpdateGlobalMap()
    {
        // 把 ndt 参数参数化出来，小场景的网格调小
        NDTType::Ptr ndt(new NDTType());
        ndt->setResolution(1.0);
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
        ndt->setStepSize(0.1);
        ndt->setMaximumIterations(4);
        ndt->setNumThreads(options_.ndt_omp_th_);

        map_->SetNewTargetForNDT(ndt);

        UL lock(match_mutex_);
        pcl_ndt_ = ndt;

        // TODO: 添加重定位功能时，触发重定位后需要执行该部分(全量地图可以不做处理)
        if (!loc_inited_)
        {
            NDTType::Ptr ndt_rough(new NDTType());
            ndt_rough->setResolution(5.0);
            ndt_rough->setNeighborhoodSearchMethod(pclomp::DIRECT7);
            ndt_rough->setStepSize(0.5);
            ndt_rough->setMaximumIterations(15);
            ndt_rough->setNumThreads(options_.ndt_omp_th_);

            map_->SetNewTargetForNDT(ndt_rough);

            pcl_ndt_rough_ = ndt_rough;
        }

        // 生成 3d-bbs voxel地图
        if (options_.use_3d_bbs_ && bbs3d_runner_)
        {
            LOG(INFO) << get_time_us() << "[INFO][LidarLoc::";
            if (options_.bbs_use_voxelmap_coord_ && bbs3d_runner_->BuildTargetVoxelMap(options_.bbs_target_clouds_))
            {
                LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] "
                          << "Building 3D-BBS target voxel map from path: " << options_.bbs_target_clouds_;
                bbs3d_runner_->BuildGicpTarget(map_->GetAllMap());
            }
            else if (!bbs3d_runner_->BuildTargetVoxelMap(map_->GetAllMap()))
                LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] "
                             << "failed to build in-memory 3d-bbs target voxel map";
        }

        if (options_.enable_icp_adjust_)
        {
            ICPType::Ptr icp(new ICPType());
            CloudPtr map_cloud(new PointCloudType);
            pcl::VoxelGrid<PointType> voxel;
            auto sz = 0.5;
            voxel.setLeafSize(sz, sz, sz);
            voxel.setInputCloud(map_->GetAllMap());
            voxel.filter(*map_cloud);
            icp->setInputTarget(map_cloud);
            icp->setMaximumIterations(4);
            icp->setTransformationEpsilon(0.01);
            pcl_icp_ = icp;
        }

        pub_->publishGlobalMap(map_->GetAllMap());

        return true;
    }

    void LidarLoc::UpdateMapThread()
    {
        LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "UpdateMapThread thread is running";
        // TODO: 全量地图且不使用动态地图的情况不开这个线程
        while (!update_map_quit_)
        {
            if (map_->MapUpdated() || map_->DynamicMapUpdated())
            {
                UpdateGlobalMap();

                map_->CleanMapUpdate();
            }
            usleep(10000);
        }
    }

    void LidarLoc::SetInitialPose(SE3 init_pose)
    {
        UL lock(initial_pose_mutex_);
        loc_inited_ = false;
        // map_->ClearMap();
        std::ifstream fin(options_.recover_pose_path_);
        double data[7] = {0, 0, 0, 0, 0, 0, 1};
        for (int i = 0; i < 7; ++i)
            fin >> data[i];
        fin.close();

        SE3 pose(Quatd(data[6], data[3], data[4], data[5]), Vec3d(data[0], data[1], data[2]));

        initial_pose_set_ = true;
        initial_pose_ = pose;
        LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "Set initial pose is: " << initial_pose_.translation().transpose();
    }

    void LidarLoc::Align(const CloudPtr &input, double timestamp)
    {
        assert(input != nullptr);

        // timestamp is LIO package end time (same as lo_state.timestamp_ / lidar_end_time_).
        double current_time = timestamp;
        current_timestamp_ = current_time;

        // LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] "
        //           << "current time: " << std::fixed << std::setprecision(12) << current_timestamp_;

        /// 设置当前帧对应的rel_pose
        if (!AssignLOPose(current_time))
            LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] " << "assign LO pose failed";

        if (!AssignDRPose(current_time))
            LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] " << "assign DR pose failed";

        /// 初始化处理
        if (!loc_inited_)
        {
            auto t1 = std::chrono::steady_clock::now();
            UL lock_init(initial_pose_mutex_);
            bool init_success = false;
            bool init_guess_pose_set = false;
            SE3 init_guess_pose;

            init_cloud_buffer_.push_back(input);
            if (init_cloud_buffer_.size() < 3)
            {
                LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "waiting init cloud buffer add";
                return;
            }

            CloudPtr dense_input(new PointCloudType);
            SE3 last_frame_pose;
            last_frame_pose = current_dr_pose_;

            for (auto &frame : init_cloud_buffer_)
            {
                *dense_input += *frame;
            }
            init_cloud_buffer_.clear();

            pcl::VoxelGrid<PointType> voxel;
            voxel.setLeafSize(0.2, 0.2, 0.2); // 0.2米采样对 Mid-360 比较合适
            CloudPtr downsampled_dense(new PointCloudType);
            voxel.setInputCloud(dense_input);
            voxel.filter(*downsampled_dense);

            LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "initing lidarloc";
            SetInitRltState();

            if (!init_success && options_.use_3d_bbs_ &&
                (!options_.use_3dbbs_just_reset_ || is_reset_))
            {
                if (!bbs3d_runner_ || !bbs3d_runner_->IsReady())
                {
                    LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] "
                                 << "skip 3d-bbs init because target voxel map is not ready";
                }
                else
                {
                    SE3 bbs_pose;
                    double bbs_score = 0.0;
                    double bbs_elapsed_ms = 0.0;
                    if (bbs3d_runner_->EstimatePose(downsampled_dense, bbs_pose, &bbs_score, &bbs_elapsed_ms))
                    {
                        Eigen::Matrix3d R_bbs = bbs_pose.rotationMatrix();
                        double yaw_bbs = atan2(R_bbs(1, 0), R_bbs(0, 0));
                        double pitch_bbs = asin(-R_bbs(2, 0));
                        double roll_bbs = atan2(R_bbs(2, 1), R_bbs(2, 2));
                        LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] "
                                  << "3d-bbs estimated pose, t: " << bbs_pose.translation().transpose()
                                  << ", rpy(rad): [" << roll_bbs << ", " << pitch_bbs << ", " << yaw_bbs << "]"
                                  << ", score: " << bbs_score << ", time(ms): " << bbs_elapsed_ms;
                        if (InitWithFP(downsampled_dense, bbs_pose))
                        {
                            LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] "
                                      << "inited success with 3d-bbs pose";
                            init_success = true;
                            init_guess_pose = bbs_pose;
                            init_guess_pose_set = true;
                        }
                        else
                        {
                            LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] "
                                         << "3d-bbs pose init failed in InitWithFP";
                        }
                    }
                    else
                    {
                        LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] "
                                     << "3d-bbs failed to estimate init pose";
                    }
                }
            }

            if (options_.init_with_recover_ && initial_pose_set_ && init_success == false)
            {
                LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "test inited with initial pose: " << initial_pose_.translation().transpose();
                /// 尝试在给定点初始化
                if (InitWithFP(downsampled_dense, initial_pose_))
                {
                    LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "init success with external pose: ";
                    // initial_pose_set_ = false;
                    init_success = true;
                    init_guess_pose = initial_pose_;
                    init_guess_pose_set = true;
                }
                else
                    LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] " << "initial pose init failed";
            }

            if (options_.init_with_fp_ && init_success == false)
            {
                /// 从功能点初始化
                /// 如果之前尝试过，那么需要间隔一段时间再进行搜索
                LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "test inited with features pose: ";
                if (!fp_init_fail_pose_vec_.empty() && current_dr_pose_set_)
                {
                    SE3 last_tried_pose = fp_init_fail_pose_vec_.back();
                    bool should_try =
                        (current_time - fp_last_tried_time_) > 2.0 ||
                        (current_dr_pose_.translation() - last_tried_pose.translation()).norm() > 0.3 ||
                        (current_dr_pose_.so3().inverse() * last_tried_pose.so3()).log().norm() > 10 * M_PI / 180.0;
                    if (!should_try)
                    {
                        LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] " << "skip trying init, please move to another place.";
                        init_success = false;
                    }
                }
                else
                {
                    LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "fp tried pose size: " << fp_init_fail_pose_vec_.size()
                              << ", dr pose is set: " << current_dr_pose_set_;
                }

                auto all_fps = map_->GetAllFP();
                bool fp_init_success = false;
                if (init_success == false)
                {
                    for (const auto &fp : all_fps)
                    {
                        if (InitWithFP(downsampled_dense, fp.pose_))
                        {
                            LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "init success with fp: " << fp.name_;
                            fp_init_success = true;
                            init_guess_pose = fp.pose_;
                            init_guess_pose_set = true;
                            break;
                        }
                    }
                }

                if (!fp_init_success)
                {
                    LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] " << "FP init failed.";
                    if (current_dr_pose_set_)
                    {
                        LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] "
                                  << "record fp failed time: " << std::setprecision(12) << current_time
                                  << ", pose: " << current_dr_pose_.translation().transpose();
                        fp_last_tried_time_ = current_time;
                        fp_init_fail_pose_vec_.emplace_back(current_dr_pose_);
                    }
                    init_success = false;
                }
                else
                {
                    fp_last_tried_time_ = 0;
                    fp_init_fail_pose_vec_.clear();
                    init_success = true;
                }
            }

            static bool need_down_ndt_resolution = false;
            static double min_init_confidence = options_.min_init_confidence_;
            if (init_success == false)
            {
                // 防止在狭窄走廊丢失定位，5 的网格分辨率过大导致得分过低无法初始化成功
                need_down_ndt_resolution = true;
                UL lock(match_mutex_);

                pcl_ndt_rough_.reset(new NDTType());
                pcl_ndt_rough_->setResolution(2.0); // 减小分辨率
                pcl_ndt_rough_->setStepSize(0.2);
                pcl_ndt_rough_->setMaximumIterations(12);
                pcl_ndt_rough_->setNeighborhoodSearchMethod(pclomp::DIRECT7);
                pcl_ndt_rough_->setNumThreads(options_.ndt_omp_th_);

                map_->SetNewTargetForNDT(pcl_ndt_rough_);

                // options_.min_init_confidence_ = min_init_confidence / 2;
                options_.min_init_confidence_ = min_init_confidence;

                LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "NDT Rough Resolution has been reset to 2.0";
            }
            else if (need_down_ndt_resolution == true)
            {
                need_down_ndt_resolution = false;
                UL lock(match_mutex_);

                pcl_ndt_rough_.reset(new NDTType());
                pcl_ndt_rough_->setResolution(5.0); // 减小分辨率
                pcl_ndt_rough_->setStepSize(0.5);
                pcl_ndt_rough_->setMaximumIterations(12);
                pcl_ndt_rough_->setNeighborhoodSearchMethod(pclomp::DIRECT7);
                pcl_ndt_rough_->setNumThreads(options_.ndt_omp_th_);

                map_->SetNewTargetForNDT(pcl_ndt_rough_);

                options_.min_init_confidence_ = min_init_confidence;
            }
            auto t2 = std::chrono::steady_clock::now();
            auto all_time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            LOG(INFO) << get_time_us() << "[INFO][Localization::" << __FUNCTION__ << "] " << "time cost " << all_time_used.count();
            if (init_success && init_guess_pose_set)
            {
                const SE3 delta_guess_to_init = init_guess_pose.inverse() * current_abs_pose_;
                const auto delta_rpy = math::SE3ToRollPitchYaw(delta_guess_to_init);
                constexpr double kRad2Deg = 57.29577951308232;
                LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] "
                          << "init guess->inited delta, xyz: " << delta_guess_to_init.translation().transpose()
                          << ", rpy(deg): " << delta_rpy.roll * kRad2Deg << ", "
                          << delta_rpy.pitch * kRad2Deg << ", " << delta_rpy.yaw * kRad2Deg;
                is_reset_ = false;
            }
            else if (is_reset_)
            {
                // 仅重定位失败才再 Reset LIO；冷启动 is_reset_==false，不影响第一次 init。
                need_reset_ = true;
            }
            return;
        }

        /// 4. 设置当前帧对应的 pose guess
        // 获取 lio 插值结果
        SE3 guess_from_lo = last_abs_pose_;
        if (last_lo_pose_set_ && current_lo_pose_set_)
        {
            // 如果有里程计，则用两个时刻的相对定位来递推，估计一个当前pose的初值
            const SE3 delta = last_lo_pose_.inverse() * current_lo_pose_;
            guess_from_lo = last_abs_pose_ * delta;
            // guess_from_lo.translation()[2] = 0;
            // LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] "
            //             << "loc using lo guess: " << guess_from_lo.translation().transpose();
        }

        // 基于自身插值并获取插值结果
        SE3 guess_from_self = guess_from_lo;
        if (lidar_loc_pose_queue_.size() >= 2)
        {
            SE3 pred;
            TimedPose match;
            if (math::PoseInterp<TimedPose>(
                    current_time, lidar_loc_pose_queue_, [](const TimedPose &p)
                    { return p.timestamp_; },
                    [](const TimedPose &p)
                    { return p.pose_; }, pred, match, 2.0))
            {
                guess_from_self = pred;
            }
        }

        // 获取 dr 插值结果
        SE3 guess_from_dr = guess_from_lo;
        if (last_dr_pose_set_ && current_dr_pose_set_)
        {
            const SE3 delta = last_dr_pose_.inverse() * current_dr_pose_;
            guess_from_dr = last_abs_pose_ * delta;
            // guess_from_dr.translation()[2] = 0;
        }

        // 基于 dr 与 lio 插值位姿的差异判定是否使用 dr 为初值尝试配准
        bool try_dr = false;
        if (((guess_from_dr.translation() - guess_from_lo.translation()).norm() >= try_other_guess_trans_th_ ||
             (guess_from_dr.so3().inverse() * guess_from_lo.so3()).log().norm() >= try_other_guess_rot_th_))
        {
            if (options_.enable_debug_log_)
                LOG(INFO) << get_time_us() << "[DEBUG][LidarLoc::" << __FUNCTION__ << "] "
                          << "trying dr pose: " << guess_from_dr.translation().transpose() << ", delta lo: "
                          << (guess_from_dr.so3().inverse() * guess_from_lo.so3()).log().norm()
                          << ", vel_norm: " << current_vel_b_.norm();
            try_dr = false;
        }

        // 基于 self 与 lio、dr 插值位姿的差异判定是否使用 self 为初值尝试配准
        bool try_self = false;
        if (options_.try_self_extrap_)
        {
            if (((guess_from_self.translation() - guess_from_lo.translation()).norm() >= try_other_guess_trans_th_ ||
                 (guess_from_self.so3().inverse() * guess_from_lo.so3()).log().norm() >= try_other_guess_rot_th_) &&
                ((guess_from_dr.translation() - guess_from_self.translation()).norm() >= try_other_guess_trans_th_ ||
                 (guess_from_dr.so3().inverse() * guess_from_self.so3()).log().norm() >= try_other_guess_rot_th_))
            {
                if (options_.enable_debug_log_)
                    LOG(INFO) << get_time_us() << "[DEBUG][LidarLoc::" << __FUNCTION__ << "] "
                              << "trying self extrap pose: " << guess_from_self.translation().transpose() << ", delta lo: "
                              << (guess_from_self.so3().inverse() * guess_from_lo.so3()).log().norm();
                try_self = true;
            }
        }

        /// 5. 载入地图, 与地图匹配定位
        /// 尝试各种初始估计
        CloudPtr output_cloud(new PointCloudType);
        double fitness_score = 0;
        SE3 current_pose_esti = guess_from_lo;
        bool loc_success_lo, loc_success_self, loc_success_dr;
        loc_success_lo = loc_success_self = loc_success_dr = false;
        bool loc_success = false;

        // map_->LoadOnPose(guess_from_dr);

        loc_success_lo = Localize(current_pose_esti, fitness_score, input, output_cloud); // LO 那个肯定会算
        const char *selected_guess_source = "LO";
        if (options_.enable_debug_log_)
            LOG(INFO) << get_time_us() << "[DEBUG][LidarLoc::" << __FUNCTION__ << "] " << "take lo guess: " << current_pose_esti.translation().transpose()
                      << " , confidence: " << fitness_score << ", v_norm: " << current_vel_b_.norm();
        double score_lo = fitness_score;

        SE3 res_of_lo = current_pose_esti;
        SE3 res_of_dr = current_pose_esti;
        SE3 res_of_self = current_pose_esti;
        double score_dr = 0;

        // 先尝试外部预测，最后用自身
        if (try_dr)
        {
            /// 尝试DR外推的pose
            res_of_dr = guess_from_dr;
            loc_success_dr = Localize(res_of_dr, score_dr, input, output_cloud);
            if (score_dr > (fitness_score - 0.1))
            {
                current_pose_esti = res_of_dr;
                fitness_score = score_dr;
                selected_guess_source = "DR";
                if (options_.enable_debug_log_)
                    LOG(INFO) << get_time_us() << "[DEBUG][LidarLoc::" << __FUNCTION__ << "] " << "take dr guess: " << current_pose_esti.translation().transpose()
                              << " , confidence: " << score_dr << ", v_norm: " << current_vel_b_.norm();
            }
        }

        /// NOTE 如果LO, DR出发点和收敛点不同，但分值相近，说明场景可能处在退化状态，此时使用DR预测的Pose
        if (try_dr && (res_of_lo.translation() - res_of_dr.translation()).head<2>().norm() > 0.2 &&
            fabs(score_lo - score_dr) < 0.2 && score_lo < 1.2)
        {
            LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] " 
                            << "判定激光定位进入退化状态，现在会使用DR递推pose而不是激光定位位置";
            current_pose_esti = res_of_dr;
            selected_guess_source = "DR(degenerate)";
        }

        double score_self = 0;
        if (try_self)
        {
            /// 尝试自身外推的pose
            if (options_.enable_debug_log_)
                LOG(INFO) << get_time_us() << "[DEBUG][LidarLoc::" << __FUNCTION__ << "] " << "localize with extrap";

            res_of_self = guess_from_self;
            loc_success_self = Localize(res_of_self, score_self, input, output_cloud);

            // 避免分值接近但长时间采信自身预测，此处更相信外部预测源
            if (score_self > (fitness_score + 0.1) && score_self > score_dr)
            {
                current_pose_esti = res_of_self;
                fitness_score = score_self;
                selected_guess_source = "SELF";
                if (options_.enable_debug_log_)
                    LOG(INFO) << get_time_us() << "[DEBUG][LidarLoc::" << __FUNCTION__ << "] " << "take self guess: " << current_pose_esti.translation().transpose()
                              << " , confidence: " << score_self;
            }
        }

        // TODO: scale 参数化出来，用纯激光定位有点太抖了，加一些权重
        // Vec6d delta = (guess_from_lo.inverse() * current_pose_esti).log();
        // SE3 esti_balanced = guess_from_lo * SE3::exp(delta * 0.1);
        // current_pose_esti = esti_balanced;

        if (options_.force_2d_)
        {
            PoseRPYD RPYXYZ = math::SE3ToRollPitchYaw(current_pose_esti);
            RPYXYZ.roll = 0;
            RPYXYZ.pitch = 0;
            RPYXYZ.z = 0;
            current_pose_esti = math::XYZRPYToSE3(RPYXYZ);
        }

        // if (options_.with_height_) {
        //     current_pose_esti.translation()[2] = map_height_;
        //     LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "adjust current pose to : " << current_pose_esti.translation().transpose();
        // }

        current_abs_pose_ = current_pose_esti;
        current_score_ = fitness_score;
        double delta_rel_abs_pose = 0;
        bool lidar_loc_odom_valid = true;

        // loc_success_lo 和 loc_success_dr 失败一段时间的时候，lio 系统已经崩溃了，需要 reset
        if (loc_success_lo || loc_success_self || loc_success_dr)
            loc_success = true;
        else
            LOG(ERROR) << get_time_us() << "[ERROR][LidarLoc::" << __FUNCTION__ << "] " << "loc success is false.";

        // TODO: 添加 LIO 的 RESET，与 loc 的 reset 区分开，lio 飘了只 reset lio
        if (loc_success)
        {
            lidar_loc_odom_valid = CheckLidarOdomValid(current_pose_esti, delta_rel_abs_pose);
            last_timestamp_ = current_timestamp_; // 成功时，更新上一时刻激光定位时间

            if (loc_success_lo || loc_success_dr)
                match_fail_count_ = 0;
            else
                match_fail_count_++;
        }
        else
        {
            current_score_ = fitness_score;
            // LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] " << "localization failed! score: " << current_score_;

            /// TODO: 若连续3帧匹配失败就设一个大分值
            ++match_fail_count_;
        }

        /// 确定激光定位是否满足平滑性要求
        Vec3d dpred = current_abs_pose_.translation() - guess_from_self.translation();
        if (fabs(dpred[0]) < 0.5 && fabs(dpred[1]) < 0.5 &&
            (current_abs_pose_.so3().inverse() * guess_from_self.so3()).log().norm() < 2.0 * M_PI / 180.0)
        {
            localization_result_.lidar_loc_smooth_flag_ = true;
        }
        else
            localization_result_.lidar_loc_smooth_flag_ = false;

        localization_result_.lidar_loc_odom_reliable_ = lo_reliable_;
        /// 7. 输出结果
        {
            UL lock(result_mutex_);
            localization_result_.timestamp_ = current_timestamp_;
            localization_result_.confidence_ = fitness_score;
            if (match_fail_count_ < 3)
            {
                recover_pose_out_.open(options_.recover_pose_path_);
                Vec3d t = current_pose_esti.translation();
                if (recover_pose_out_)
                {
                    Vec3d t = current_pose_esti.translation();
                    Quatd q = current_pose_esti.unit_quaternion();
                    recover_pose_out_ << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
                    // recover_pose_out_ << t[0] << " " << t[1] << " " << t[2] << " " << 0 << " " << 0 << " " << 0 << " " << 1;
                    recover_pose_out_.close();
                }
                reset_pose_ = first_abs_pose_.inverse() * current_pose_esti;
                LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << std::fixed << std::setprecision(4)
                          << "Save recover pose: " << t[0] << " " << t[1] << " " << t[2]
                          << ", Bad frame num: " << match_fail_count_ << ", NDT score: " << fitness_score
                          << ", source: " << selected_guess_source;
                localization_result_.lidar_loc_valid_ = true;
                localization_result_.status_ = LocalizationStatus::GOOD;
            }
            else if (match_fail_count_ < 10)
            {
                if (current_score_ > 1.3)
                {
                    recover_pose_out_.open(options_.recover_pose_path_);
                    if (recover_pose_out_)
                    {
                        Vec3d t = current_pose_esti.translation();
                        Quatd q = current_pose_esti.unit_quaternion();
                        recover_pose_out_ << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
                        // recover_pose_out_ << t[0] << " " << t[1] << " " << t[2] << " " << 0 << " " << 0 << " " << 0 << " " << 1;
                        recover_pose_out_.close();
                    }
                    reset_pose_ = first_abs_pose_.inverse() * current_pose_esti;
                    LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << std::fixed << std::setprecision(4)
                              << "Save recover pose, Bad frame num: " << match_fail_count_
                              << ", NDT score: " << fitness_score
                              << ", source: " << selected_guess_source;
                    localization_result_.lidar_loc_valid_ = true;
                    localization_result_.status_ = LocalizationStatus::GOOD;
                }
                else
                {
                    LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] "
                                 << "Not save recover pose, Bad frame num: " << match_fail_count_ << " NDT score: " << fitness_score;
                    localization_result_.lidar_loc_valid_ = false;
                    localization_result_.status_ = LocalizationStatus::FOLLOWING_DR;
                }
            }
            else if (match_fail_count_ >= 10 && match_fail_count_ < 100)
            {
                need_reset_ = true;
                localization_result_.lidar_loc_valid_ = false;
                localization_result_.status_ = LocalizationStatus::FAIL;
            }
            else
            {
                match_fail_count_ = 100;
                localization_result_.lidar_loc_valid_ = false;
                localization_result_.status_ = LocalizationStatus::FAIL;
            }

            localization_result_.lidar_loc_odom_delta_ = delta_rel_abs_pose;
            localization_result_.lidar_loc_odom_error_normal_ = lidar_loc_odom_valid;
            localization_result_.pose_ = current_pose_esti;
        }

        UpdateState();

        /// 8. 更新动态图层
        /// 条件：定位成功，并且与上次更新间隔一定距离，匹配分值超过阈值
        bool score_cond = current_score_ > options_.update_lidar_loc_score_;

        if (options_.update_dynamic_cloud_ && loc_success &&
            (((current_pose_esti.translation() - last_dyn_upd_pose_.pose_.translation()).norm() >
              options_.update_kf_dis_) ||
             fabs(current_time - last_dyn_upd_pose_.timestamp_) > options_.update_kf_time_))
        {
            if (score_cond /*  || update_cache_dis_ < options_.max_update_cache_dis_ */)
            {
                // LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "passing through z filter, input:" << input->size();
                pcl::PassThrough<PointType> pass;
                pass.setInputCloud(input);
                pass.setFilterFieldName("z");
                // TODO: 参数化这里的 z 轴下限
                pass.setFilterLimits(0.5, options_.filter_z_max_);

                CloudPtr input_z_filter(new PointCloudType());
                pass.filter(*input_z_filter);

                if (!input_z_filter->empty())
                {
                    CloudPtr cloud_t(new PointCloudType());
                    pcl::transformPointCloud(*input_z_filter, *cloud_t, current_pose_esti.matrix());

                    // 以现在的scan来更新地图
                    map_->UpdateDynamicCloud(cloud_t, true);

                    last_dyn_upd_pose_.timestamp_ = current_time;
                    last_dyn_upd_pose_.pose_ = current_pose_esti;
                }
            }
        }

        if (lidar_loc_pose_queue_.empty())
            lidar_loc_pose_queue_.emplace_back(current_time, current_abs_pose_);
        else if (current_time > lidar_loc_pose_queue_.back().timestamp_)
            lidar_loc_pose_queue_.emplace_back(current_time, current_abs_pose_);

        while (lidar_loc_pose_queue_.size() > 200)
            lidar_loc_pose_queue_.pop_front();

        ave_scores_.emplace_back(current_score_);

        while (ave_scores_.size() > 20)
            ave_scores_.pop_front();
    }

    bool LidarLoc::CheckLidarOdomValid(const SE3 &current_pose_esti, double &delta_posi)
    {
        delta_posi = ((last_lo_pose_.inverse() * current_lo_pose_).translation() -
                      (last_abs_pose_.inverse() * current_pose_esti).translation())
                         .head(2)
                         .norm();

        bool valid = true;

        if (delta_posi > options_.lidar_loc_odom_th_ && last_lo_pose_set_)
        {
            LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] " << "delta_rel_abs_pose is: " << delta_posi;
            LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] " << "LO相对pose: "
                         //  << last_lo_pose_.translation().transpose() << " "
                         //  << current_lo_pose_.translation().transpose() << " "
                         << (last_lo_pose_.inverse() * current_lo_pose_).translation().transpose();
            LOG(WARNING) << get_time_us() << "[WARN][LidarLoc::" << __FUNCTION__ << "] " << "Lidar Loc 计算相对 pose: "
                         //  << last_abs_pose_.translation().transpose() << " "
                         //  << current_pose_esti.translation().transpose() << " "
                         << (last_abs_pose_.inverse() * current_pose_esti).translation().transpose();
            lo_reliable_ = false;
            lo_reliable_cnt_ = 10;
            valid = false;
        }

        last_abs_pose_ = current_pose_esti;
        last_lo_pose_ = current_lo_pose_;
        last_lo_pose_set_ = true;
        last_dr_pose_ = current_dr_pose_;
        last_dr_pose_set_ = true;

        return valid;
    }

    bool LidarLoc::Localize(SE3 &pose, double &confidence, CloudPtr input, CloudPtr output, bool use_rough_res)
    {
        Eigen::Matrix4f trans;
        bool loc_success = false;
        Eigen::Matrix4f guess_pose = pose.matrix().cast<float>();

        // LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "loc from: " << pose.translation().transpose();

        if (pcl_ndt_->getInputTarget() == nullptr)
        {
            LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "lidar loc target is null, skip";
            return false;
        }

        UL lock(match_mutex_);
        NDTType::Ptr ndt = nullptr;

        if (use_rough_res)
            ndt = pcl_ndt_rough_;
        else
            ndt = pcl_ndt_;

        ndt->setInputSource(input);
        ndt->align(*output, guess_pose);
        trans = ndt->getFinalTransformation();
        confidence = ndt->getTransformationProbability();
        // TODO: 这里可能写错了，要做匹配失败的处理
        // if (loc_inited_ == false && confidence > options_.min_init_confidence_)
        //     loc_success = true;
        // else
        //     loc_success = true;

        // auto tgt = ndt->getInputTarget();
        // if (!tgt->empty())
        // {
        //     pcl::io::savePCDFile("/home/abc/slam_dc/localization_dc/tgt.pcd", *tgt);
        // }

        if (loc_inited_ == false)
            loc_success = true;
        else if (confidence > options_.min_init_confidence_)
            loc_success = true;

        if (options_.enable_icp_adjust_ && loc_inited_)
        {
            Eigen::Matrix4f adjust_trans;
            CloudPtr input_voxel(new PointCloudType);
            pcl::VoxelGrid<PointType> voxel_icp;

            double ls = 0.2;
            voxel_icp.setLeafSize(ls, ls, ls);
            voxel_icp.setInputCloud(input);
            voxel_icp.filter(*input_voxel);
            pcl_icp_->setInputSource(input_voxel);
            pcl_icp_->align(*output, trans);
            adjust_trans = pcl_icp_->getFinalTransformation();

            Eigen::Matrix3f rotation_diff = trans.block<3, 3>(0, 0).transpose() * adjust_trans.block<3, 3>(0, 0);
            Eigen::AngleAxisf angle_axis(rotation_diff);
            float a = angle_axis.angle();
            float d = (trans.block<3, 1>(0, 3) - adjust_trans.block<3, 1>(0, 3)).norm();
            if (options_.enable_debug_log_)
                LOG(INFO) << get_time_us() << "[DEBUG][LidarLoc::" << __FUNCTION__ << "] " << "icp adjust d: " << d << ", a: " << a;

            // TODO: 阈值从 .05 改成了现在的 .1
            if (pcl_icp_->hasConverged() && std::fabs(d) <= 0.1 && std::fabs(a) <= 0.1)
            {
                LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "icp ajust trans set success";
                trans = adjust_trans;
            }
        }

        Eigen::Matrix3d rot = trans.block<3, 3>(0, 0).cast<double>();
        Quatd q_3d = Quatd(rot);
        Vec3d t_3d = trans.block<3, 1>(0, 3).cast<double>();
        q_3d.normalize();
        pose = SE3(q_3d, t_3d);

        return loc_success;
    }

    // void LidarLoc::UpdateState(const CloudPtr &input) { last_timestamp_ = current_timestamp_; }
    void LidarLoc::UpdateState() { last_timestamp_ = current_timestamp_; }

    void LidarLoc::SetInitRltState()
    {
        UL lock(result_mutex_);
        localization_result_.confidence_ = 0.0;
        localization_result_.timestamp_ = current_timestamp_;
        localization_result_.lidar_loc_valid_ = false;
        localization_result_.status_ = LocalizationStatus::INITIALIZING;
    }

    bool LidarLoc::LocInited()
    {
        // UL lock( data_mutex_);
        return loc_inited_;
    }

    bool LidarLoc::AssignLOPose(double timestamp)
    {
        UL lock(lo_pose_mutex_);
        SE3 interp_pose;
        NavState best_match;
        // 无法拿到最新的，插值结果都是外推出来的，和真实有时偏差会很大（╯﹏╰）
        // if (!lo_pose_queue_.empty()) {
        //     LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] "
        //               << "lo interp: " << timestamp << " " << lo_pose_queue_.back().timestamp_ << " "
        //               << lo_pose_queue_.back().pose_.translation().transpose();
        // }

        bool pose_interp_success = math::PoseInterp<NavState>(
            timestamp, lo_pose_queue_, [](const NavState &dr)
            { return dr.timestamp_; },
            [](const NavState &dr)
            { return dr.GetPose(); }, interp_pose, best_match, 5.0);

        if (pose_interp_success)
        {
            current_lo_pose_ = interp_pose;
            current_lo_pose_set_ = true;

            current_vel_b_ = best_match.GetRot().inverse() * best_match.GetVel();
            current_vel_ = best_match.GetVel();

            // if (options_.with_height_) {
            //     current_lo_pose_.translation()[2] = map_height_;
            // }

            return true;
        }
        else
        {
            current_lo_pose_set_ = false;
            return false;
        }
    }

    bool LidarLoc::AssignDRPose(double timestamp)
    {
        UL lock(dr_pose_mutex_);
        SE3 interp_pose;
        NavState best_match;
        bool pose_interp_success = math::PoseInterp<NavState>(
            timestamp, dr_pose_queue_, [](const NavState &dr)
            { return dr.timestamp_; },
            [](const NavState &dr)
            { return dr.GetPose(); }, interp_pose, best_match, 5.0);

        if (pose_interp_success)
        {
            current_dr_pose_ = interp_pose;
            current_dr_pose_set_ = true;

            // if (options_.with_height_) {
            //     current_dr_pose_.translation()[2] = map_height_;
            // }

            return true;
        }
        else
        {
            current_dr_pose_set_ = false;
            return false;
        }
    }

    void LidarLoc::Finish()
    {
        if (map_)
        {
            LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "saving maps";
            update_map_quit_ = true;
            if (update_map_thread_.joinable())
                update_map_thread_.join();

            if (!has_set_pose_)
            {
                map_->SaveToBin(false);
                LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "function points saved";
            }
        }
    }

    void LidarLoc::Reset()
    {
        {
            UL lock(dr_pose_mutex_);
            dr_pose_queue_.clear();
        }
        {
            UL lock(lo_pose_mutex_);
            lo_pose_queue_.clear();
        }
        lidar_loc_pose_queue_.clear();
        current_dr_pose_set_ = false;
        current_lo_pose_set_ = false;
        last_lo_pose_set_ = false;
        last_dr_pose_set_ = false;

        match_fail_count_ = 0;
        current_score_ = 1e5;
        init_cloud_buffer_.clear();
        fp_init_fail_pose_vec_.clear();

        ResetLastPose(reset_pose_);
        SetInitialPose(SE3());

        loc_inited_ = false;
        is_reset_ = true;
        need_reset_ = false;
    }
} // namespace lightning::loc
