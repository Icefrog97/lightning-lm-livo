#include <pcl/common/transforms.h>
#include <iomanip>

#include "localization/lidar_loc.h"
#include "localization/localization.h"
#include "localization/pose_graph/pgo.h"
#include "pub/pub_ROS1.h"
#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>

namespace lightning::loc
{
    Localization::Localization(Options options) { options_ = options; }

    bool Localization::Init(const rclcpp::Node::SharedPtr &nh, const lightning::LaserMapping::Config &lio_cfg,
                            const LidarLoc::Options &lidar_loc_options, const lightning::PubROSTopic::Options &pub_opt)
    {
        UL lock(global_mutex_);
        if (lidar_loc_ != nullptr)
            Finish();

        LaserMapping::Options opt_lio;
        opt_lio.is_in_slam_mode_ = false;
        opt_lio.T_body_wrt_lio_ = lio_cfg.T_body_lio.translation();
        opt_lio.Q_body_wrt_lio_ = Eigen::Quaterniond(lio_cfg.T_body_lio.linear());

        lio_ = std::make_shared<LaserMapping>(opt_lio);
        if (!lio_->Init(lio_cfg, nh))
        {
            LOG(ERROR) << get_time_us() << "[ERROR][" << __FUNCTION__ << "] " << "failed to init lio";
            return false;
        }

        enable_debug_log_ = lidar_loc_options.enable_debug_log_;
        lidar_loc_ = std::make_shared<LidarLoc>(lidar_loc_options);

        pub_ = std::make_shared<lightning::PubROSTopic>();
        pub_->Init(nh, opt_lio.is_in_slam_mode_, pub_opt);

        lidar_loc_->SetRviz(pub_);
        lidar_loc_->Init();

        pgo_ = std::make_shared<PGO>();
        pgo_->SetDebug(false);

        // 纯 LIO 仍由一帧雷达触发一次 Run；LIVO 改由独立前端线程推进两步状态机。
        lidar_odom_proc_cloud_.SetMaxSize(1);
        lidar_loc_proc_cloud_.SetMaxSize(1);

        lidar_odom_proc_cloud_.SetName("激光里程计");
        lidar_loc_proc_cloud_.SetName("激光定位");

        // 允许跳帧
        lidar_loc_proc_cloud_.SetSkipParam(options_.enable_lidar_loc_skip_, options_.lidar_loc_skip_num_);
        lidar_odom_proc_cloud_.SetSkipParam(options_.enable_lidar_odom_skip_, options_.lidar_odom_skip_num_);

        lidar_odom_proc_cloud_.SetProcFunc([this](CloudPtr cloud)
                                           { LidarOdomProcCloud(cloud, false); });
        lidar_loc_proc_cloud_.SetProcFunc([this](const LidarLocInput &in)
                                          { LidarLocProcCloud(in.cloud, in.timestamp); });

        if (options_.online_mode_)
        {
            if (!lio_->VisualEnabled())
                lidar_odom_proc_cloud_.Start();
            lidar_loc_proc_cloud_.Start();
        }

        pgo_->SetHighFrequencyGlobalOutputHandleFunction([this](const LocalizationResult &res, const bool &is_dr_frame)
                                                         {
            loc_result_ = res;

            pub_->SetPosestamp(loc_result_.ToNavState());
            pub_->PublishOdom(loc_result_.timestamp_);
            (void)is_dr_frame;
            pub_->PublishPath(loc_result_.timestamp_); });

        p_pre_.reset(new PointCloudPreprocess());
        p_pre_->blind_ = options_.preprocess_blind_;
        lidar_type_ = options_.preprocess_lidar_type_;
        p_pre_->point_filter_num_ = options_.preprocess_point_filter_num_;
        p_pre_->N_SCANS_ = options_.preprocess_scan_line_;
        p_pre_->SetHeightROI(options_.preprocess_height_max_, options_.preprocess_height_min_);

        LOG(INFO) << get_time_us() << "[INFO][Localization::" << __FUNCTION__ << "] " << "lidar_type " << lidar_type_;
        if (lidar_type_ == 1)
            p_pre_->SetLidarType(LidarType::ROBOSENSE);
        else if (lidar_type_ == 2)
            p_pre_->SetLidarType(LidarType::VELO16);
        else if (lidar_type_ == 3)
            p_pre_->SetLidarType(LidarType::OUST64);
        else if (lidar_type_ == 4)
            p_pre_->SetLidarType(LidarType::mid360);
        else if (lidar_type_ == 5)
            p_pre_->SetLidarType(LidarType::LIVOX_CUSTOM);
        else
            LOG(WARNING) << get_time_us() << "[WARN][Localization::" << __FUNCTION__ << "] " << "unknown lidar_type";

        // 所有模块和输出回调准备好以后再启动 LIVO 前端，避免初始化过程中访问半成品状态。
        if (options_.online_mode_ && lio_->VisualEnabled())
        {
            stop_livo_frontend_.store(false);
            livo_frontend_thread_ = std::thread(&Localization::LivoFrontendLoop, this);
        }

        return true;
    }

    void Localization::ProcessLidarMsg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud)
    {
        // LOG(INFO) << get_time_us() << "[INFO][Localization::" << __FUNCTION__ << "] " << "start ";
        UL lock(global_mutex_);
        HandlePendingReset();

        if (lidar_loc_ == nullptr || lio_ == nullptr || pgo_ == nullptr)
            return;

        CloudPtr laser_cloud(new PointCloudType);
        p_pre_->Process(cloud, laser_cloud);

        laser_cloud->header.stamp = cloud->header.stamp.sec * 1e9 + cloud->header.stamp.nanosec;

        if (!lio_->VisualEnabled() && lio_->need_trigger_imu)
        {
            lidar_odom_proc_cloud_.AddMessage(nullptr);
            // lio_->need_trigger_imu--;
            lio_->need_trigger_imu = false;
        }

        if (options_.online_mode_)
        {
            if (lio_->VisualEnabled())
                lio_->ProcessPointCloud2(laser_cloud);
            else
                lidar_odom_proc_cloud_.AddMessage(laser_cloud);
        }
        else
            LidarOdomProcCloud(laser_cloud, false);
    }

    void Localization::ProcessLivoxLidarMsg(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr cloud)
    {
        UL lock(global_mutex_);
        HandlePendingReset();

        if (lidar_loc_ == nullptr || lio_ == nullptr || pgo_ == nullptr)
            return;

        CloudPtr laser_cloud(new PointCloudType);
        p_pre_->Process(cloud, laser_cloud);

        laser_cloud->header.stamp = cloud->header.stamp.sec * 1e9 + cloud->header.stamp.nanosec;

        if (!lio_->VisualEnabled() && lio_->need_trigger_imu)
        {
            lidar_odom_proc_cloud_.AddMessage(nullptr);
            lio_->need_trigger_imu = false;
        }

        if (options_.online_mode_)
        {
            if (lio_->VisualEnabled())
                lio_->ProcessPointCloud2(laser_cloud);
            else
                lidar_odom_proc_cloud_.AddMessage(laser_cloud);
        }
        else
            LidarOdomProcCloud(laser_cloud, false);
    }

    void Localization::ProcessImageMsg(const sensor_msgs::msg::Image::ConstSharedPtr image_msg, int cam_id)
    {
        UL lock(global_mutex_);
        if (lio_ == nullptr || !lio_->NeedImageInput())
            return;

        try
        {
            // Always convert to OpenCV BGR so debug publish (BGR8) and point coloring stay consistent.
            cv_bridge::CvImageConstPtr image =
                cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
            lio_->ProcessImage(cam_id, image->image, SecFromStamp(image_msg->header.stamp));
            if (options_.online_mode_ && !lio_->VisualEnabled())
                lidar_odom_proc_cloud_.AddMessage(nullptr);
            else if (!options_.online_mode_)
                LidarOdomProcCloud(nullptr, false);
        }
        catch (const cv_bridge::Exception &e)
        {
            LOG(ERROR) << get_time_us() << "[ERROR][Localization::" << __FUNCTION__ << "] " << e.what();
        }
    }

    bool Localization::VisualEnabled() const
    {
        return lio_ != nullptr && lio_->VisualEnabled();
    }

    bool Localization::NeedImageInput() const
    {
        return lio_ != nullptr && lio_->NeedImageInput();
    }

    void Localization::LivoFrontendLoop()
    {
        while (!stop_livo_frontend_.load())
        {
            // Run() 的 false 既可能表示数据不足，也可能表示已经消费了空 LIO/VIO。
            // 固定周期重试不依赖该返回值，因此状态机不会因 false 永久休眠。

            LidarOdomProcCloud(nullptr, false);

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void Localization::LidarOdomProcCloud(CloudPtr cloud, const bool &trigger_imu)
    {
        // LOG(INFO) << get_time_us() << "[INFO][Localization::" << __FUNCTION__ << "] " << "start ";
        if (lio_ == nullptr)
            return;

        if (cloud != nullptr)
            lio_->ProcessPointCloud2(cloud);

        if (!lio_->Run())
            return;

        const bool vio_step = lio_->LastRunWasVio();
        if (!vio_step)
            lio_->PublishFrameBody(pub_);

        auto lo_state = lio_->GetExtrinsicRemappedState(lio_->GetState());

        if (!vio_step)
        {
            lidar_loc_->ProcessLO(lo_state);
            pgo_->ProcessLidarOdom(lo_state);
        }

        const double lio_end_time = lo_state.timestamp_;

        // Publish LIO/LIVO path in map frame, locked to Loc pose at Loc-init time.
        // Successful VIO updates this path from kf_/state_point_; empty VIO never gets here.
        if (lidar_loc_->LocInited() && pub_)
        {
            const SE3 T_lio = lo_state.GetPose();
            if (!lio_path_align_set_)
            {
                const SE3 T_loc = lidar_loc_->GetLocalizationResult().pose_;
                T_map_from_lio_ = T_loc * T_lio.inverse();
                lio_path_align_set_ = true;
                LOG(INFO) << get_time_us() << "[INFO][Localization::" << __FUNCTION__ << "] "
                          << "aligned LIO path to Loc init, loc_t=" << T_loc.translation().transpose()
                          << ", lio_t=" << T_lio.translation().transpose();
            }
            pub_->PublishLioPath(T_map_from_lio_ * T_lio, lio_end_time);
        }
        else if (!lidar_loc_->LocInited() && lio_path_align_set_)
        {
            lio_path_align_set_ = false;
        }

        if (vio_step)
            return;

        auto scan = lio_->GetScanUndist();

        /// 获得lio的关键帧。未初始化时仍每帧送给定位（和冷启动一样攒 3 帧），不要 loc_on_kf 把门。
        if (lidar_loc_->LocInited() && options_.loc_on_kf_)
        {
            auto kf = lio_->GetKeyframe();
            if (kf == lio_kf_)
                return;
            lio_kf_ = kf;
        }

        if (!lidar_loc_->LocInited())
            scan.reset(new PointCloudType(*scan));

        if (options_.online_mode_)
            lidar_loc_proc_cloud_.AddMessage(LidarLocInput{scan, lio_end_time});
        else
            LidarLocProcCloud(scan, lio_end_time);
    }

    void Localization::LidarLocProcCloud(CloudPtr scan_undist, double timestamp)
    {
        lidar_loc_->ProcessCloud(scan_undist, timestamp);

        auto res = lidar_loc_->GetLocalizationResult();

        global_status_ = res.status_;

        // const auto q = res.pose_.unit_quaternion();
        // const auto rpy = math::SE3ToRollPitchYaw(res.pose_);
        // constexpr double kRad2Deg = 57.29577951308232;
        // LOG(WARNING) << "[LidarLoc Result] "
        //           << "Time: " << std::fixed << std::setprecision(3) << res.timestamp_
        //           << " | Pos: " << res.pose_.translation().transpose()
        //           << " | Q(x y z w): " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
        //           << " | RPY(deg): " << rpy.roll * kRad2Deg << " " << rpy.pitch * kRad2Deg << " " << rpy.yaw * kRad2Deg;

        pgo_->ProcessLidarLoc(res);

        if (loc_state_callback_)
        {
            auto loc_state = std::make_shared<std_msgs::msg::Int32>();
            loc_state->data = static_cast<int>(res.status_);
            LOG(INFO) << get_time_us() << "[INFO][Localization::[" << __FUNCTION__ << "] " << "loc_state: " << loc_state->data;
            loc_state_callback_(*loc_state);
        }
    }

    void Localization::ProcessIMUMsg(IMUPtr imu)
    {
        UL lock(global_mutex_);

        if (lidar_loc_ == nullptr || lio_ == nullptr || pgo_ == nullptr)
            return;

        double this_imu_time = imu->timestamp;
        if (last_imu_time_ > 0 && this_imu_time < last_imu_time_)
            LOG(WARNING) << get_time_us() << "[WARN][Localization::" << __FUNCTION__ << "] "
                         << "IMU 时间异常：" << this_imu_time << ", last: " << last_imu_time_;

        last_imu_time_ = this_imu_time;

        // mid360 的重力加速度单位是 G，imu_predict 之后直接给到 Pgo 了，所以这里直接转换成 m/s^2
        if (lidar_type_ == 4 || lidar_type_ == 5 || lidar_type_ == 3)
            imu->linear_acceleration *= G_m_s2;

        /// 里程计处理IMU
        lio_->ProcessIMU(imu);

        auto dr_state = lio_->GetExtrinsicRemappedState(lio_->GetIMUState());
        if (!dr_state.pose_is_ok_)
            return;

        // 这里只能判定 Lio 未发散的时候的静止，该情况下不需要重新初始化
        // DR 增量诊断：PGO/外推器用 last_dr^{-1} * cur_dr 推红线。
        // 正常一步只有一个 IMU 周期的位移；回绕重放会让某一步的增量远超真实运动。
        {
            const SE3 cur_dr_pose = dr_state.GetPose();
            if (last_dr_dbg_set_)
            {
                const SE3 incre = last_dr_dbg_pose_.inverse() * cur_dr_pose;
                const double dt = dr_state.timestamp_ - last_dr_dbg_time_;
                const double incre_trans = incre.translation().norm();
                const double incre_rot_deg = incre.so3().log().norm() * 57.29577951308232;
                // 一步 DR 增量对应的等效速度；远大于 |v| 就是回绕/重放引入的假运动
                const double implied_speed = dt > 1e-6 ? incre_trans / dt : -1.0;
                if (incre_trans > 0.05 || incre_rot_deg > 1.0 || dt <= 0.0 || dt > 0.05)
                {
                    LOG(WARNING) << get_time_us() << "[WARN][Localization::" << __FUNCTION__ << "] "
                                 << "[DR_STEP] " << std::fixed << std::setprecision(4)
                                 << "dt=" << dt
                                 << " incre_trans=" << incre_trans
                                 << " incre_rot_deg=" << incre_rot_deg
                                 << " implied_speed=" << implied_speed
                                 << " |v|=" << dr_state.GetVel().norm()
                                 << " t=" << dr_state.timestamp_
                                 << " p=" << cur_dr_pose.translation().transpose()
                                 << " last_p=" << last_dr_dbg_pose_.translation().transpose();
                }
            }
            last_dr_dbg_pose_ = cur_dr_pose;
            last_dr_dbg_time_ = dr_state.timestamp_;
            last_dr_dbg_set_ = true;
        }

        lidar_loc_->ProcessDR(dr_state);
        pgo_->ProcessDR(dr_state);
    }

    void Localization::Finish()
    {
        stop_livo_frontend_.store(true);
        if (livo_frontend_thread_.joinable())
            livo_frontend_thread_.join();

        lidar_loc_->Finish();
        lidar_loc_proc_cloud_.Quit();
        if (!lio_ || !lio_->VisualEnabled())
            lidar_odom_proc_cloud_.Quit();
    }

    void Localization::SetExternalPose(const Eigen::Quaterniond &q, const Eigen::Vector3d &t)
    {
        UL lock(global_mutex_);
        /// 设置外部重定位的pose
        if (lidar_loc_)
            lidar_loc_->SetInitialPose(SE3(q, t));
    }

    void Localization::HandlePendingReset()
    {
        if (!lidar_loc_ || !lidar_loc_->need_reset_)
            return;

        Reset();
    }

    void Localization::Reset()
    {
        LOG(ERROR) << get_time_us() << "[ERROR][Localization::[" << __FUNCTION__ << "] " << "WAITING Reset ";
        lidar_loc_->Reset();
        pgo_->Reset();
        lio_->Reset(lidar_loc_->reset_pose_);
        lio_path_align_set_ = false;
        last_dr_dbg_set_ = false;
    }

} // namespace lightning::loc
