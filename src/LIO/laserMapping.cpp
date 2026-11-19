#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include "lio/laser_mapping.h"
#include "lightning_math.hpp"
#include <Eigen/Geometry>
#include "common/constant.h"
#include <pcl/common/transforms.h>
#include <csignal>
#include <cctype>
#include <yaml_loader.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "pub/pub_ROS1.h"

namespace
{
    std::string Lowercase(std::string text)
    {
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) { return std::tolower(ch); });
        return text;
    }

    bool IsFisheyeModel(const std::string &model)
    {
        const std::string lower = Lowercase(model);
        return lower == "fisheye" || lower == "equidistant" || lower == "equidistantcamera";
    }

    double DistortionAt(const lightning::CameraSlot &camera, size_t idx)
    {
        return idx < camera.distortion.size() ? camera.distortion[idx] : 0.0;
    }

    YAML::Node CameraSlotToVikitYaml(const lightning::CameraSlot &slot)
    {
        YAML::Node node;
        const std::string lower = Lowercase(slot.model);
        if (IsFisheyeModel(slot.model))
            node["cam_model"] = "EquidistantCamera";
        else if (lower.empty() || lower == "pinhole")
            node["cam_model"] = "Pinhole";
        else
            node["cam_model"] = slot.model;
        node["cam_width"] = slot.width;
        node["cam_height"] = slot.height;
        node["scale"] = slot.scale;
        node["cam_fx"] = slot.fx;
        node["cam_fy"] = slot.fy;
        node["cam_cx"] = slot.cx;
        node["cam_cy"] = slot.cy;
        if (IsFisheyeModel(slot.model))
        {
            node["k1"] = DistortionAt(slot, 0);
            node["k2"] = DistortionAt(slot, 1);
            node["k3"] = DistortionAt(slot, 2);
            node["k4"] = DistortionAt(slot, 3);
        }
        else
        {
            node["cam_d0"] = DistortionAt(slot, 0);
            node["cam_d1"] = DistortionAt(slot, 1);
            node["cam_d2"] = DistortionAt(slot, 2);
            node["cam_d3"] = DistortionAt(slot, 3);
            node["cam_d4"] = DistortionAt(slot, 4);
        }
        return node;
    }

    lightning::Mat3d Mat3FromVector(const std::vector<double> &v)
    {
        lightning::Mat3d m = lightning::Mat3d::Identity();
        if (v.size() == 9)
        {
            m << v[0], v[1], v[2],
                 v[3], v[4], v[5],
                 v[6], v[7], v[8];
        }
        return m;
    }

    lightning::Vec3d Vec3FromVector(const std::vector<double> &v)
    {
        if (v.size() == 3)
            return lightning::Vec3d(v[0], v[1], v[2]);
        return lightning::Vec3d::Zero();
    }

    lightning::SE3 SE3FromRT(const std::vector<double> &R, const std::vector<double> &t)
    {
        return lightning::SE3(lightning::SO3(lightning::Quatd(Mat3FromVector(R))), Vec3FromVector(t));
    }

    void FillStatesFromNav(const lightning::NavState &n, StatesGroup &s)
    {
        s.rot_end = n.rot_.matrix();
        s.pos_end = n.pos_;
        s.vel_end = n.vel_;
        s.bias_g = n.bg_;
        s.bias_a = n.ba_;
        s.gravity = n.grav_.vec_;
        s.inv_expo_time = n.inv_expo_time_;
    }

    double ElapsedMs(const std::chrono::steady_clock::time_point &a, const std::chrono::steady_clock::time_point &b)
    {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }

}

namespace lightning
{
    LaserMapping::LaserMapping(Options options) : options_(options)
    {
        p_pre_.reset(new PointCloudPreprocess());
        p_imu_.reset(new ImuProcess());
    }

    LaserMapping::~LaserMapping()
    {
        scan_down_body_ = nullptr;
        scan_undistort_ = nullptr;
        scan_down_world_ = nullptr;
        LOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] " << "laser mapping deconstruct";
    }

    bool LaserMapping::Init(const Config &cfg, const rclcpp::Node::SharedPtr &nh)
    {
        LOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] " << "init laser mapping from ";

        if (!ApplyConfig(cfg))
            return false;

        // esekf init
        ESKF::Options eskf_options;
        eskf_options.max_iterations_ = use_voxel_map_ ? lio_max_iterations_ : fasterlio::NUM_MAX_ITERATIONS;
        eskf_options.degeneracy_threshold_ratio_ = fasterlio::degeneracy_threshold_ratio;
        eskf_options.epsi_ = 1e-3 * ESKF::StateVecType::Ones();
        eskf_options.use_degeneracy_ = fasterlio::use_degeneracy;
        eskf_options.lidar_obs_func_ = [this](NavState &s, ESKF::CustomObservationModel &obs)
        { ObsModel(s, obs); };
        eskf_options.acc_as_gravity_obs_func_ = [this](NavState &s, ESKF::CustomObservationModel &obs)
        { AccAsGravityObsModel(s, obs); };
        eskf_options.bias_obs_func_ = [this](NavState &s, ESKF::CustomObservationModel &obs)
        { BiasObsModel(s, obs); };
        eskf_options.stationary_bias_and_acc_obs_func_ = [this](NavState &s, ESKF::CustomObservationModel &obs)
        { StationaryBiasAndAccObsModel(s, obs); };
        eskf_options.use_aa_ = use_aa_;
        eskf_options.exposure_estimate_en_ = visual_en_ && vio_exposure_estimate_en_;
        eskf_options.inv_expo_cov_ = inv_expo_cov_;

        kf_.Init(eskf_options);
        kf_imu_.Init(eskf_options);

        if (use_voxel_map_)
        {
            voxel_map_ = std::make_shared<VoxelMap>(voxel_map_config_);
            voxel_map_->SetExtrinsic(math::MatFromArray<double>(extrinR_), math::VecFromArray<double>(extrinT_));
            LOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] "
                      << (visual_en_ ? "LIVO VoxelMap + FAST VIO" : "VoxelMap pure LIO (VIO off)")
                      << ", voxel_size=" << voxel_map_config_.max_voxel_size
                      << ", max_layer=" << voxel_map_config_.max_layer
                      << ", max_iter=" << lio_max_iterations_
                      << ", lru_capacity=" << voxel_map_config_.lru_capacity;
        }
        else
        {
            ivox_ = std::make_shared<IVoxType>(ivox_options_);
        }

        if (visual_en_ || buffer_images_for_dyn_)
        {
            image_buffers_.resize(std::max(1, camera_num_));
            last_timestamp_img_.assign(image_buffers_.size(), -1.0);
        }

        if (visual_en_)
        {
            if (camera_slots_.empty())
            {
                LOG(ERROR) << get_time_us() << "[ERROR][LaserMapping::" << __FUNCTION__ << "] "
                           << "visual.enable=true but no camera slot was loaded";
                return false;
            }
            vio_manager_.reset(new VIOManager());
            if (!vk::camera_loader::loadFromYaml(CameraSlotToVikitYaml(camera_slots_[0]), vio_manager_->cam))
            {
                LOG(ERROR) << get_time_us() << "[ERROR][LaserMapping::" << __FUNCTION__ << "] "
                           << "failed to load FAST VIO camera from yaml (cam_model=" << camera_slots_[0].model
                           << ")";
                return false;
            }
            vio_manager_->grid_size = vio_grid_size_;
            vio_manager_->patch_size = vio_patch_size_;
            vio_manager_->outlier_threshold = vio_outlier_threshold_;
            if (camera_Rcl_.size() != 9)
                camera_Rcl_ = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            if (camera_Pcl_.size() != 3)
                camera_Pcl_ = {0, 0, 0};
            vio_manager_->setImuToLidarExtrinsic(math::VecFromArray<double>(extrinT_),
                                                 math::MatFromArray<double>(extrinR_));
            vio_manager_->setLidarToCameraExtrinsic(camera_Rcl_, camera_Pcl_);
            vio_manager_->state = &vio_state_;
            vio_manager_->state_propagat = &vio_state_propagat_;
            vio_manager_->max_iterations = vio_max_iterations_;
            vio_manager_->img_point_cov = vio_img_point_cov_;
            vio_manager_->normal_en = vio_normal_en_;
            vio_manager_->inverse_composition_en = vio_inverse_composition_en_;
            vio_manager_->raycast_en = vio_raycast_en_;
            vio_manager_->grid_n_height = vio_grid_n_height_;
            vio_manager_->patch_pyrimid_level = vio_patch_pyrimid_level_;
            vio_manager_->exposure_estimate_en = vio_exposure_estimate_en_;
            vio_manager_->ncc_en = vio_ncc_en_;
            vio_manager_->ncc_thre = vio_ncc_threshold_;
            const int visual_lru_capacity = cfg.visual_lru_capacity;
            vio_manager_->setFeatLruCapacity(static_cast<std::size_t>(std::max(0, visual_lru_capacity)));
            const int visual_max_points_per_voxel = cfg.visual_max_points_per_voxel;
            vio_manager_->setFeatMaxPointsPerVoxel(
                static_cast<std::size_t>(std::max(0, visual_max_points_per_voxel)));
            vio_manager_->initializeVIO();
            LOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] "
                      << "visual feat_map lru_capacity=" << visual_lru_capacity
                      << " max_points_per_voxel=" << visual_max_points_per_voxel;
        }

        if (buffer_images_for_dyn_)
        {
            std::vector<perception::DynCameraModel> dyn_cams;
            const SE3 T_imu_lidar = SE3FromRT(extrinR_, extrinT_);
            dyn_cams.reserve(camera_slots_.size());
            for (const auto &slot : camera_slots_)
            {
                perception::DynCameraModel cam;
                cam.cam_id = slot.cam_id;
                cam.model = slot.model;
                cam.fx = slot.fx;
                cam.fy = slot.fy;
                cam.cx = slot.cx;
                cam.cy = slot.cy;
                cam.width = slot.width;
                cam.height = slot.height;
                cam.scale = slot.scale;
                cam.distortion = slot.distortion;
                cam.camera_loader_ns = slot.camera_loader_ns;
                // slot.T_cam_imu = T_cam_lidar * T_lidar_imu  →  T_cam_lidar = T_cam_imu * T_imu_lidar
                cam.T_cam_lidar = slot.T_cam_imu * T_imu_lidar;
                dyn_cams.push_back(cam);
            }
            dynamic_filter_ = std::make_unique<perception::DynamicFilter>();
            if (!dynamic_filter_->Init(cfg.dynamic_filter, dyn_cams) || !dynamic_filter_->Enabled())
            {
                LOG(WARNING) << get_time_us() << "[WARN][LaserMapping::" << __FUNCTION__ << "] "
                             << "dynamic_filter enable requested but not ready; image buffering off";
                buffer_images_for_dyn_ = false;
                dynamic_filter_.reset();
            }
            else if (dynamic_filter_->PublishRemovedCloud())
            {
                pub_dyn_removed_body_ = nh->create_publisher<sensor_msgs::msg::PointCloud2>(
                    dynamic_filter_->RemovedBodyTopic(), ReliableQos(1));
                pub_dyn_removed_world_ = nh->create_publisher<sensor_msgs::msg::PointCloud2>(
                    dynamic_filter_->RemovedWorldTopic(), ReliableQos(1));
            }
        }

        LOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] " << "Using lidar type: " << std::to_string(p_pre_->lidar_type_);
        LOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] " << "SLAM initialization successful";

        return true;
    }

    bool LaserMapping::ApplyConfig(const Config &cfg)
    {
        Mat3d lidar_R_wrt_IMU;
        Vec3d lidar_T_wrt_IMU;

        enable_debug_log_ = cfg.enable_debug_log;

        path_en_ = cfg.path_en;
        scan_pub_en_ = cfg.scan_pub_en;
        dense_pub_en_ = cfg.dense_pub_en;
        scan_body_pub_en_ = cfg.scan_body_pub_en;
        rgb_img_pub_en_ = cfg.rgb_img_pub_en;
        rgb_cloud_pub_en_ = cfg.rgb_cloud_pub_en;
        blind_rgb_points_ = cfg.blind_rgb_points;
        visual_en_ = cfg.visual_en;
        use_voxel_map_ = cfg.use_voxel_map;
        lio_max_iterations_ = cfg.lio_max_iterations;
        buffer_images_for_dyn_ = cfg.dynamic_filter_enable;
        camera_num_ = cfg.camera_num;
        image_time_offset_ = cfg.image_time_offset;
        exposure_time_init_ = cfg.exposure_time_init;
        vio_raycast_en_ = cfg.vio_raycast_en;
        vio_normal_en_ = cfg.vio_normal_en;
        vio_exposure_estimate_en_ = cfg.vio_exposure_estimate_en;
        vio_inverse_composition_en_ = cfg.vio_inverse_composition_en;
        inv_expo_cov_ = cfg.inv_expo_cov;
        vio_grid_size_ = cfg.vio_grid_size;
        vio_grid_n_height_ = cfg.vio_grid_n_height;
        vio_patch_size_ = cfg.vio_patch_size;
        vio_patch_pyrimid_level_ = cfg.vio_patch_pyrimid_level;
        vio_max_iterations_ = cfg.vio_max_iterations;
        vio_outlier_threshold_ = cfg.vio_outlier_threshold;
        vio_img_point_cov_ = cfg.vio_img_point_cov;
        vio_ncc_en_ = cfg.vio_ncc_en;
        vio_ncc_threshold_ = cfg.vio_ncc_threshold;
        options_.kf_dis_th_ = cfg.kf_dis_th;
        options_.kf_angle_th_ = cfg.kf_angle_th;
        options_.plane_icp_weight_ = cfg.plane_icp_weight;
        options_.enable_icp_part_ = cfg.enable_icp_part;
        options_.proj_kfs_ = cfg.proj_kfs;
        options_.min_pts = cfg.min_pts;
        use_aa_ = cfg.use_aa;
        extrinT_ = cfg.extrin_T;
        extrinR_ = cfg.extrin_R;
        camera_slots_ = cfg.camera_slots;
        camera_Rcl_ = cfg.camera_Rcl;
        camera_Pcl_ = cfg.camera_Pcl;
        omp_num_ = cfg.omp_num;
        filter_size_map_min_ = cfg.filter_size_map;
        dense_pcd_save_en_ = cfg.dense_pcd_save_en;
        imu_reset_level_ = cfg.imu_reset_level;
        lidar_type_ = cfg.lidar_type;

        p_pre_->blind_ = cfg.blind;
        p_pre_->point_filter_num_ = cfg.point_filter_num;
        p_pre_->N_SCANS_ = cfg.scan_line;
        p_pre_->SetHeightROI(cfg.height_max, cfg.height_min);
        p_imu_->setMaxInitCount(cfg.max_init_count);
        p_imu_->SetUseIMUFilter(cfg.use_imu_filter);

        T_body_lio_ = cfg.T_body_lio;

        LOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] " << "lidar_type " << lidar_type_;

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
        {
            LOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] " << "unknown lidar_type";
            return false;
        }

        const int ivox_nearby_type = cfg.ivox_nearby_type;
        if (ivox_nearby_type == 0)
            ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
        else if (ivox_nearby_type == 6)
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
        else if (ivox_nearby_type == 18)
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
        else if (ivox_nearby_type == 26)
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
        else
        {
            LOG(WARNING) << get_time_us() << "[WARN][LaserMapping::" << __FUNCTION__ << "] " << "unknown ivox_nearby_type, use NEARBY18";
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
        }
        ivox_options_.resolution_ = cfg.ivox_grid_resolution;
        ivox_options_.capacity_ = cfg.ivox_capacity;

        voxel_map_config_.max_layer = cfg.max_layer;
        voxel_map_config_.max_voxel_size = cfg.voxel_size;
        voxel_map_config_.planner_threshold = cfg.min_eigen_value;
        voxel_map_config_.sigma_num = cfg.sigma_num;
        voxel_map_config_.beam_err = cfg.beam_err;
        voxel_map_config_.dept_err = cfg.dept_err;
        voxel_map_config_.max_points_num = cfg.max_points_num;
        voxel_map_config_.layer_init_num = cfg.layer_init_num;
        voxel_map_config_.omp_num = omp_num_;
        voxel_map_config_.lru_capacity = static_cast<std::size_t>(std::max(0, cfg.voxel_lru_capacity));

        fasterlio::NUM_MAX_ITERATIONS = cfg.max_iteration;
        fasterlio::degeneracy_threshold_ratio = cfg.degeneracy_threshold_ratio;
        fasterlio::ESTI_PLANE_THRESHOLD = cfg.esti_plane_threshold;
        fasterlio::use_degeneracy = cfg.use_degeneracy;

        voxel_scan_.setLeafSize(cfg.filter_size_surf, cfg.filter_size_surf, cfg.filter_size_surf);

        lidar_T_wrt_IMU = math::VecFromArray<double>(extrinT_);
        lidar_R_wrt_IMU = math::MatFromArray<double>(extrinR_);
        p_imu_->setExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
        p_imu_->setGyrCov(Vec3d(cfg.gyr_cov, cfg.gyr_cov, cfg.gyr_cov));
        p_imu_->setAccCov(Vec3d(cfg.acc_cov, cfg.acc_cov, cfg.acc_cov));
        p_imu_->setGyrBiasCov(Vec3d(cfg.b_gyr_cov, cfg.b_gyr_cov, cfg.b_gyr_cov));
        p_imu_->setAccBiasCov(Vec3d(cfg.b_acc_cov, cfg.b_acc_cov, cfg.b_acc_cov));

        return true;
    }

    bool LaserMapping::SyncPackages()
    {
        if (lidar_buffer_.empty() || imu_buffer_.empty())
        {
            LOG(INFO) << "lidar or imu is empty";
            return false;
        }

        /*** push a lidar scan ***/
        if (!lidar_pushed_)
        {
            measures_.scan_ = lidar_buffer_.front();
            measures_.lidar_begin_time_ = time_buffer_.front();

            // 可能存在异常 看下时间戳是不是有坑 /1000
            if (measures_.scan_->points.size() <= 1)
            {
                LOG(ERROR) << get_time_us() << "[ERROR][LaserMapping::" << __FUNCTION__ << "] " << "Too few input point cloud!";
                lidar_end_time_ = measures_.lidar_begin_time_ + lidar_mean_scantime_;
            }
            else if (measures_.scan_->points.back().time / double(1000) < 0.5 * lidar_mean_scantime_)
            {
                lidar_end_time_ = measures_.lidar_begin_time_ + lidar_mean_scantime_;
            }
            else
            {
                scan_num_++;
                lidar_end_time_ = measures_.lidar_begin_time_ + measures_.scan_->points.back().time / double(1000);
                lidar_mean_scantime_ +=
                    (measures_.scan_->points.back().time / double(1000) - lidar_mean_scantime_) / scan_num_;

                if ((lidar_end_time_ - measures_.lidar_begin_time_) > 5 * lo::lidar_time_interval)
                {
                    /// timestamp 有异常
                    LOG(ERROR) << get_time_us() << "[ERROR][LaserMapping::" << __FUNCTION__ << "] " 
                                << "lidar frame error: " << lidar_end_time_ - measures_.lidar_begin_time_;
                    lidar_end_time_ = measures_.lidar_begin_time_ + lo::lidar_time_interval;
                    lidar_mean_scantime_ = lo::lidar_time_interval;
                }
            }

            lo::lidar_time_interval = lidar_mean_scantime_;

            measures_.lidar_end_time_ = lidar_end_time_;
            lidar_pushed_ = true;
        }

        // TODO: 对 SyncPackages 的调用不应随着 lidar 帧触发，否则每触发一次下面的 if 都会导致 lidar 缓存多一帧且不会被处理
        if (last_timestamp_imu_ < lidar_end_time_)
        {
            need_trigger_imu = true;
            return false;
        }
        /*** push imu_ data, and pop from imu_ buffer ***/
        {
            std::lock_guard<std::mutex> lock(mtx_buffer_);
            double imu_time = imu_buffer_.front()->timestamp;
            measures_.imu_.clear();
            while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_))
            {
                imu_time = imu_buffer_.front()->timestamp;
                if (imu_time > lidar_end_time_)
                    break;

                measures_.imu_.push_back(imu_buffer_.front());
                imu_buffer_.pop_front();
            }
        }

        lidar_buffer_.pop_front();
        time_buffer_.pop_front();
        lidar_pushed_ = false;

        // LOG(INFO) << get_time_us() <<  "[INFO][LaserMapping::" << __FUNCTION__ << "] " << "sync: " << std::setprecision(14) << measures_.lidar_begin_time_ << ", " <<
        // measures_.lidar_end_time_;

        return true;
    }

    bool LaserMapping::SyncLIVOPackages(LivoStep &step)
    {
        std::lock_guard<std::mutex> lock(mtx_buffer_);
        if (image_buffers_.empty())
            return false;

        constexpr double kImageSyncTolerance = 0.005;
        constexpr double kTimeEps = 1e-5;

        // Drop one hard-synced image set from every camera (multi-cam analogue of FAST's
        // single img_buffer.pop_front when throwing a stale frame).
        auto pop_synced_image_set = [&](double target_img_time) {
            for (auto &buffer : image_buffers_)
            {
                if (!buffer.empty() &&
                    std::abs(buffer.front().timestamp - target_img_time) <= kImageSyncTolerance)
                {
                    buffer.pop_front();
                }
            }
        };

        // Align buffers to a hard-sync stamp; optionally take Mats (VIO step only, like FAST).
        auto align_cameras_to = [&](double target_img_time, bool take_images) -> bool {
            if (take_images)
            {
                synced_images_.clear();
                synced_images_.reserve(image_buffers_.size());
            }
            for (auto &buffer : image_buffers_)
            {
                while (buffer.size() > 1 && buffer.front().timestamp < target_img_time - kImageSyncTolerance)
                    buffer.pop_front();

                if (buffer.empty() ||
                    std::abs(buffer.front().timestamp - target_img_time) > kImageSyncTolerance)
                {
                    if (take_images)
                        synced_images_.clear();
                    return false;
                }
                if (take_images)
                    synced_images_.push_back(buffer.front().image);
            }
            return true;
        };

        if (last_livo_step_ == LivoStep::LIO)
        {
            // FAST sync_packages: VIO also returns false while lidar or IMU is empty.
            // The next scan is not consumed here; LIO packaging uses it on the next step.
            if (lidar_buffer_.empty() || imu_buffer_.empty())
                return false;

            // Match FAST-LIVO2: take images only on the VIO step (not during LIO packaging).
            const double target_img_time = synced_image_time_ - exposure_time_init_;
            if (!align_cameras_to(target_img_time, /*take_images=*/true))
            {
                LOG(ERROR) << get_time_us() << "[ERROR][LaserMapping::SyncLIVOPackages] "
                           << "VIO step missing hard-synced images, synced_image_time="
                           << synced_image_time_
                           << "; abandoning orphaned LIO pairing so a new LIO package can form";
                last_livo_step_ = LivoStep::VIO;
                return false;
            }

            for (auto &buffer : image_buffers_)
                buffer.pop_front();

            step = LivoStep::VIO;
            last_livo_step_ = step;
            return true;
        }

        // LIO packaging needs a queued scan+IMU to cut and to stitch leftover.
        if (lidar_buffer_.empty() || imu_buffer_.empty())
            return false;

        auto &base_img_buffer = image_buffers_.front();
        if (base_img_buffer.empty())
            return false;

        const double img_time = base_img_buffer.front().timestamp;
        const double img_capture_time = img_time + exposure_time_init_;

        // LIO step: timestamp hard-sync check only; Mats stay in buffers until VIO.
        if (!align_cameras_to(img_time, /*take_images=*/false))
        {
            LOG(ERROR) << get_time_us() << "[ERROR][LaserMapping::SyncLIVOPackages] "
                       << "multi-camera image sync failed, target_img_time=" << img_time
                       << ", tolerance=" << kImageSyncTolerance;
            return false;
        }

        if (last_livo_update_time_ < 0.0)
            last_livo_update_time_ = time_buffer_.front();

        // FAST: throw exactly one stale image and wait for the next sync call.
        if (img_capture_time < last_livo_update_time_ + kTimeEps)
        {
            pop_synced_image_set(img_time);
            LOG(ERROR) << get_time_us() << "[ERROR][LaserMapping::SyncLIVOPackages] "
                       << "Throw one image frame, img_capture_time=" << img_capture_time
                       << " last_livo_update_time=" << last_livo_update_time_;
            return false;
        }

        const auto &newest_lidar = lidar_buffer_.back();
        if (newest_lidar == nullptr || newest_lidar->empty())
        {
            LOG(ERROR) << get_time_us() << "[ERROR][LaserMapping::SyncLIVOPackages] "
                       << "newest lidar frame is null or empty";
            return false;
        }
        // FAST lid_newest_time uses only the newest queued scan, not leftover pcl_proc_next.
        const double lidar_newest_time =
            time_buffer_.back() + newest_lidar->points.back().time / double(1000);
        if (img_capture_time > lidar_newest_time || imu_buffer_.back()->timestamp < img_capture_time)
            return false;

        // FAST Data Cut: move leftover into pcl_proc_cur, then consume queued
        // scans whose header <= img_capture_time. If the next scan header is
        // already after the image, FAST still emits LIO on leftover alone.
        CloudPtr pcl_proc_cur(new PointCloudType());
        *pcl_proc_cur = *livo_pcl_proc_next_;
        livo_pcl_proc_next_->clear();

        while (!lidar_buffer_.empty())
        {
            if (time_buffer_.front() > img_capture_time)
                break;

            const auto frame = lidar_buffer_.front();
            const double frame_header_time = time_buffer_.front();
            const double max_offset_time_ms = (img_capture_time - frame_header_time) * 1000.0;

            for (const auto &pt : frame->points)
            {
                PointType cut_pt = pt;
                if (pt.time < max_offset_time_ms)
                {
                    cut_pt.time += (frame_header_time - last_livo_update_time_) * 1000.0;
                    pcl_proc_cur->points.push_back(cut_pt);
                }
                else
                {
                    cut_pt.time += (frame_header_time - img_capture_time) * 1000.0;
                    livo_pcl_proc_next_->points.push_back(cut_pt);
                }
            }

            lidar_buffer_.pop_front();
            time_buffer_.pop_front();
            lidar_pushed_ = false;
        }

        pcl_proc_cur->width = pcl_proc_cur->size();
        pcl_proc_cur->height = 1;
        pcl_proc_cur->is_dense = false;
        livo_pcl_proc_next_->width = livo_pcl_proc_next_->size();
        livo_pcl_proc_next_->height = 1;
        livo_pcl_proc_next_->is_dense = false;

        measures_.scan_ = pcl_proc_cur;
        measures_.lidar_begin_time_ = last_livo_update_time_;
        measures_.lidar_end_time_ = img_capture_time;
        lidar_end_time_ = img_capture_time;
        synced_image_time_ = img_capture_time;

        measures_.imu_.clear();
        while (!imu_buffer_.empty() && imu_buffer_.front()->timestamp <= img_capture_time)
        {
            if (imu_buffer_.front()->timestamp > last_livo_update_time_)
                measures_.imu_.push_back(imu_buffer_.front());
            imu_buffer_.pop_front();
        }

        step = LivoStep::LIO;
        last_livo_step_ = step;
        return true;
    }

    void CalcBodyCov(Vec3d& point_body, double range_inc, double degree_inc, Mat3d& cov)
    {
        if (point_body.z() == 0.0) {
            point_body.z() = 0.0001;
        }
        const double range = point_body.norm();
        const double range_var = range_inc * range_inc;
        Eigen::Matrix2d direction_var;
        direction_var << std::pow(std::sin(degree_inc * constant::kDEG2RAD), 2), 0.0, 0.0,
            std::pow(std::sin(degree_inc * constant::kDEG2RAD), 2);
        Vec3d direction = point_body.normalized();
        Mat3d direction_hat = math::SKEW_SYM_MATRIX(direction);
        Vec3d base_vector1(1.0, 1.0, -(direction.x() + direction.y()) / direction.z());
        base_vector1.normalize();
        const Vec3d base_vector2 = base_vector1.cross(direction).normalized();
        Eigen::Matrix<double, 3, 2> N;
        N << base_vector1.x(), base_vector2.x(), base_vector1.y(), base_vector2.y(), base_vector1.z(),
            base_vector2.z();
        const Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
        cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
    }

    PointWithVar LaserMapping::BuildPointWithVar(const PointType &point_body)
    {
        constexpr double kDeptErr = 0.02;
        constexpr double kBeamErr = 0.05;

        const Mat3d R_wi = state_point_.rot_.matrix();
        const Mat3d R_wl = (state_point_.rot_ * state_point_.offset_R_lidar_).matrix();
        const Vec3d t_wl = state_point_.rot_ * state_point_.offset_t_lidar_ + state_point_.pos_;
        const Mat3d off_R = state_point_.offset_R_lidar_.matrix();
        const Vec3d off_t = state_point_.offset_t_lidar_;
        const Mat3d rot_var = kf_.GetP().block<3, 3>(3, 3);
        const Mat3d t_var = kf_.GetP().block<3, 3>(0, 0);

        PointWithVar point;
        point.point_b = point_body.getVector3fMap().cast<double>();
        point.point_i = off_R * point.point_b + off_t;
        point.point_w = R_wl * point.point_b + t_wl;
        point.point_crossmat = math::SKEW_SYM_MATRIX(point.point_i);
        Vec3d body_point = point.point_b;
        CalcBodyCov(body_point, kDeptErr, kBeamErr, point.body_var);
        const Mat3d R_wl_cov = R_wi * off_R;
        point.var = R_wl_cov * point.body_var * R_wl_cov.transpose() +
                    (-point.point_crossmat) * rot_var * (-point.point_crossmat.transpose()) + t_var;
        return point;
    }

    std::vector<PointWithVar> LaserMapping::ExportPgForVio()
    {
        if (voxel_map_)
            return voxel_map_->pv_list();
        return {};
    }

    void LaserMapping::RunFastVio(cv::Mat image)
    {
        if (!vio_manager_ || image.empty())
            return;

        ESKF::CovType P = kf_.GetP();
        // FAST VIO: Process2 无 IMU（空包），随后 state_propagat = _state，即 LIO 后验。
        livo_propagated_state_ = state_point_;
        FillStatesFromNav(state_point_, vio_state_);
        FillStatesFromNav(livo_propagated_state_, vio_state_propagat_);
        vio_manager_->state = &vio_state_;
        vio_manager_->state_propagat = &vio_state_propagat_;
        vio_manager_->nav_state = &state_point_;
        vio_manager_->nav_state_propagat = &livo_propagated_state_;
        vio_manager_->nav_P = &P;

        static const std::unordered_map<VoxelLocation, VoxelOctoTree *> kEmptyPlaneMap;
        const auto &plane_map = voxel_map_ ? voxel_map_->voxel_map() : kEmptyPlaneMap;
        const Vec3d pos_before = state_point_.pos_;
        const SO3 rot_before = state_point_.rot_;
        const auto t_pf0 = std::chrono::steady_clock::now();
        vio_manager_->processFrame(image, pg_list_, plane_map, synced_image_time_);
        const double process_frame_ms = ElapsedMs(t_pf0, std::chrono::steady_clock::now());

        const double dpos = (state_point_.pos_ - pos_before).norm();
        const double drot_deg = (rot_before.inverse() * state_point_.rot_).log().norm() * 180.0 / M_PI;
        if (enable_debug_log_ || dpos > 0.5 || drot_deg > 5.0)
            LOG(INFO) << get_time_us() << "[INFO][LaserMapping::RunFastVio] "
                      << "VIO dpos=" << dpos << " m, drot=" << drot_deg << " deg, expo=" << state_point_.inv_expo_time_
                      << ", processFrame_ms=" << process_frame_ms;
        kf_.ChangeX(state_point_);
        kf_.ChangeP(P);
    }

    void LaserMapping::ProcessIMU(const lightning::IMUPtr &msg_in)
    {
        publish_count_++;
        std::lock_guard<std::mutex> lock(mtx_buffer_);
        lightning::IMUPtr cur_msg = msg_in;

        // imu 滤波
        if (!IMUCheck(cur_msg))
        {
            // cur_msg->angular_velocity = last_imu_->angular_velocity;
            // cur_msg->linear_acceleration.y() = last_imu_->linear_acceleration.y();
            // cur_msg->linear_acceleration.z() = last_imu_->linear_acceleration.z();
            // cur_msg->angular_velocity.x() = last_imu_->angular_velocity.x();
        }

        if (p_imu_->IsIMUInited())
        {
            kf_imu_.Predict(cur_timestamp_imu_ - last_timestamp_imu_, p_imu_->Q_, cur_msg->angular_velocity, cur_msg->linear_acceleration);
        }

        last_timestamp_imu_ = cur_timestamp_imu_;
        last_imu_ = cur_msg;

        imu_buffer_.emplace_back(cur_msg);
    }

    void LaserMapping::ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
    {
        std::lock_guard<std::mutex> lock(mtx_buffer_);

        scan_count_++;
        double timestamp = SecFromStamp(msg->header.stamp);
        if (timestamp < last_timestamp_lidar_)
        {
            LOG(ERROR) << get_time_us() << "[ERROR][LaserMapping::" << __FUNCTION__ << "] "
                       << "lidar loop back, dt: " << timestamp - last_timestamp_lidar_;
            // return;
            lidar_buffer_.clear();
        }

        // if (enable_debug_log_)
        //     LOG(INFO) << get_time_us() << "[DEBUG][LaserMapping::" << __FUNCTION__ << "] "
        //               << "get cloud at " << std::setprecision(14) << timestamp

        CloudPtr cloud(new PointCloudType());
        p_pre_->Process(msg, cloud);

        lidar_buffer_.push_back(cloud);
        time_buffer_.push_back(timestamp);
        last_timestamp_lidar_ = timestamp;
    }

    void LaserMapping::ProcessPointCloud2(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg)
    {
        std::lock_guard<std::mutex> lock(mtx_buffer_);

        scan_count_++;
        double timestamp = SecFromStamp(msg->header.stamp);
        if (timestamp < last_timestamp_lidar_)
        {
            LOG(ERROR) << get_time_us() << "[ERROR][LaserMapping::" << __FUNCTION__ << "] " << "lidar loop back, clear buffer";
            lidar_buffer_.clear();
        }

        CloudPtr cloud(new PointCloudType());
        p_pre_->Process(msg, cloud);

        lidar_buffer_.push_back(cloud);
        time_buffer_.push_back(timestamp);
        last_timestamp_lidar_ = timestamp;
    }

    void LaserMapping::ProcessPointCloud2(CloudPtr cloud)
    {
        std::lock_guard<std::mutex> lock(mtx_buffer_);

        scan_count_++;
        double timestamp = math::ToSec(cloud->header.stamp);
        if (timestamp < last_timestamp_lidar_)
        {
            LOG(ERROR) << get_time_us() << "[ERROR][LaserMapping::" << __FUNCTION__ << "] " << "lidar loop back, clear buffer";
            lidar_buffer_.clear();
        }

        lidar_buffer_.push_back(cloud);
        time_buffer_.push_back(timestamp);
        last_timestamp_lidar_ = timestamp;
    }

    void LaserMapping::TryFilterDynamicObjects(double stamp)
    {
        if (!dynamic_filter_ || !dynamic_filter_->Enabled() || !scan_undistort_)
            return;

        std::vector<cv::Mat> images;
        {
            std::lock_guard<std::mutex> lock(mtx_buffer_);
            if (image_buffers_.empty())
                return;
            images.resize(image_buffers_.size());
            for (size_t i = 0; i < image_buffers_.size(); ++i)
            {
                if (image_buffers_[i].empty())
                    continue;
                // Prefer hard-synced stamp when available; else latest frame.
                if (stamp > 0.0)
                {
                    double best_dt = 1e9;
                    for (const auto &fr : image_buffers_[i])
                    {
                        const double dt = std::abs(fr.timestamp - stamp);
                        if (dt < best_dt)
                        {
                            best_dt = dt;
                            images[i] = fr.image;
                        }
                    }
                    if (best_dt > 0.05)
                        images[i] = image_buffers_[i].back().image;
                }
                else
                {
                    images[i] = image_buffers_[i].back().image;
                }
            }
        }
        dynamic_filter_->FilterCloud(scan_undistort_, images, stamp);
        PublishDynamicRemovedCloud(stamp);
    }

    void LaserMapping::PublishDynamicRemovedCloud(double stamp)
    {
        if (!dynamic_filter_ || !dynamic_filter_->PublishRemovedCloud())
            return;
        const CloudPtr &removed = dynamic_filter_->LastRemovedCloud();
        if (!removed)
            return;

        const NavState state_backup = state_point_;
        state_point_ = kf_.GetX();

        const int size = static_cast<int>(removed->points.size());
        CloudPtr body(new PointCloudType(size, 1));
        CloudPtr world(new PointCloudType(size, 1));
        const Eigen::Isometry3d T_level = T_body_lio_.inverse();
        for (int i = 0; i < size; ++i)
        {
            PointBodyLidarToIMU(removed->points[i], body->points[i]);
            PointBodyToWorld(removed->points[i], world->points[i]);
            const Vec3d p = T_level * Vec3d(world->points[i].x, world->points[i].y, world->points[i].z);
            world->points[i].x = p.x();
            world->points[i].y = p.y();
            world->points[i].z = p.z();
        }
        state_point_ = state_backup;

        const auto ros_stamp = StampFromSec(stamp > 0.0 ? stamp : lidar_end_time_);
        if (pub_dyn_removed_body_)
        {
            sensor_msgs::msg::PointCloud2 msg;
            pcl::toROSMsg(*body, msg);
            msg.header.stamp = ros_stamp;
            msg.header.frame_id = "body";
            pub_dyn_removed_body_->publish(msg);
        }
        if (pub_dyn_removed_world_)
        {
            sensor_msgs::msg::PointCloud2 msg;
            pcl::toROSMsg(*world, msg);
            msg.header.stamp = ros_stamp;
            msg.header.frame_id = "world";
            pub_dyn_removed_world_->publish(msg);
        }
    }

    void LaserMapping::ProcessImage(int cam_id, const cv::Mat &image, double timestamp)
    {
        if (image.empty())
            return;
        if (!visual_en_ && !buffer_images_for_dyn_)
            return;

        timestamp += image_time_offset_;
        std::lock_guard<std::mutex> lock(mtx_buffer_);
        if (image_buffers_.empty())
            image_buffers_.resize(std::max(1, camera_num_));
        if (last_timestamp_img_.size() != image_buffers_.size())
            last_timestamp_img_.assign(image_buffers_.size(), -1.0);
        if (cam_id < 0 || cam_id >= static_cast<int>(image_buffers_.size()))
        {
            LOG(WARNING) << get_time_us() << "[WARN][LaserMapping::" << __FUNCTION__ << "] "
                         << "invalid camera id: " << cam_id;
            return;
        }

        if (visual_en_)
        {
            // FAST img_cbk gates. last_timestamp_img is per-camera so hard-synced
            // multi-cam frames with the same stamp are not dropped by the 20ms jump rule.
            if (last_timestamp_lidar_ < 0.0)
                return;
            const double last_img = last_timestamp_img_[cam_id];
            if (std::abs(timestamp - last_img) < 0.001)
                return;
            if (timestamp < last_img)
            {
                LOG(ERROR) << get_time_us() << "[ERROR][LaserMapping::" << __FUNCTION__ << "] "
                           << "image loop back on cam " << cam_id;
                return;
            }
            if (timestamp - last_img < 0.02)
            {
                LOG(WARNING) << get_time_us() << "[WARN][LaserMapping::" << __FUNCTION__ << "] "
                             << "Image need Jumps: " << timestamp;
                return;
            }
        }

        ImageFrame frame;
        frame.cam_id = cam_id;
        frame.timestamp = timestamp;
        frame.image = image.clone();
        image_buffers_[cam_id].emplace_back(std::move(frame));
        if (visual_en_)
            last_timestamp_img_[cam_id] = timestamp;
        else
        {
            // Dyn-only: FilterCloud only reads the buffer, SyncLIVOPackages never runs,
            // so without a cap every 1080p clone stays forever (~6MB * 15Hz).
            while (image_buffers_[cam_id].size() > 3)
                image_buffers_[cam_id].pop_front();
        }
    }

    bool LaserMapping::Run()
    {
        auto t1 = std::chrono::steady_clock::now();
        last_run_was_vio_ = false;
        LivoStep livo_step = LivoStep::NONE;
        if (visual_en_)
        {
            if (!SyncLIVOPackages(livo_step))
                return false;
        }
        else if (!SyncPackages())
            return false;

        if (visual_en_ && livo_step == LivoStep::VIO)
        {
            // FAST handleVIO: skip photometric if the previous LIO did not publish a world cloud
            // (IMU-init LIO / empty undistort). The VIO sync already popped the image.
            if (vio_manager_ && !synced_images_.empty() && livo_has_lio_cloud_)
            {
                const auto frame_begin = std::chrono::steady_clock::now();
                RunFastVio(synced_images_.front());
                livo_has_lio_cloud_ = false;
                const auto vio_end = std::chrono::steady_clock::now();

                if (pub_)
                {
                    pub_->SetPosestamp(GetExtrinsicRemappedState(state_point_));
                    pub_->PublishOdom(state_point_.timestamp_);
                    if (path_en_)
                        pub_->PublishPath(synced_image_time_ > 0.0 ? synced_image_time_ : state_point_.timestamp_);
                    if (rgb_img_pub_en_ || rgb_cloud_pub_en_)
                        PublishLivoVisualizations(pub_);
                }

                // VIO updates kf_/state_point_ (blue path). kf_imu_ was already
                // synced on the preceding LIO step; do not rewind again.
                const auto frame_end = std::chrono::steady_clock::now();
                const double vio_total_s =
                    std::chrono::duration_cast<std::chrono::duration<double>>(frame_end - t1).count();
                LOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] "
                          << "VIO use time: " << vio_total_s << " s"
                          << ", processFrame_ms=" << ElapsedMs(frame_begin, vio_end)
                          << ", total_ms=" << ElapsedMs(t1, frame_end);
                synced_images_.clear();
                last_run_was_vio_ = true;
                // Successful VIO: loc publishes aligned LIO path (blue) from kf_/state_point_.
                // Empty VIO below still returns false so it does not touch that path.
                return true;
            }
            else
            {
                // Leftover-empty LIO (or IMU-init) left no world cloud. FAST handleVIO
                // returns here without publishing. Returning false keeps loc from
                // ProcessLO/PGO-ing a duplicate last-LIO pose at the same timestamp.
                LOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] "
                          << "[ VIO ] No point!!!";
                synced_images_.clear();
                return false;
            }
        }

        p_imu_->Process(measures_, kf_, scan_undistort_);

        if (!p_imu_->IsIMUInited())
            return false;

        // IMU 初始化成功后的第一帧 scan_undistort_ 为空。
        if (!IMU_inited_)
        {
            IMU_inited_ = true;
            if (use_voxel_map_)
            {
                // FAST StatesGroup ctor: I*INIT_COV(0.01); expo/bg/ba/grav = 1e-5
                ESKF::CovType P = kf_.GetP();
                P.block<3, 3>(0, 0) = Mat3d::Identity() * 0.01;
                P.block<3, 3>(3, 3) = Mat3d::Identity() * 0.01;
                P.block<3, 3>(12, 12) = Mat3d::Identity() * 0.01;
                if (visual_en_)
                {
                    P.block<3, 3>(15, 15) = Mat3d::Identity() * 0.00001;
                    P.block<3, 3>(18, 18) = Mat3d::Identity() * 0.00001;
                    P.block<2, 2>(21, 21) = Eigen::Matrix2d::Identity() * 0.00001;
                    P(NavState::inv_expo_idx, NavState::inv_expo_idx) = 0.00001;
                }
                kf_.ChangeP(P);
                kf_imu_.ChangeP(P);
            }
            return false;
        }

        // FAST last_lio_update_time is written only in UndistortPcl, not during imu_need_init.
        if (visual_en_ && livo_step == LivoStep::LIO)
            last_livo_update_time_ = measures_.lidar_end_time_;

        if (scan_undistort_->empty() || (scan_undistort_ == nullptr))
        {
            LOG(WARNING) << get_time_us() << "[WARN][LaserMapping::" << __FUNCTION__ << "] " << "No point, skip this scan!";
            return false;
        }

        // YOLO 动态物剔除：去畸变后、下采样/匹配前（需要图像 buffer）
        if ((!visual_en_ || livo_step == LivoStep::LIO || livo_step == LivoStep::NONE) &&
            dynamic_filter_ && dynamic_filter_->Enabled())
        {
            const double filter_stamp = (visual_en_ && synced_image_time_ > 0.0)
                                            ? synced_image_time_
                                            : measures_.lidar_end_time_;
            TryFilterDynamicObjects(filter_stamp);
        }

        if (flg_first_scan_)
        {
            if (enable_debug_log_)
                LOG(INFO) << get_time_us() << "[DEBUG][LaserMapping::" << __FUNCTION__ << "] " << "first scan pts: " << scan_undistort_->size();

            first_lidar_time_ = measures_.lidar_end_time_;
            state_point_ = kf_.GetX();
            state_point_.timestamp_ = lidar_end_time_;
            flg_first_scan_ = false;

            if (!voxel_map_)
            {
                scan_down_world_->resize(scan_undistort_->size());
                for (int i = 0; i < scan_undistort_->size(); i++)
                    PointBodyToWorld(scan_undistort_->points[i], scan_down_world_->points[i]);
                ivox_->AddPoints(scan_down_world_->points);
                return true;
            }
            // VoxelMap LIO: FAST handleLIO 首帧 BuildVoxelMap 后立刻 StateEstimation，不在这里 return
        }

        // yxliang  2026.2.2
        // 删减了 lidar 帧跳帧方法，CPU 紧缺的时候可以参考 lidar_slam-lm 添加

        double lidar_frame_time = measures_.lidar_begin_time_ - last_lidar_time_;
        const double reset_gap = (last_reset_wall_time_ > 0) ? (rclcpp::Clock().now().seconds() - last_reset_wall_time_) : 1e9;
        const bool in_reset_grace = reset_gap < 5.0; // 增加到 5 秒
        if (last_lidar_time_ > 0 && lidar_frame_time > 0.5)
        {
            if (in_reset_grace)
            {
                LOG(WARNING) << get_time_us() << "[WARN][LaserMapping::" << __FUNCTION__ << "] "
                             << "reset grace period, skip lidar interruption warning. dt=" << lidar_frame_time;
            }
            else
            {
                LOG(ERROR) << get_time_us() << "[ERROR][LaserMapping::" << __FUNCTION__ << "] " << "检测到雷达断流，时长 " << lidar_frame_time;
            }
        }

        last_lidar_time_ = measures_.lidar_begin_time_;

        flg_EKF_inited_ = (measures_.lidar_begin_time_ - first_lidar_time_) >= fasterlio::INIT_TIME;

        /// downsample
        voxel_scan_.setInputCloud(scan_undistort_);
        voxel_scan_.filter(*scan_down_body_);

        int cur_pts = scan_down_body_->size();

        if (cur_pts < (scan_undistort_->size() * 0.06) || cur_pts < options_.min_pts)
        {
            /// 降采样太狠了,有效点数不够
            LOG(WARNING) << get_time_us() << "[WARN][LaserMapping::" << __FUNCTION__ << "] "
                         << "Too few points, Re-downsampling with voxel size of 0.4, scan num: " << scan_undistort_->size() << ", " << scan_down_body_->size();
            auto v = voxel_scan_;
            v.setLeafSize(0.4, 0.4, 0.4);
            v.setInputCloud(scan_undistort_);
            v.filter(*scan_down_body_);
            cur_pts = scan_down_body_->size();
        }

        if (cur_pts < 5)
        {
            LOG(WARNING) << get_time_us() << "[WARN][LaserMapping::" << __FUNCTION__ << "] "
                         << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_body_->size();
            return false;
        }

        scan_down_world_->resize(cur_pts);
        nearest_points_.resize(cur_pts);

        residuals_.resize(cur_pts, 0);
        point_selected_surf_.resize(cur_pts, 1);
        point_selected_icp_.resize(cur_pts, 1);
        plane_coef_.resize(cur_pts, Vec4f::Zero());

        if (voxel_map_ && voxel_map_->NumRootVoxels() == 0)
        {
            state_point_ = kf_.GetX();
            for (int i = 0; i < cur_pts; ++i)
                PointBodyToWorld(scan_down_body_->points[i], scan_down_world_->points[i]);
            voxel_map_->BuildVoxelMap(scan_down_body_, scan_down_world_, state_point_, kf_.GetP());
        }

        auto old_state = kf_.GetX();
        if (voxel_map_)
        {
            livo_propagated_state_ = old_state;
            livo_propagated_cov_ = kf_.GetP();
        }

        kf_.Update(ESKF::ObsType::LIDAR, 1);

        state_point_ = kf_.GetX();

        state_point_.timestamp_ = measures_.lidar_end_time_;
        const double delta_velocity = (old_state.vel_ - state_point_.vel_).norm();
        if (voxel_map_)
        {
            voxel_map_->RefreshWorldAndVar(state_point_, kf_.GetP());
            if (visual_en_)
                pg_list_ = voxel_map_->pv_list();
        }
        const double current_speed = state_point_.vel_.norm();

        if (pub_)
        {
            // 全部外参调平：T' = V^{-1} * T_w_imu（含旋转+平移），用于 odom/TF/path/bag；
            // world 点云同样左乘 V^{-1}，与 TF 同系。
            pub_->SetPosestamp(GetExtrinsicRemappedState(state_point_));
            pub_->PublishOdom(state_point_.timestamp_);

            if (path_en_)
                pub_->PublishPath(lidar_end_time_);
            if (scan_pub_en_)
                PublishFrameWorld(pub_);
            if (scan_pub_en_ && scan_body_pub_en_)
                PublishFrameBody(pub_);
        }

        MapIncremental();
            
        if (enable_debug_log_)
        {
            LOG(INFO) << get_time_us() << "[DEBUG][LaserMapping::" << __FUNCTION__ << "] "
                      << scan_undistort_->points.size() << " down " << cur_pts
                      << (voxel_map_ ? " Voxel roots: " : " Map grid num: ")
                      << (voxel_map_ ? voxel_map_->NumRootVoxels()
                                     : (ivox_ ? ivox_->NumValidGrids() : 0))
                      << (voxel_map_ ? (" lru_evict=" + std::to_string(voxel_map_->LruEvicted())) : "")
                      << (vio_manager_ ? (" feat_map=" + std::to_string(vio_manager_->FeatMapSize()) +
                                          " feat_evict=" + std::to_string(vio_manager_->FeatLruEvicted()) +
                                          " feat_pt_evict=" + std::to_string(vio_manager_->FeatPointsEvicted()))
                                       : "")
                      << " effect num : " << effect_feat_surf_ << ", " << effect_feat_icp_;
            const auto &imu = state_point_;
            const Vec3d &t = imu.pos_;
            const Quatd q = imu.rot_.unit_quaternion();
            LOG(INFO) << get_time_us() << "[DEBUG][LaserMapping::" << __FUNCTION__ << "] " << std::fixed
                      << std::setprecision(4) << "LIO pose t: " << t.transpose()
                      << " | q(xyzw): " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                      << " | v: " << imu.vel_.transpose()
                      << " | t=" << imu.timestamp_;

            if (delta_velocity > 1.0 || current_speed > 5.0) {
                LOG(ERROR) << get_time_us() << "[ERROR][LaserMapping::" << __FUNCTION__ << "] "
                           << "detected very large vel change, last: " 
                           << ", cur:" << state_point_.vel_.transpose();
            }
        }

        /// keyframes
        if (last_kf_ == nullptr)
        {
            MakeKF();
        }
        else
        {
            SE3 last_pose = last_kf_->GetLIOPose();
            SE3 cur_pose = state_point_.GetPose();

            if ((last_pose.translation() - cur_pose.translation()).norm() > options_.kf_dis_th_ ||
                (last_pose.so3().inverse() * cur_pose.so3()).log().norm() > options_.kf_angle_th_)
            {
                MakeKF();
            }
            else if (!options_.is_in_slam_mode_ &&
                     (state_point_.timestamp_ - last_kf_->GetState().timestamp_) > 2.0)
            {
                MakeKF();
            }
        }

        /// LIO 后验拷回 kf_imu_，再补剩余 IMU。LIVO 只在 LIO 步做一次，VIO 不再拷。
        {
            std::lock_guard<std::mutex> lock(mtx_buffer_);

            // 回绕诊断：kf_imu_ 是 loc 红线(PGO DR)的唯一来源。
            // PGO 吃的是相邻两帧 DR 的相对位姿，所以这里关心的是
            // "回绕前的 kf_imu_" 与 "回绕重放后的 kf_imu_" 之间的位姿差(jump_*)。
            const NavState dr_before = kf_imu_.GetX();
            const NavState lio_post = kf_.GetX();
            const size_t buf_sz = imu_buffer_.size();
            const double buf_front_t = buf_sz > 0 ? imu_buffer_.front()->timestamp : -1.0;
            const double buf_back_t = buf_sz > 0 ? imu_buffer_.back()->timestamp : -1.0;

            kf_imu_ = kf_;
            int replay_cnt = 0;
            double replay_dt = 0.0;
            if (!imu_buffer_.empty())
            {
                double t = kf_imu_.GetX().timestamp_;
                const double replay_t0 = t;
                for (auto &imu : imu_buffer_)
                {
                    if (imu->timestamp <= t)
                        continue;
                    kf_imu_.Predict(imu->timestamp - t, p_imu_->Q_,
                                    imu->angular_velocity, imu->linear_acceleration);
                    t = imu->timestamp;
                    ++replay_cnt;
                }
                replay_dt = t - replay_t0;
            }

            const NavState dr_after = kf_imu_.GetX();
            const SE3 dr_jump = dr_before.GetPose().inverse() * dr_after.GetPose();
            const double jump_trans = dr_jump.translation().norm();
            const double jump_rot_deg = dr_jump.so3().log().norm() * 57.29577951308232;
            // 同一时间差下 DR 自身该走多远，用来判断 jump 是否远超真实运动
            const double dr_stamp_dt = dr_after.timestamp_ - dr_before.timestamp_;
            // LOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] "
            //           << "[REWIND] step=" << (visual_en_ ? (livo_step == LivoStep::LIO ? "LIVO_LIO" : "LIVO_OTHER") : "PURE_LIO")
            //           << std::fixed << std::setprecision(4)
            //           << " | kf_t0=" << lio_post.timestamp_
            //           << " state_point_t=" << state_point_.timestamp_
            //           << " lidar_end_t=" << measures_.lidar_end_time_
            //           << " dr_before_t=" << dr_before.timestamp_
            //           << " dr_after_t=" << dr_after.timestamp_
            //           << " dr_stamp_dt=" << dr_stamp_dt
            //           << " | replay_dt=" << replay_dt << " replay_n=" << replay_cnt
            //           << " buf_n=" << buf_sz << " buf_t=[" << buf_front_t << "," << buf_back_t << "]"
            //           << " | jump_trans=" << jump_trans << " jump_rot_deg=" << jump_rot_deg
            //           << " | dr_before_p=" << dr_before.pos_.transpose()
            //           << " dr_after_p=" << dr_after.pos_.transpose()
            //           << " lio_p=" << lio_post.pos_.transpose()
            //           << " | dr_before_v=" << dr_before.vel_.transpose()
            //           << " lio_v=" << lio_post.vel_.transpose()
            //           << " dr_after_v=" << dr_after.vel_.transpose();
        }
        auto t2 = std::chrono::steady_clock::now();
        auto vo_time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        if (enable_debug_log_)
            LOG(INFO) << get_time_us() << "[DEBUG][LidarLoc::" << __FUNCTION__ << "] " << "LIO use time: " << vo_time_used.count() << " s";

        if (visual_en_)
            livo_has_lio_cloud_ = true;

        return true;
    }

    void LaserMapping::ProjectKFs(CloudPtr cloud, int size_limit)
    {
        auto state = GetExtrinsicRemappedState(kf_.GetX());
        SE3 pose_cur(state.rot_, state.pos_);
        pose_cur = pose_cur.inverse();

        for (auto kf : proj_kfs_)
        {
            // LOG(INFO) << "projecting kf: " << kf->GetID();
            // if (last_kf_) {
            // auto kf = last_kf_;
            SE3 pose = pose_cur * kf->GetLIOPose();

            int cnt = 0;
            for (auto &pt : kf->GetCloud()->points)
            {
                Vec3d p = pose * ToVec3d(pt);
                PointType pcl_pt;

                pcl_pt.x = p.x();
                pcl_pt.y = p.y();
                pcl_pt.z = p.z();
                pcl_pt.intensity = pt.intensity;

                cloud->push_back(pcl_pt);
                cnt++;

                if (cnt > size_limit)
                {
                    break;
                }
            }
            // }
        }
    }

    void LaserMapping::MakeKF()
    {
        Keyframe::Ptr kf = std::make_shared<Keyframe>(kf_id_++, scan_undistort_, state_point_);

        if (last_kf_)
        {
            DLOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] " << "last kf lio: " << last_kf_->GetLIOPose().translation().transpose()
                       << ", opt: " << last_kf_->GetOptPose().translation().transpose();

            /// opt pose 用之前的递推
            SE3 delta = last_kf_->GetLIOPose().inverse() * kf->GetLIOPose();
            kf->SetOptPose(last_kf_->GetOptPose() * delta);
        }
        else
            kf->SetOptPose(kf->GetLIOPose());

        kf->SetState(state_point_);

        // LOG(INFO) << get_time_us() <<  "[INFO][lasermapping makekf: create kf " << kf->GetID() << ", state: " << state_point_.pos_.transpose()
        //    << ", kf opt pose: " << kf->GetOptPose().translation().transpose()
        //    << ", lio pose: " << kf->GetLIOPose().translation().transpose() << ", time: " << std::setprecision(14)
        //    << state_point_.timestamp_;

        if (options_.is_in_slam_mode_)
            all_keyframes_.emplace_back(kf);

        last_kf_ = kf;

        // MapIncremental();

        /// 更新project kfs
        if (proj_kfs_.size() >= options_.max_proj_kfs_)
        {
            auto last = proj_kfs_.back();

            SE3 delta = last->GetLIOPose().inverse() * kf->GetLIOPose();

            if (delta.translation().norm() < 2 || delta.so3().log().norm() < 20 / 180 * M_PI)
            {
                // proj_kfs_.pop_back();
            }
            else
            {
                proj_kfs_.pop_front();
                proj_kfs_.emplace_back(kf);
            }
        }
        else
        {
            proj_kfs_.emplace_back(kf);
        }

        // for (auto &kf : proj_kfs_) {
        //     LOG(INFO) << "proj kf: " << kf->GetID();
        // }
    }

    // IMU异常数据检测
    bool LaserMapping::IMUCheck(const lightning::IMUPtr &msg_in)
    {
        double acc_norm = msg_in->linear_acceleration.norm();
        double gyr_norm = msg_in->angular_velocity.norm();

        cur_timestamp_imu_ = msg_in->timestamp;
        lightning::IMUPtr cur_msg;

        if (cur_timestamp_imu_ < last_timestamp_imu_)
        {
            LOG(WARNING) << get_time_us() << "[WARN][LaserMapping::" << __FUNCTION__ << "] " << "imu loop back, clear buffer";
            imu_buffer_.clear();
        }

        return true;
    }

    void LaserMapping::MapIncremental()
    {
        if (voxel_map_)
        {
            voxel_map_->UpdateVoxelMap(voxel_map_->pv_list());
            return;
        }

        PointVector points_to_add;
        PointVector point_no_need_downsample;

        size_t cur_pts = scan_down_body_->size();
        points_to_add.reserve(cur_pts);
        point_no_need_downsample.reserve(cur_pts);

        std::vector<size_t> index(cur_pts);
        for (size_t i = 0; i < cur_pts; ++i)
            index[i] = i;

        std::for_each(index.begin(), index.end(), [&](const size_t &i)
                      {
        /* transform to world frame */
        PointBodyToWorld(scan_down_body_->points[i], scan_down_world_->points[i]);

        /* decide if need add to map */
        PointType &point_world = scan_down_world_->points[i];
        if (!nearest_points_[i].empty() && flg_EKF_inited_) {
            const PointVector &points_near = nearest_points_[i];

            Eigen::Vector3f center = ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_;
            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

            if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) {
                point_no_need_downsample.emplace_back(point_world);
                return;
            }

            bool need_add = true;
            float dist = math::calc_dist(point_world.getVector3fMap(), center);
            if (points_near.size() >= fasterlio::NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < fasterlio::NUM_MATCH_POINTS; readd_i++) {
                    if (math::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                        need_add = false;
                        break;
                    }
                }
            }

            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            points_to_add.emplace_back(point_world);
        } });

        ivox_->AddPoints(points_to_add);
        ivox_->AddPoints(point_no_need_downsample);
    }

    void LaserMapping::AccAsGravityObsModel(NavState &s, ESKF::CustomObservationModel &obs)
    {
        if (!last_imu_)
        {
            obs.valid_ = false;
            return;
        }
        // Static-specific-force consistency: a_meas ~= R^T(-g) + ba.
        const Vec3d pred_acc = s.rot_.inverse() * (-s.grav_.vec_) + s.ba_;
        Vec3d residual = last_imu_->linear_acceleration - pred_acc;
        // H = I3 at ba index (18), H^T H = I3 at (18,18), H^T r = residual at (18)
        obs.HTH_.setZero();
        obs.HTr_.setZero();
        obs.HTH_.block<3, 3>(18, 18) = Mat3d::Identity() / stationary_acc_noise_;
        obs.HTr_.segment<3>(18) = residual / stationary_acc_noise_;
        obs.lidar_residual_mean_ = residual.squaredNorm();
        obs.lidar_residual_max_ = residual.cwiseAbs().maxCoeff();
    }

    void LaserMapping::BiasObsModel(NavState &s, ESKF::CustomObservationModel &obs)
    {
        // Weak bias prior in long stationary windows.
        // H = I3 at bg(15) + I3 at ba(18), H^T H = I3 at (15,15) + I3 at (18,18)
        obs.HTH_.setZero();
        obs.HTr_.setZero();
        obs.HTH_.block<3, 3>(15, 15) = Mat3d::Identity() / stationary_bias_noise_;
        obs.HTH_.block<3, 3>(18, 18) = Mat3d::Identity() / stationary_bias_noise_;
        obs.HTr_.segment<3>(15) = -s.bg_ / stationary_bias_noise_;
        obs.HTr_.segment<3>(18) = -s.ba_ / stationary_bias_noise_;
        double res_sq = s.bg_.squaredNorm() + s.ba_.squaredNorm();
        obs.lidar_residual_mean_ = res_sq;
        obs.lidar_residual_max_ = std::max(s.bg_.cwiseAbs().maxCoeff(), s.ba_.cwiseAbs().maxCoeff());
    }

    void LaserMapping::StationaryBiasAndAccObsModel(NavState &s, ESKF::CustomObservationModel &obs)
    {
        if (!last_imu_)
        {
            obs.valid_ = false;
            return;
        }
        obs.HTH_.setZero();
        obs.HTr_.setZero();

        // Part 1: ACC_AS_GRAVITY — 加计量测约束 ba (index 18)
        const Vec3d pred_acc = s.rot_.inverse() * (-s.grav_.vec_) + s.ba_;
        Vec3d acc_residual = last_imu_->linear_acceleration - pred_acc;
        obs.HTH_.block<3, 3>(18, 18) += Mat3d::Identity() / stationary_acc_noise_;
        obs.HTr_.segment<3>(18) += acc_residual / stationary_acc_noise_;

        // Part 2 不对：随机游走不知道 bias 真值，不能造 bg/ba ≈ 0 的观测。
        // // Part 2: BIAS — 偏置弱先验 bg (index 15) + ba (index 18)
        // obs.HTH_.block<3, 3>(15, 15) += Mat3d::Identity() / stationary_bias_noise_;
        // obs.HTH_.block<3, 3>(18, 18) += Mat3d::Identity() / stationary_bias_noise_;
        // obs.HTr_.segment<3>(15) += -s.bg_ / stationary_bias_noise_;
        // obs.HTr_.segment<3>(18) += -s.ba_ / stationary_bias_noise_;

        obs.lidar_residual_mean_ = acc_residual.squaredNorm();
        obs.lidar_residual_max_ = acc_residual.cwiseAbs().maxCoeff();
        // double res_sq = acc_residual.squaredNorm() + s.bg_.squaredNorm() + s.ba_.squaredNorm();
        // obs.lidar_residual_mean_ = res_sq;
        // obs.lidar_residual_max_ = std::max({acc_residual.cwiseAbs().maxCoeff(),
        //                                      s.bg_.cwiseAbs().maxCoeff(),
        //                                      s.ba_.cwiseAbs().maxCoeff()});
    }

    void LaserMapping::ObsModelVoxelMap(NavState &s, ESKF::CustomObservationModel &obs)
    {
        voxel_map_->BuildPvList(scan_down_body_, s, kf_.GetP());
        voxel_map_->BuildResidualListOMP();

        const auto &pv_list = voxel_map_->pv_list();
        const auto &ptpl_list = voxel_map_->ptpl_list();
        const size_t n_body = scan_down_body_->size();
        scan_down_world_->resize(n_body);
        for (size_t i = 0; i < n_body && i < pv_list.size(); ++i)
        {
            scan_down_world_->points[i].getVector3fMap() = pv_list[i].point_w.cast<float>();
            scan_down_world_->points[i].intensity = scan_down_body_->points[i].intensity;
        }

        effect_feat_surf_ = static_cast<int>(ptpl_list.size());
        effect_feat_icp_ = 0;

        const Mat3d off_R = s.offset_R_lidar_.matrix();
        const Vec3d off_t = s.offset_t_lidar_;
        const Mat3d R_wi = s.rot_.matrix();
        const Mat3d Rt = R_wi.transpose();

        obs.HTH_.setZero();
        obs.HTr_.setZero();

        std::vector<Mat6d> JTJ(static_cast<size_t>(effect_feat_surf_));
        std::vector<Vec6d> JTr(static_cast<size_t>(effect_feat_surf_));
        std::vector<double> res_sq(static_cast<size_t>(effect_feat_surf_), 0.0);

        omp_set_num_threads(omp_num_);
#pragma omp parallel for
        for (int i = 0; i < effect_feat_surf_; ++i)
        {
            const PointToPlane &ptpl = ptpl_list[static_cast<size_t>(i)];
            const Vec3d point_this = off_R * ptpl.point_b + off_t;
            const Mat3d point_crossmat = math::SKEW_SYM_MATRIX(point_this);
            const Vec3d n = ptpl.normal;
            const Vec3d A = point_crossmat * (Rt * n);

            const Mat3d R_prop = livo_propagated_state_.rot_.matrix();
            const Vec3d point_world = R_prop * point_this + livo_propagated_state_.pos_;
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = (point_world - ptpl.center).transpose();
            J_nq.block<1, 3>(0, 3) = -n.transpose();
            const double sigma_l = (J_nq * ptpl.plane_var * J_nq.transpose())(0, 0);
            const Mat3d var = (R_prop * off_R) * ptpl.body_cov * (R_prop * off_R).transpose();
            const double R_inv = 1.0 / (0.001 + sigma_l + n.dot(var * n));

            Eigen::Matrix<double, 1, ESKF::pose_obs_dim_> J;
            J << n[0], n[1], n[2], A[0], A[1], A[2];
            const double res = -static_cast<double>(ptpl.dis_to_plane);
            JTJ[static_cast<size_t>(i)] = (J.transpose() * J) * R_inv;
            JTr[static_cast<size_t>(i)] = J.transpose() * (res * R_inv);
            res_sq[static_cast<size_t>(i)] = res * res;
        }

        for (int i = 0; i < effect_feat_surf_; ++i)
        {
            obs.HTH_.block<ESKF::pose_obs_dim_, ESKF::pose_obs_dim_>(0, 0) += JTJ[static_cast<size_t>(i)];
            obs.HTr_.head<ESKF::pose_obs_dim_>() += JTr[static_cast<size_t>(i)];
        }

        if (!res_sq.empty())
        {
            std::sort(res_sq.begin(), res_sq.end());
            obs.lidar_residual_mean_ = res_sq[res_sq.size() / 2];
            obs.lidar_residual_max_ = res_sq.back();
        }
    }

    void LaserMapping::ObsModel(NavState &s, ESKF::CustomObservationModel &obs)
    {
        if (voxel_map_)
        {
            ObsModelVoxelMap(s, obs);
            return;
        }

        //  auto t1 = std::chrono::steady_clock::now();
        int cnt_pts = scan_down_body_->size();

        Mat3f R_wl = (s.rot_ * s.offset_R_lidar_).matrix().cast<float>();
        Vec3f t_wl = (s.rot_ * s.offset_t_lidar_ + s.pos_).cast<float>();

        omp_set_num_threads(omp_num_);
#pragma omp parallel for
        for (size_t i = 0; i < cnt_pts; i++)
        {
            PointType &point_body = scan_down_body_->points[i];
            PointType &point_world = scan_down_world_->points[i];

            /* transform to world frame */
            Vec3f p_body = point_body.getVector3fMap();
            point_world.getVector3fMap() = R_wl * p_body + t_wl;
            point_world.intensity = point_body.intensity;

            auto &points_near = nearest_points_[i];
            points_near.clear();

            /** Find the closest surfaces in the map **/
            // if (obs.converge_) {
            ivox_->GetClosestPoint(point_world, points_near, fasterlio::NUM_MATCH_POINTS);
            point_selected_surf_[i] = points_near.size() >= fasterlio::MIN_NUM_MATCH_POINTS;
            point_selected_icp_[i] = point_selected_surf_[i];
            if (point_selected_surf_[i])
                point_selected_surf_[i] = math::esti_plane(plane_coef_[i], points_near, fasterlio::ESTI_PLANE_THRESHOLD);

            point_selected_icp_[i] = point_selected_icp_[i] && !point_selected_surf_[i];

            /// 计算平面阈值
            if (point_selected_surf_[i])
            {
                auto temp = point_world.getVector4fMap();
                temp[3] = 1.0;
                float pd2 = plane_coef_[i].dot(temp);

                bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                if (valid_corr)
                {
                    point_selected_surf_[i] = true;
                    residuals_[i] = pd2;
                }
                else
                    point_selected_surf_[i] = false;
            }
        }

        effect_feat_surf_ = 0;
        effect_feat_icp_ = 0;

        corr_pts_.resize(cnt_pts);
        corr_norm_.resize(cnt_pts);
        for (int i = 0; i < cnt_pts; i++)
        {
            if (point_selected_surf_[i])
            {
                corr_norm_[effect_feat_surf_] = plane_coef_[i];
                corr_pts_[effect_feat_surf_] = scan_down_body_->points[i].getVector4fMap();
                corr_pts_[effect_feat_surf_][3] = residuals_[i];

                effect_feat_surf_++;
            }

            if (point_selected_icp_[i])
            {
                effect_feat_icp_++;
            }
        }
        corr_pts_.resize(effect_feat_surf_);
        corr_norm_.resize(effect_feat_surf_);

        if (effect_feat_surf_ < 20)
        {
            obs.valid_ = false;
            LOG(WARNING) << get_time_us() << "[WARN][LaserMapping::" << __FUNCTION__ << "] "
                         << "No enough effective surface points: " << effect_feat_surf_ << ", icp: " << effect_feat_icp_
                         << ", required: " << 20;
            return;
        }

        // obs.h_x_ = Eigen::MatrixXd::Zero(effect_feat_surf_, 6); // 23
        // obs.residual_.resize(effect_feat_surf_);

        const Mat3f off_R = s.offset_R_lidar_.matrix().cast<float>();
        const Vec3f off_t = s.offset_t_lidar_.cast<float>();
        const Mat3f Rt = s.rot_.matrix().transpose().cast<float>();

        // /// 点面ICP部分
        obs.HTH_.setZero();
        obs.HTr_.setZero();

        std::vector<Mat6d> JTJ(effect_feat_surf_);
        std::vector<Vec6d> JTr(effect_feat_surf_);

        std::vector<double> res_sq(cnt_pts);

#pragma omp parallel for
        for (size_t i = 0; i < effect_feat_surf_; i++)
        {
            Vec3f point_this_be = corr_pts_[i].head<3>();
            Vec3f point_this = off_R * point_this_be + off_t;
            Mat3f point_crossmat = math::SKEW_SYM_MATRIX(point_this);

            /*** get the normal vector of closest surface/corner ***/
            Vec3f norm_vec = corr_norm_[i].head<3>();

            /*** calculate the Measurement Jacobian matrix H ***/
            Vec3f C(Rt * norm_vec);
            Vec3f A(point_crossmat * C);

            // obs.h_x_.block<1, ESKF::pose_obs_dim_>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2];

            Eigen::Matrix<double, 1, ESKF::pose_obs_dim_> J;
            J.setZero();

            J << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2];

            // double w = huber_weight(res);
            double w = 1.0;

            float res = -corr_pts_[i][3];

            JTJ[i] = (J.transpose() * J).eval() * w;
            JTr[i] = J.transpose() * res * w;

            res_sq[i] = res * res;
            // obs.residual_(i) = res;
        }

        for (int i = 0; i < effect_feat_surf_; ++i)
        {
            obs.HTH_.block<ESKF::pose_obs_dim_, ESKF::pose_obs_dim_>(0, 0) += JTJ[i] * options_.plane_icp_weight_;
            obs.HTr_.head<ESKF::pose_obs_dim_>() += JTr[i] * options_.plane_icp_weight_;
            // obs.HTH_ += JTJ[i] * options_.plane_icp_weight_;
            // obs.HTr_ += JTr[i] * options_.plane_icp_weight_;
        }

        if (!res_sq.empty())
        {
            std::sort(res_sq.begin(), res_sq.end());
            obs.lidar_residual_mean_ = res_sq[res_sq.size() / 2];
            obs.lidar_residual_max_ = res_sq[res_sq.size() - 1];
        }

        // 点到点 icp
        if (options_.enable_icp_part_)
        {
            JTJ.resize(cnt_pts);
            JTr.resize(cnt_pts);

#pragma omp parallel for
            for (size_t i = 0; i < cnt_pts; i++)
            {
                if (point_selected_icp_[i] == false)
                {
                    continue;
                }

                Vec3d q_body = scan_down_body_->points[i].getVector3fMap().cast<double>();
                Vec3d q_lidar = s.offset_R_lidar_.matrix() * q_body + s.offset_t_lidar_;
                Vec3d q_world = scan_down_world_->points[i].getVector3fMap().cast<double>();
                Vec3d q_target = nearest_points_[i][0].getVector3fMap().cast<double>();
                Vec3d e = q_world - q_target;

                if (e.norm() > 0.5)
                {
                    point_selected_icp_[i] = false;
                    continue;
                }

                Eigen::Matrix<double, 3, 6> J;
                J.setZero();

                /// translation 部分
                J.block<3, 3>(0, 0) = Mat3d::Identity();

                /// rotation 部分
                J.block<3, 3>(0, 3) = -s.rot_.matrix() * SO3::hat(q_lidar);

                JTJ[i] = J.transpose() * J;
                JTr[i] = -J.transpose() * e;
            }

            for (int i = 0; i < cnt_pts; ++i)
            {
                if (point_selected_icp_[i] == false)
                {
                    continue;
                }
                // obs.HTH_ += JTJ[i] * options_.icp_weight_;
                // obs.HTr_ += JTr[i] * options_.icp_weight_;
                obs.HTH_.block<ESKF::pose_obs_dim_, ESKF::pose_obs_dim_>(0, 0) += JTJ[i] * options_.icp_weight_;
                obs.HTr_.head<ESKF::pose_obs_dim_>() += JTr[i] * options_.icp_weight_;
            }
        }

        // auto t2 = std::chrono::steady_clock::now();
        // auto vo_time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        //     LOG(INFO) << get_time_us() << "[INFO][LidarLoc::" << __FUNCTION__ << "] " << "LIO use time: " << vo_time_used.count() << " s";

    }

    void LaserMapping::HuberWeight(float &res)
    {
        // float rho, drho;

        // const float delta = 2.0;
        // const float dsqr = delta * delta;
        // const float dsqr_inv = 1.0 / dsqr;

        // if (res >= 0)
        // {
        //     rho = dsqr * std::log(1 + res * dsqr_inv);
        //     drho = 1.0 / (1 + res * dsqr_inv);
        // }
        // else
        // {
        //     rho = -dsqr * std::log(1 - res * dsqr_inv);
        //     drho = 1.0 / (1 - res * dsqr_inv);
        // }

        // obs.residual_(i) = rho;
        // obs.h_x_.block<1, 12>(i, 0) = obs.h_x_.block<1, 12>(i, 0).eval() * drho;
    }

    CloudPtr LaserMapping::GetGlobalMap(bool use_lio_pose, bool use_voxel, float res)
    {
        CloudPtr global_map(new PointCloudType);

        pcl::VoxelGrid<PointType> voxel;
        voxel.setLeafSize(res, res, res);

        CloudPtr sub_map(new PointCloudType);
        int count = 0;
        for (auto &kf : all_keyframes_)
        {
            CloudPtr cloud = kf->GetCloud();
            CloudPtr cloud_trans(new PointCloudType);

            if (use_lio_pose)
            {
                pcl::transformPointCloud(*cloud, *cloud_trans, kf->GetLIOPose().matrix());
            }
            else
            {
                pcl::transformPointCloud(*cloud, *cloud_trans, kf->GetOptPose().matrix());
            }

            *sub_map += *cloud_trans;
            count++;

            if (count >= 100)
            {
                voxel.setInputCloud(sub_map);
                voxel.filter(*sub_map); // 局部降采样

                *global_map += *sub_map;
                sub_map->clear(); // 释放内存
                count = 0;
            }

            LOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] "
                      << "kf " << kf->GetID() << ", pose: " << kf->GetOptPose().translation().transpose();
        }

        *global_map += *sub_map;

        // for (auto &kf : all_keyframes_)
        // {
        //     CloudPtr cloud = kf->GetCloud();
        //     CloudPtr cloud_filter(new PointCloudType);

        //     if (use_voxel)
        //     {
        //         voxel.setInputCloud(cloud);
        //         voxel.filter(*cloud_filter);
        //     }
        //     else
        //     {
        //         cloud_filter = cloud;
        //     }

        //     CloudPtr cloud_trans(new PointCloudType);

        //     if (use_lio_pose)
        //     {
        //         pcl::transformPointCloud(*cloud_filter, *cloud_trans, kf->GetLIOPose().matrix());
        //     }
        //     else
        //     {
        //         pcl::transformPointCloud(*cloud_filter, *cloud_trans, kf->GetOptPose().matrix());
        //     }

        //     *global_map += *cloud_trans;

        //     LOG(INFO) << "kf " << kf->GetID() << ", pose: " << kf->GetOptPose().translation().transpose();
        // }

        // CloudPtr global_map_filtered(new PointCloudType);
        // if (use_voxel)
        // {
        //     voxel.setInputCloud(global_map);
        //     voxel.filter(*global_map_filtered);
        // }
        // else
        // {
        //     global_map_filtered = global_map;
        // }

        global_map->is_dense = false;
        global_map->height = 1;
        global_map->width = global_map->size();

        LOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] " << "global map: " << global_map->size();
        return global_map;
    }

    void LaserMapping::SaveMap()
    {
        if (dense_pcd_save_en_ && pub_)
        {
            if (visual_en_ && pcl_wait_save_rgb_ && !pcl_wait_save_rgb_->empty())
                pub_->SaveRgbMap(pcl_wait_save_rgb_);
            else if (pcl_wait_save_->size() > 0)
                pub_->SaveMap(pcl_wait_save_, PubROSTopic::DENSE);
        }

        SaveDownMap();
    }

    void LaserMapping::SaveDownMap()
    {
        auto global_map = GetGlobalMap(true);
        pub_->SaveMap(global_map, PubROSTopic::DOWNSAMPLED);
    }

    // 发布当前帧 world 系点云：与 odom/TF 相同，左乘完整外参 V^{-1} 调平。
    void LaserMapping::PublishFrameWorld(const std::shared_ptr<PubROSTopic> &pub)
    {
        CloudPtr laserCloudFullRes(dense_pub_en_ ? scan_undistort_ : scan_down_body_);
        int size = laserCloudFullRes->points.size();
        CloudPtr laserCloudWorld(new PointCloudType(size, 1));
        const Eigen::Isometry3d T_level = T_body_lio_.inverse();

        for (int i = 0; i < size; i++)
        {
            PointBodyToWorld(laserCloudFullRes->points[i], laserCloudWorld->points[i]);
            const Vec3d p = T_level * Vec3d(laserCloudWorld->points[i].x, laserCloudWorld->points[i].y,
                                            laserCloudWorld->points[i].z);
            laserCloudWorld->points[i].x = p.x();
            laserCloudWorld->points[i].y = p.y();
            laserCloudWorld->points[i].z = p.z();
        }

        if (dense_pcd_save_en_ && !visual_en_)
            *pcl_wait_save_ += *laserCloudWorld;

        pub->PublishFrameWorld(laserCloudWorld, lidar_end_time_);

        DLOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] " << "Published world frame with " << std::to_string(size) << " points";

        // yxliang 2026.2.4
        // 删除分段保存地图逻辑，地图过大需要分段保存时添加
    }

    void LaserMapping::PublishLivoVisualizations(const std::shared_ptr<PubROSTopic> &pub)
    {
        const bool save_rgb_dense = visual_en_ && dense_pcd_save_en_;
        if (!pub || !vio_manager_ || (!rgb_img_pub_en_ && !rgb_cloud_pub_en_ && !save_rgb_dense))
            return;

        if (rgb_img_pub_en_ && !vio_manager_->img_cp.empty())
        {
            if (dynamic_filter_ && dynamic_filter_->Enabled())
                dynamic_filter_->OverlayOnScaledImage(vio_manager_->img_cp);
            pub->PublishRgbImage(vio_manager_->img_cp, synced_image_time_);
        }
        if (rgb_cloud_pub_en_ || save_rgb_dense)
            PublishFrameWorldRgb(pub);
    }

    void LaserMapping::PublishFrameWorldRgb(const std::shared_ptr<PubROSTopic> &pub)
    {
        if (!vio_manager_ || vio_manager_->img_rgb.empty() || vio_manager_->new_frame_ == nullptr)
            return;

        CloudPtr laserCloudFullRes(dense_pub_en_ ? scan_undistort_ : scan_down_body_);
        if (!laserCloudFullRes || laserCloudFullRes->empty())
            return;

        const Eigen::Isometry3d T_level = T_body_lio_.inverse();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        rgb_cloud->reserve(laserCloudFullRes->size());
        for (const auto &pt : laserCloudFullRes->points)
        {
            PointType world_pt;
            PointBodyToWorld(pt, world_pt);
            const Vec3d p_level = T_level * Vec3d(world_pt.x, world_pt.y, world_pt.z);
            const Vec3d p_w(p_level.x(), p_level.y(), p_level.z());
            const Vec3d pf = vio_manager_->new_frame_->w2f(p_w);
            if (pf[2] < 0)
                continue;
            const Vec2d pc = vio_manager_->new_frame_->w2c(p_w);
            if (!vio_manager_->new_frame_->cam_->isInFrame(pc.cast<int>(), 3))
                continue;
            if (pf.norm() <= blind_rgb_points_)
                continue;
            const V3F pixel = vio_manager_->getInterpolatedPixel(vio_manager_->img_rgb, pc);
            pcl::PointXYZRGB rgb_pt;
            rgb_pt.x = static_cast<float>(p_w.x());
            rgb_pt.y = static_cast<float>(p_w.y());
            rgb_pt.z = static_cast<float>(p_w.z());
            rgb_pt.r = static_cast<uint8_t>(pixel[2]);
            rgb_pt.g = static_cast<uint8_t>(pixel[1]);
            rgb_pt.b = static_cast<uint8_t>(pixel[0]);
            rgb_cloud->push_back(rgb_pt);
        }
        if (dense_pcd_save_en_ && visual_en_ && !rgb_cloud->empty())
            *pcl_wait_save_rgb_ += *rgb_cloud;
        if (rgb_cloud_pub_en_ && pub && !rgb_cloud->empty())
            pub->PublishRgbFrameWorld(rgb_cloud, state_point_.timestamp_);
    }

    // 发布当前帧 body 系下的全量点云
    void LaserMapping::PublishFrameBody(const std::shared_ptr<PubROSTopic> &pub)
    {
        int size = scan_undistort_->points.size();
        CloudPtr laserCloudIMUBody(new PointCloudType(size, 1));

        for (int i = 0; i < size; i++)
            PointBodyLidarToIMU(scan_undistort_->points[i], laserCloudIMUBody->points[i]);

        pub->PublishFrameBody(laserCloudIMUBody, lidar_end_time_);

        DLOG(INFO) << get_time_us() << "[INFO][LaserMapping::" << __FUNCTION__ << "] " << "Published body frame with " << std::to_string(size) << " points";
    }

    void LaserMapping::SortVec(const Eigen::VectorXf &vec, Eigen::VectorXf &sorted_vec, Eigen::VectorXi &ind)
    {
        ind = Eigen::VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1);
        auto rule = [vec](int i, int j) -> bool
        {
            return abs(vec(i)) > abs(vec(j));
        };

        std::sort(ind.data(), ind.data() + ind.size(), rule);
        sorted_vec.resize(vec.size());

        for (int i = 0; i < vec.size(); i++)
        {
            sorted_vec(i) = vec(ind(i));
        }
    }

    NavState LaserMapping::GetExtrinsicRemappedState(const NavState &state) const
    {
        NavState newstate = state;
        Eigen::Isometry3d T_w_imu = Eigen::Isometry3d::Identity();
        T_w_imu.matrix() = state.GetPose().matrix();
        // 全部外参左乘调平：T' = V^{-1} * T_w_imu（旋转+平移）
        const Eigen::Isometry3d T_level = T_body_lio_.inverse() * T_w_imu;
        newstate.pos_ = T_level.translation();
        Eigen::Quaterniond q_level(T_level.linear());
        q_level.normalize();
        newstate.rot_ = SO3(q_level);

        return newstate;
    }

    CloudPtr LaserMapping::GetProjCloud()
    {
        auto cloud = scan_undistort_;
        ProjectKFs(cloud);
        return cloud;
    }

    void LaserMapping::Reset(SE3 &pose)
    {
        LOG(WARNING) << get_time_us() << "[WARN][LaserMapping::" << __FUNCTION__ << "] " 
                        << "Lio System Resetting to Target Pose...";
        last_reset_wall_time_ = rclcpp::Clock().now().seconds();

        NavState init_state = kf_.GetX();

        init_state.SetPose(pose);
        init_state.SetVel(Vec3d::Zero());

        kf_.ChangeX(init_state);
        kf_imu_ = kf_;
        if (imu_reset_level_ == 2)
            p_imu_->reset_level_ = ImuResetLevel::DYNAMIC_RESET;
        else if (imu_reset_level_ == 1)
            p_imu_->reset_level_ = ImuResetLevel::STATIC_RESET;

        p_imu_->Reset();

        {
            std::lock_guard<std::mutex> lock_msg(mtx_buffer_);
            imu_buffer_.clear();
            lidar_buffer_.clear();
            time_buffer_.clear();
            for (auto &buffer : image_buffers_)
                buffer.clear();
            synced_images_.clear();
            pg_list_.clear();
            livo_pcl_proc_next_->clear();
        }

        if (ivox_ != nullptr)
        {
            ivox_ = std::make_shared<IVoxType>(ivox_options_);
        }
        if (voxel_map_ != nullptr)
        {
            voxel_map_->Clear();
        }
        if (vio_manager_ != nullptr)
            vio_manager_->resetFeatMap();

        first_lidar_time_ = 0;
        lidar_pushed_ = false;
        flg_first_scan_ = true; // 重新触发首帧逻辑
        IMU_inited_ = false;
        flg_EKF_inited_ = false;
        last_timestamp_lidar_ = -1.0;
        last_timestamp_img_.assign(image_buffers_.size(), -1.0);
        last_timestamp_imu_ = -1.0;
        cur_timestamp_imu_ = -1.0;
        last_lidar_time_ = 0;
        lidar_end_time_ = 0;
        synced_image_time_ = 0.0;
        last_livo_update_time_ = -1.0;
        last_livo_step_ = LivoStep::NONE;
        last_run_was_vio_ = false;
        livo_has_lio_cloud_ = false;

        last_kf_ = nullptr;
    }
}
