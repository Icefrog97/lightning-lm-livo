#ifndef FASTER_LIO_LASER_MAPPING_H
#define FASTER_LIO_LASER_MAPPING_H

#include <pcl/filters/voxel_grid.h>
#include <opencv2/core.hpp>
#include <algorithm>
// #include <condition_variable>
// #include <thread>

#include "common/options.h"
#include "lio/eskf.hpp"
#include "lio/IMU_Processing.hpp"
#include "lio/preprocess.h"
#include "common/imu.h"
#include "common/keyframe.h"
#include "ivox3d/ivox3d.h"
#include "lio/voxel_map.h"
#include "vio/camera_slot.h"
#include "vio/vio.h"
#include "perception/dynamic_filter.h"

namespace lightning
{
    class OnlineLoop;
    class PubROSTopic;
    /**
     * laser mapping
     * 目前有个问题：点云在缓存之后，实际处理的并不是最新的那个点云（通常是buffer里的前一个），这是因为bag里的点云用的开始时间戳，导致
     * 点云的结束时间要比IMU多0.1s左右。为了同步最近的IMU，就只能处理缓冲队列里的那个点云，而不是最新的点云
     */
    class LaserMapping
    {
    public:
        struct Options
        {
            Options() {}

            bool is_in_slam_mode_ = true;   // 是否在slam模式下
            bool enable_icp_part_ = true;   // 是否添加ICP部分
            double plane_icp_weight_ = 1.0; // 点面ICP部分的权重
            double icp_weight_ = 100;       // ICP部分的权重

            int min_pts = 300; // 配准所需的点数
            /// 关键帧阈值
            double kf_dis_th_ = 1.0;
            double kf_angle_th_ = 10 * M_PI / 180.0;
            bool proj_kfs_ = false;
            int max_proj_kfs_ = 5;

            Eigen::Quaterniond Q_body_wrt_lio_;
            Eigen::Vector3d T_body_wrt_lio_;
        };

        /// 由 ROS 参数加载器填好，Init 不再读 nh.param。
        struct Config
        {
            bool enable_debug_log = false;
            std::string map_origin_path;

            bool path_en = true;
            bool scan_pub_en = true;
            bool dense_pub_en = true;
            bool scan_body_pub_en = true;
            bool rgb_img_pub_en = false;
            bool rgb_cloud_pub_en = false;
            double blind_rgb_points = 0.0;

            bool visual_en = false;
            bool use_voxel_map = false;
            int lio_max_iterations = 5;
            bool dynamic_filter_enable = false;
            perception::DynamicFilter::Options dynamic_filter;
            int camera_num = 1;
            double image_time_offset = 0.0;
            double exposure_time_init = 0.0;
            bool vio_raycast_en = false;
            bool vio_normal_en = true;
            bool vio_exposure_estimate_en = true;
            bool vio_inverse_composition_en = false;
            double inv_expo_cov = 0.1;
            int vio_grid_size = 5;
            int vio_grid_n_height = 17;
            int vio_patch_size = 8;
            int vio_patch_pyrimid_level = 4;
            int vio_max_iterations = 5;
            double vio_outlier_threshold = 1000.0;
            double vio_img_point_cov = 100.0;
            bool vio_ncc_en = false;
            double vio_ncc_threshold = 0.85;
            int visual_lru_capacity = 100000;
            int visual_max_points_per_voxel = 10;

            int max_iteration = 4;
            double degeneracy_threshold_ratio = 0.2;
            double esti_plane_threshold = 0.001;
            double filter_size_surf = 0.5;
            double filter_size_map = 0.5;
            int omp_num = 4;
            double gyr_cov = 0.1;
            double acc_cov = 0.1;
            double b_gyr_cov = 0.0001;
            double b_acc_cov = 0.0001;
            std::vector<double> extrin_T;
            std::vector<double> extrin_R;
            bool use_aa = false;
            double kf_dis_th = 1.0;
            double kf_angle_th = 10.0 * M_PI / 180.0;
            std::vector<CameraSlot> camera_slots;
            std::vector<double> camera_Rcl;
            std::vector<double> camera_Pcl;

            double blind = 0.01;
            int lidar_type = 4;
            int point_filter_num = 1;
            int scan_line = 1;
            double height_max = 0.01;
            double height_min = 0.01;
            bool dense_pcd_save_en = false;

            double ivox_grid_resolution = 2.0;
            double plane_icp_weight = 2.0;
            int ivox_nearby_type = 0;
            int ivox_capacity = 1000000;
            bool enable_icp_part = true;
            bool proj_kfs = true;

            int max_layer = 2;
            double voxel_size = 0.5;
            double min_eigen_value = 0.0025;
            double sigma_num = 3.0;
            double beam_err = 0.05;
            double dept_err = 0.02;
            int max_points_num = 50;
            std::vector<int> layer_init_num{5, 5, 5, 5, 5};
            int voxel_lru_capacity = 100000;
            bool use_imu_filter = true;
            int imu_reset_level = 2;
            int min_pts = 2;
            bool use_degeneracy = true;
            int max_init_count = 10;
            Eigen::Isometry3d T_body_lio = Eigen::Isometry3d::Identity();
        };

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;

        LaserMapping(Options options = Options());
        ~LaserMapping();

        /// nh 只用于发布和 vikit 读已经写好的相机参数。
        bool Init(const Config &cfg, const rclcpp::Node::SharedPtr &nh);

        bool Run();

        void ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);

        void ProcessPointCloud2(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg);

        void ProcessPointCloud2(CloudPtr cloud);

        void ProcessImage(int cam_id, const cv::Mat &image, double timestamp);

        void ProcessIMU(const lightning::IMUPtr &msg_in);

        bool VisualEnabled() const { return visual_en_; }
        /// True after a successful VIO Run() (empty VIO still returns false).
        bool LastRunWasVio() const { return last_run_was_vio_; }
        /// FAST VoxelMap LIO (with or without VIO). Independent of visual_en_.
        bool UseVoxelMap() const { return use_voxel_map_; }
        /// Subscribe/buffer images for VIO and/or dynamic filter.
        bool NeedImageInput() const { return visual_en_ || buffer_images_for_dyn_; }
        int CameraCount() const { return std::max(1, camera_num_); }

        bool IMUCheck(const lightning::IMUPtr &msg_in);
        /// 保存前端的地图
        void SaveDownMap();

        void SaveMap();

        /// 获取关键帧
        Keyframe::Ptr GetKeyframe() const { return last_kf_; }

        /// 获取激光的状态（IMU in LIO world）
        NavState GetState() const { return state_point_; }

        /// 全部外参左乘调平：T' = V^{-1} * T_w_imu（旋转+平移；odom/TF/path/bag/Loc）
        NavState GetExtrinsicRemappedState(const NavState &state) const;

        /// 获取IMU状态
        NavState GetIMUState() const
        {
            if (p_imu_->IsIMUInited())
                return kf_imu_.GetX();
            else
            {
                NavState s;
                s.pose_is_ok_ = false;
                return s;
            }
        }

        CloudPtr GetScanUndist() const { return scan_undistort_; }
        CloudPtr GetProjCloud();

        void PublishFrameWorld(const std::shared_ptr<PubROSTopic> &pub);

        void PublishFrameWorldRgb(const std::shared_ptr<PubROSTopic> &pub);

        void PublishLivoVisualizations(const std::shared_ptr<PubROSTopic> &pub);

        void PublishFrameBody(const std::shared_ptr<PubROSTopic> &pub);

        void Reset(SE3 &pose);

        std::vector<Keyframe::Ptr> GetAllKeyframes() { return all_keyframes_; }

        /**
         * 计算全局地图
         * @param use_lio_pose
         * @return
         */
        CloudPtr GetGlobalMap(bool use_lio_pose, bool use_voxel = true, float res = 0.1);

        void SetRviz(std::shared_ptr<PubROSTopic> pub) { pub_ = pub; }

        bool need_trigger_imu = false;
        double lidar_end_time_ = 0;

        /// 将附近的关键帧投影至cloud中
        void ProjectKFs(CloudPtr cloud, int size_limit = 1000);

    private:
        enum class LivoStep
        {
            NONE,
            LIO,
            VIO,
        };

        struct ImageFrame
        {
            int cam_id = 0;
            double timestamp = 0.0;
            cv::Mat image;
        };

        std::shared_ptr<PubROSTopic> pub_ = nullptr;
        // sync lidar with imu
        bool SyncPackages();
        bool SyncLIVOPackages(LivoStep &step);
        PointWithVar BuildPointWithVar(const PointType &point_body);
        std::vector<PointWithVar> ExportPgForVio();

        void ObsModel(NavState &s, ESKF::CustomObservationModel &obs);
        void ObsModelVoxelMap(NavState &s, ESKF::CustomObservationModel &obs);
        void RunFastVio(cv::Mat image);
        void AccAsGravityObsModel(NavState &s, ESKF::CustomObservationModel &obs);
        void BiasObsModel(NavState &s, ESKF::CustomObservationModel &obs);
        void StationaryBiasAndAccObsModel(NavState &s, ESKF::CustomObservationModel &obs);

        inline void PointBodyToWorld(const PointType &pi, PointType &po)
        {
            Vec3d p_global(state_point_.rot_ * (state_point_.offset_R_lidar_ * pi.getVector3fMap().cast<double>() +
                                                state_point_.offset_t_lidar_) +
                           state_point_.pos_);

            po.x = p_global(0);
            po.y = p_global(1);
            po.z = p_global(2);
            po.intensity = pi.intensity;
        }

        inline void PointBodyLidarToIMU(const PointType &pi, PointType &po)
        {
            Vec3d p_global(state_point_.offset_R_lidar_ * pi.getVector3fMap().cast<double>() + state_point_.offset_t_lidar_);

            po.x = p_global(0);
            po.y = p_global(1);
            po.z = p_global(2);
            po.intensity = pi.intensity;
        }

        void MapIncremental();

        bool ApplyConfig(const Config &cfg);

        /// 创建关键帧
        void MakeKF();

        void SortVec(const Eigen::VectorXf &vec, Eigen::VectorXf &sorted_vec, Eigen::VectorXi &ind);

        void HuberWeight(float &res);

    private:
        Options options_;

        /// modules
        IVoxType::Options ivox_options_;
        std::shared_ptr<IVoxType> ivox_ = nullptr;              // localmap in ivox (iVox LIO)
        VoxelMapConfig voxel_map_config_;
        std::shared_ptr<VoxelMap> voxel_map_ = nullptr;         // FAST voxel plane map (VoxelMap LIO / LIVO)
        std::shared_ptr<PointCloudPreprocess> p_pre_ = nullptr; // point cloud preprocess
        std::shared_ptr<ImuProcess> p_imu_ = nullptr;           // imu process

        /// local map related
        double filter_size_map_min_ = 0;

        /// params
        std::vector<double> extrinT_{3, 0.0}; // lidar-imu translation
        std::vector<double> extrinR_{9, 0.0}; // lidar-imu rotation
        std::string map_file_path_;

        std::vector<Keyframe::Ptr> all_keyframes_;
        Keyframe::Ptr last_kf_ = nullptr;
        int kf_id_ = 0;
        /// point clouds data
        CloudPtr scan_undistort_{new PointCloudType()};  // scan after undistortion
        CloudPtr scan_down_body_{new PointCloudType()};  // downsampled scan in body
        CloudPtr scan_down_world_{new PointCloudType()}; // downsampled scan in world
        CloudPtr pcl_wait_save_{new PointCloudType()};
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_wait_save_rgb_{new pcl::PointCloud<pcl::PointXYZRGB>()};
        std::vector<PointVector> nearest_points_; // nearest points of current scan
        std::vector<Vec4f> corr_pts_;             // inlier pts
        std::vector<Vec4f> corr_norm_;            // inlier plane norms
        pcl::VoxelGrid<PointType> voxel_scan_;    // voxel filter for current scan

        /// 点面相关
        std::vector<float> residuals_;            // point-to-plane residuals
        std::vector<char> point_selected_surf_;   // selected points
        std::vector<Vec4f> plane_coef_;           // plane coeffs

        /// 点到点相关
        std::vector<char> point_selected_icp_; // 点到点的selected points

        std::mutex mtx_buffer_;
        std::deque<double> time_buffer_;

        std::deque<PointCloudType::Ptr> lidar_buffer_;
        std::deque<lightning::IMUPtr> imu_buffer_;
        std::vector<std::deque<ImageFrame>> image_buffers_;

        /// options
        double last_timestamp_lidar_ = -1.0;
        std::vector<double> last_timestamp_img_;
        int imu_reset_level_ = 1;

        double last_timestamp_imu_ = -1.0;
        double cur_timestamp_imu_ = -1.0;
        lightning::IMUPtr last_imu_;
        double first_lidar_time_ = 0.0;
        bool lidar_pushed_ = false;
        int omp_num_ = 4;

        /// statistics and flags ///
        int scan_count_ = 0;
        int publish_count_ = 0;
        bool flg_first_scan_ = true;
        bool flg_EKF_inited_ = false;
        double lidar_mean_scantime_ = 0.0;
        int scan_num_ = 0;
        int effect_feat_surf_ = 0, frame_num_ = 0, effect_feat_icp_ = 0;

        double last_lidar_time_ = 0;

        ///////////////////////// EKF inputs and output ///////////////////////////////////////////////////////
        MeasureGroup measures_; // sync IMU and lidar scan

        ESKF kf_;     // 点云时刻的IMU状态
        ESKF kf_imu_; // imu 最新时刻的eskf状态

        NavState state_point_; // ekf current state
        NavState livo_propagated_state_;
        ESKF::CovType livo_propagated_cov_ = ESKF::CovType::Identity();

        bool visual_en_ = false;
        bool use_voxel_map_ = false;  // FAST VoxelMap LIO; forced true when visual_en_
        int lio_max_iterations_ = 5;
        int camera_num_ = 1;
        double image_time_offset_ = 0.0;
        double exposure_time_init_ = 0.0;
        double inv_expo_cov_ = 0.1;
        bool vio_raycast_en_ = false;
        bool vio_normal_en_ = true;
        bool vio_exposure_estimate_en_ = true;
        bool vio_inverse_composition_en_ = false;
        bool vio_ncc_en_ = false;
        int vio_grid_size_ = 5;
        int vio_grid_n_height_ = 17;
        int vio_patch_size_ = 8;
        int vio_patch_pyrimid_level_ = 4;
        int vio_max_iterations_ = 5;
        double vio_outlier_threshold_ = 1000.0;
        double vio_img_point_cov_ = 100.0;
        double vio_ncc_threshold_ = 0.85;
        std::vector<double> camera_Rcl_;
        std::vector<double> camera_Pcl_;
        LivoStep last_livo_step_ = LivoStep::NONE;
        bool last_run_was_vio_ = false;
        /// FAST pcl_w_wait_pub: set after a LIO that produced a world cloud; VIO no-ops if false.
        bool livo_has_lio_cloud_ = false;
        std::vector<CameraSlot> camera_slots_;
        std::vector<cv::Mat> synced_images_;
        double synced_image_time_ = 0.0;
        double last_livo_update_time_ = -1.0;
        CloudPtr livo_pcl_proc_next_{new PointCloudType()};
        std::vector<PointWithVar> pg_list_;
        VIOManagerPtr vio_manager_;
        StatesGroup vio_state_;
        StatesGroup vio_state_propagat_;

        bool buffer_images_for_dyn_ = false;
        std::unique_ptr<perception::DynamicFilter> dynamic_filter_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_dyn_removed_body_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_dyn_removed_world_;
        void TryFilterDynamicObjects(double stamp);
        void PublishDynamicRemovedCloud(double stamp);

        bool use_aa_ = false;               // use anderson acceleration?
        std::list<Keyframe::Ptr> proj_kfs_; // 投影到当前帧的关键帧
        bool dense_pcd_save_en_ = false;

        //------------------------------------------------------

    private:
        bool path_en_ = true;
        bool scan_pub_en_ = false;
        bool dense_pub_en_ = false;
        bool scan_body_pub_en_ = false;
        bool rgb_img_pub_en_ = false;
        bool rgb_cloud_pub_en_ = false;
        double blind_rgb_points_ = 0.0;
        bool enable_debug_log_ = false;

        bool IMU_inited_ = false;
        double last_reset_wall_time_ = -1.0;
        double stationary_acc_noise_ = 1e-3;
        double stationary_bias_noise_ = 1e-4;

        std::deque<lightning::IMUPtr> difference_in_acceleration_;
        Eigen::Isometry3d T_body_lio_;
        int lidar_type_;
    };

}

#endif // namespace lightning