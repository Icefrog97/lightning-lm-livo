#include "ros/param_loader.h"

#include "STDesc/STDesc.h"
#include "common/options.h"

#include <algorithm>
#include <cctype>
#include "ros/ros2_types.h"

namespace lightning
{
namespace
{

double CastParam(const rclcpp::Parameter &p, const double &)
{
    if (p.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
        return static_cast<double>(p.as_int());
    return p.as_double();
}

int CastParam(const rclcpp::Parameter &p, const int &)
{
    if (p.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
        return static_cast<int>(p.as_double());
    return static_cast<int>(p.as_int());
}

bool CastParam(const rclcpp::Parameter &p, const bool &) { return p.as_bool(); }

std::string CastParam(const rclcpp::Parameter &p, const std::string &) { return p.as_string(); }

std::vector<double> CastParam(const rclcpp::Parameter &p, const std::vector<double> &)
{
    if (p.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY)
    {
        std::vector<double> out;
        for (int64_t v : p.as_integer_array())
            out.push_back(static_cast<double>(v));
        return out;
    }
    return p.as_double_array();
}

std::vector<int> CastParam(const rclcpp::Parameter &p, const std::vector<int> &)
{
    // Humble types an empty array as a byte array because it has no element type.
    if (p.get_type() == rclcpp::ParameterType::PARAMETER_BYTE_ARRAY && p.as_byte_array().empty())
        return {};
    std::vector<int> out;
    for (int64_t v : p.as_integer_array())
        out.push_back(static_cast<int>(v));
    return out;
}

std::vector<std::string> CastParam(const rclcpp::Parameter &p, const std::vector<std::string> &)
{
    return p.as_string_array();
}

template <typename T>
T Param(const rclcpp::Node::SharedPtr &nh, const std::string &key, const T &fallback)
{
    const std::string name = RosParamName(key);
    if (!nh->has_parameter(name))
        return fallback;
    rclcpp::Parameter value;
    if (!nh->get_parameter(name, value))
        return fallback;
    return CastParam(value, fallback);
}

template <typename T>
bool ReadParam(const rclcpp::Node::SharedPtr &nh, const std::string &key, T &value)
{
    const std::string name = RosParamName(key);
    if (!nh->has_parameter(name))
        return false;
    value = Param<T>(nh, key, value);
    return true;
}

bool ReadVectorParam(const rclcpp::Node::SharedPtr &nh, const std::string &key, std::vector<double> &value)
{
    return ReadParam(nh, key, value);
}

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

Mat3d Mat3FromVector(const std::vector<double> &v)
{
    Mat3d m = Mat3d::Identity();
    if (v.size() == 9)
    {
        m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
    }
    return m;
}

Vec3d Vec3FromVector(const std::vector<double> &v)
{
    if (v.size() == 3)
        return Vec3d(v[0], v[1], v[2]);
    return Vec3d::Zero();
}

SE3 SE3FromRT(const std::vector<double> &R, const std::vector<double> &t)
{
    return SE3(SO3(Quatd(Mat3FromVector(R))), Vec3FromVector(t));
}

Eigen::Isometry3d LevelFromArray(const std::vector<double> &t_v, const std::vector<double> &e_v)
{
    const Eigen::Vector3d t(t_v[0], t_v[1], t_v[2]);
    const Eigen::Vector3d e(e_v[0] * M_PI / 180.0, e_v[1] * M_PI / 180.0, e_v[2] * M_PI / 180.0);
    const Eigen::Quaterniond q = Eigen::Quaterniond(Eigen::AngleAxisd(e(0), Eigen::Vector3d::UnitZ()) *
                                                    Eigen::AngleAxisd(e(1), Eigen::Vector3d::UnitY()) *
                                                    Eigen::AngleAxisd(e(2), Eigen::Vector3d::UnitX()));
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = q.toRotationMatrix();
    T.translation() = t;
    return T;
}

std::vector<std::string> LoadImageTopics(const rclcpp::Node::SharedPtr &nh)
{
    std::vector<std::string> image_topics;
    if (!ReadParam(nh, "common/image_topics", image_topics) || image_topics.empty())
        image_topics.push_back(Param<std::string>(nh, "common/image_topic", std::string("/camera/image_raw")));
    return image_topics;
}

Eigen::Isometry3d LoadBodyLevel(const rclcpp::Node::SharedPtr &nh)
{
    const std::vector<double> t_default{-1, 0, -0.1};
    const std::vector<double> e_default{-0.8611351114, -9.9666036744, -0.5294170198};
    return LevelFromArray(Param<std::vector<double>>(nh, "common/T_body_wrt_lio", t_default),
                          Param<std::vector<double>>(nh, "common/E_body_wrt_lio", e_default));
}

void LoadStdesc(const rclcpp::Node::SharedPtr &nh, STDesc::ConfigSetting &config_setting)
{
    config_setting.ds_size_ = Param<double>(nh, "ds_size", 0.5);
    config_setting.maximum_corner_num_ = Param<int>(nh, "maximum_corner_num", 100);
    config_setting.plane_merge_normal_thre_ = Param<double>(nh, "plane_merge_normal_thre", 0.1);
    config_setting.plane_detection_thre_ = Param<double>(nh, "plane_detection_thre", 0.01);
    config_setting.voxel_size_ = Param<double>(nh, "voxel_size", 2.0);
    config_setting.voxel_init_num_ = Param<int>(nh, "voxel_init_num", 10);
    config_setting.proj_image_resolution_ = Param<double>(nh, "proj_image_resolution", 0.5);
    config_setting.proj_dis_min_ = Param<double>(nh, "proj_dis_min", 0);
    config_setting.proj_dis_max_ = Param<double>(nh, "proj_dis_max", 2);
    config_setting.corner_thre_ = Param<double>(nh, "corner_thre", 10);
    config_setting.descriptor_near_num_ = Param<int>(nh, "descriptor_near_num", 10);
    config_setting.descriptor_min_len_ = Param<double>(nh, "descriptor_min_len", 2);
    config_setting.descriptor_max_len_ = Param<double>(nh, "descriptor_max_len", 50);
    config_setting.non_max_suppression_radius_ = Param<double>(nh, "non_max_suppression_radius", 2.0);
    config_setting.std_side_resolution_ = Param<double>(nh, "std_side_resolution", 0.2);
    config_setting.skip_near_num_ = Param<int>(nh, "skip_near_num", 50);
    config_setting.omp_nums_ = Param<int>(nh, "system/omp_nums", 8);
    config_setting.candidate_num_ = Param<int>(nh, "candidate_num", 50);
    config_setting.sub_frame_num_ = Param<int>(nh, "sub_frame_num", 10);
    config_setting.rough_dis_threshold_ = Param<double>(nh, "rough_dis_threshold", 0.01);
    config_setting.vertex_diff_threshold_ = Param<double>(nh, "vertex_diff_threshold", 0.5);
    config_setting.icp_threshold_ = Param<double>(nh, "icp_threshold", 0.5);
    config_setting.normal_threshold_ = Param<double>(nh, "normal_threshold", 0.2);
    config_setting.dis_threshold_ = Param<double>(nh, "dis_threshold", 0.5);
    config_setting.center_dis_threshold_ = Param<double>(nh, "center_dis_threshold", 40);
}

void LoadCameras(const rclcpp::Node::SharedPtr &nh, LaserMapping::Config &cfg)
{
    if (!cfg.visual_en && !cfg.dynamic_filter_enable)
        return;

    const std::vector<std::string> image_topics = LoadImageTopics(nh);
    const SE3 T_imu_lidar = SE3FromRT(cfg.extrin_R, cfg.extrin_T);
    const SE3 T_lidar_imu = T_imu_lidar.inverse();
    const int num_cameras = std::max(1, cfg.camera_num);
    cfg.camera_slots.clear();
    cfg.camera_slots.reserve(num_cameras);
    for (int i = 0; i < num_cameras; ++i)
    {
        const std::string suffix = num_cameras == 1 ? "" : std::to_string(i);
        std::vector<double> Rcl, Pcl;
        if (!ReadVectorParam(nh, "camera/Rcl" + suffix, Rcl))
            Rcl = Param<std::vector<double>>(nh, "camera/Rcl", {1, 0, 0, 0, 1, 0, 0, 0, 1});
        if (!ReadVectorParam(nh, "camera/Pcl" + suffix, Pcl))
            Pcl = Param<std::vector<double>>(nh, "camera/Pcl", {0, 0, 0});

        CameraSlot slot;
        slot.cam_id = i;
        if (i < static_cast<int>(image_topics.size()))
            slot.topic = image_topics[i];
        if (!ReadParam(nh, "camera/model" + suffix, slot.model))
            slot.model = Param<std::string>(nh, "cam_model" + suffix, std::string("Pinhole"));
        if (!ReadParam(nh, "camera/width" + suffix, slot.width))
            slot.width = Param<int>(nh, "cam_width" + suffix, 0);
        if (!ReadParam(nh, "camera/height" + suffix, slot.height))
            slot.height = Param<int>(nh, "cam_height" + suffix, 0);
        slot.scale = Param<double>(nh, "camera/scale" + suffix, 1.0);
        slot.scale = Param<double>(nh, "scale" + suffix, slot.scale);

        std::vector<double> intrinsics;
        if (!ReadVectorParam(nh, "camera/intrinsics" + suffix, intrinsics))
            intrinsics = Param<std::vector<double>>(nh, "camera/intrinsics", {});
        if (intrinsics.size() >= 4)
        {
            slot.fx = intrinsics[0];
            slot.fy = intrinsics[1];
            slot.cx = intrinsics[2];
            slot.cy = intrinsics[3];
        }
        else
        {
            slot.fx = Param<double>(nh, "camera/fx" + suffix, 0.0);
            slot.fy = Param<double>(nh, "camera/fy" + suffix, 0.0);
            slot.cx = Param<double>(nh, "camera/cx" + suffix, 0.0);
            slot.cy = Param<double>(nh, "camera/cy" + suffix, 0.0);
            slot.fx = Param<double>(nh, "cam_fx" + suffix, slot.fx);
            slot.fy = Param<double>(nh, "cam_fy" + suffix, slot.fy);
            slot.cx = Param<double>(nh, "cam_cx" + suffix, slot.cx);
            slot.cy = Param<double>(nh, "cam_cy" + suffix, slot.cy);
        }

        if (!ReadVectorParam(nh, "camera/distortion" + suffix, slot.distortion))
            slot.distortion = Param<std::vector<double>>(nh, "camera/distortion", {});
        if (slot.distortion.empty())
        {
            if (IsFisheyeModel(slot.model))
            {
                double k1 = 0.0, k2 = 0.0, k3 = 0.0, k4 = 0.0;
                k1 = Param<double>(nh, "k1" + suffix, 0.0);
                k2 = Param<double>(nh, "k2" + suffix, 0.0);
                k3 = Param<double>(nh, "k3" + suffix, 0.0);
                k4 = Param<double>(nh, "k4" + suffix, 0.0);
                slot.distortion = {k1, k2, k3, k4};
            }
            else
            {
                double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0, d4 = 0.0;
                d0 = Param<double>(nh, "cam_d0" + suffix, 0.0);
                d1 = Param<double>(nh, "cam_d1" + suffix, 0.0);
                d2 = Param<double>(nh, "cam_d2" + suffix, 0.0);
                d3 = Param<double>(nh, "cam_d3" + suffix, 0.0);
                d4 = Param<double>(nh, "cam_d4" + suffix, 0.0);
                slot.distortion = {d0, d1, d2, d3, d4};
            }
        }

        slot.undistort_scale_x = Param<double>(nh, "camera/undistort_scale_x" + suffix, 1.0);
        slot.undistort_scale_y = Param<double>(nh, "camera/undistort_scale_y" + suffix, 1.0);
        slot.undistort_shift_x = Param<double>(nh, "camera/undistort_shift_x" + suffix, 0.0);
        slot.undistort_shift_y = Param<double>(nh, "camera/undistort_shift_y" + suffix, 0.0);
        slot.undistort_scale_x = Param<double>(nh, "undistort_scale_x" + suffix, slot.undistort_scale_x);
        slot.undistort_scale_y = Param<double>(nh, "undistort_scale_y" + suffix, slot.undistort_scale_y);
        slot.undistort_shift_x = Param<double>(nh, "undistort_shift_x" + suffix, slot.undistort_shift_x);
        slot.undistort_shift_y = Param<double>(nh, "undistort_shift_y" + suffix, slot.undistort_shift_y);

        const SE3 T_cam_lidar = SE3FromRT(Rcl, Pcl);
        slot.T_cam_imu = T_cam_lidar * T_lidar_imu;
        if (i == 0)
        {
            cfg.camera_Rcl = Rcl;
            cfg.camera_Pcl = Pcl;
        }
        cfg.camera_slots.push_back(slot);
    }
}

void LoadLaserMappingConfig(const rclcpp::Node::SharedPtr &nh, LaserMapping::Config &cfg)
{
    cfg.enable_debug_log = Param<bool>(nh, "system/enable_debug_log", false);
    cfg.map_origin_path = Param<std::string>(nh, "system/map_origin_path", "");

    cfg.path_en = Param<bool>(nh, "publish/path_en", true);
    cfg.scan_pub_en = Param<bool>(nh, "publish/scan_publish_en", true);
    cfg.dense_pub_en = Param<bool>(nh, "publish/dense_publish_en", true);
    cfg.scan_body_pub_en = Param<bool>(nh, "publish/scan_bodyframe_pub_en", true);
    cfg.rgb_img_pub_en = Param<bool>(nh, "publish/rgb_img_en", false);
    cfg.rgb_cloud_pub_en = Param<bool>(nh, "publish/rgb_cloud_en", false);
    cfg.blind_rgb_points = Param<double>(nh, "publish/blind_rgb_points", 0.0);

    cfg.visual_en = Param<bool>(nh, "visual/enable", false);
    cfg.use_voxel_map = Param<bool>(nh, "lio/use_voxel_map", false);
    if (cfg.visual_en)
        cfg.use_voxel_map = true;
    cfg.lio_max_iterations = Param<int>(nh, "lio/max_iterations", 5);
    cfg.dynamic_filter_enable = Param<bool>(nh, "dynamic_filter/enable", false);
    cfg.camera_num = Param<int>(nh, "camera/num", 1);
    cfg.image_time_offset = Param<double>(nh, "time_offset/img_time_offset", 0.0);
    cfg.exposure_time_init = Param<double>(nh, "camera/exposure_time_init", 0.0);
    cfg.vio_raycast_en = Param<bool>(nh, "vio/raycast_en", false);
    cfg.vio_normal_en = Param<bool>(nh, "vio/normal_en", true);
    cfg.vio_exposure_estimate_en = Param<bool>(nh, "vio/exposure_estimate_en", true);
    cfg.vio_inverse_composition_en = Param<bool>(nh, "vio/inverse_composition_en", false);
    cfg.inv_expo_cov = Param<double>(nh, "vio/inv_expo_cov", 0.1);
    cfg.vio_grid_size = Param<int>(nh, "vio/grid_size", 5);
    cfg.vio_grid_n_height = Param<int>(nh, "vio/grid_n_height", 17);
    cfg.vio_patch_size = Param<int>(nh, "vio/patch_size", 8);
    cfg.vio_patch_pyrimid_level = Param<int>(nh, "vio/patch_pyrimid_level", 4);
    cfg.vio_max_iterations = Param<int>(nh, "vio/max_iterations", 5);
    cfg.vio_outlier_threshold = Param<double>(nh, "vio/outlier_threshold", 1000.0);
    cfg.vio_img_point_cov = Param<double>(nh, "vio/img_point_cov", 100.0);
    cfg.vio_ncc_en = Param<bool>(nh, "vio/ncc_en", false);
    cfg.vio_ncc_threshold = Param<double>(nh, "vio/ncc_threshold", 0.85);
    cfg.visual_lru_capacity = Param<int>(nh, "visual/lru_capacity", 100000);
    cfg.visual_max_points_per_voxel = Param<int>(nh, "visual/max_points_per_voxel", 10);

    cfg.dynamic_filter.enable = cfg.dynamic_filter_enable;
    cfg.dynamic_filter.yolo.model_path = Param<std::string>(nh, "dynamic_filter/model_path", "");
    cfg.dynamic_filter.yolo.imgsz = Param<int>(nh, "dynamic_filter/imgsz", 640);
    cfg.dynamic_filter.yolo.conf_thres = Param<double>(nh, "dynamic_filter/conf_thres", 0.4);
    cfg.dynamic_filter.yolo.iou_thres = Param<double>(nh, "dynamic_filter/iou_thres", 0.45);
    cfg.dynamic_filter.yolo.num_classes = Param<int>(nh, "dynamic_filter/num_classes", 80);
    cfg.dynamic_filter.dilate_px = Param<int>(nh, "dynamic_filter/dilate_px", 5);
    cfg.dynamic_filter.max_range = Param<double>(nh, "dynamic_filter/max_range", 60.0);
    cfg.dynamic_filter.publish_removed_cloud = Param<bool>(nh, "dynamic_filter/publish_removed_cloud", true);
    cfg.dynamic_filter.removed_body_topic =
        Param<std::string>(nh, "dynamic_filter/removed_body_topic", "/dynamic_filter/removed_body");
    cfg.dynamic_filter.removed_world_topic =
        Param<std::string>(nh, "dynamic_filter/removed_world_topic", "/dynamic_filter/removed");
    // config 里是 []。Humble 的参数文件不能保存空数组，launch 会省掉这个键。
    // 空列表和 yaml 一样：保留全部类别。键写成非空数组时仍用那个数组。
    cfg.dynamic_filter.yolo.keep_class_ids =
        Param<std::vector<int>>(nh, "dynamic_filter/keep_class_ids", {});

    cfg.max_iteration = Param<int>(nh, "fasterlio/max_iteration", 4);
    cfg.degeneracy_threshold_ratio = Param<double>(nh, "fasterlio/degeneracy_threshold_ratio", 0.2);
    cfg.esti_plane_threshold = Param<double>(nh, "common/esti_plane_threshold", 0.001);
    cfg.filter_size_surf = Param<double>(nh, "common/filter_size_surf", 0.5);
    if (cfg.use_voxel_map)
        cfg.filter_size_surf = Param<double>(nh, "common/livo_filter_size_surf", 0.4);
    cfg.filter_size_map = Param<double>(nh, "common/filter_size_map", 0.5);
    cfg.omp_num = Param<int>(nh, "mapping/omp_num", 4);
    cfg.gyr_cov = Param<double>(nh, "mapping/gyr_cov", 0.1);
    cfg.acc_cov = Param<double>(nh, "mapping/acc_cov", 0.1);
    cfg.b_gyr_cov = Param<double>(nh, "mapping/b_gyr_cov", 0.0001);
    cfg.b_acc_cov = Param<double>(nh, "mapping/b_acc_cov", 0.0001);
    cfg.extrin_T = Param<std::vector<double>>(nh, "mapping/extrinsic_T", {});
    cfg.extrin_R = Param<std::vector<double>>(nh, "mapping/extrinsic_R", {});
    cfg.use_aa = Param<bool>(nh, "mapping/use_aa", false);
    cfg.kf_dis_th = Param<double>(nh, "mapping/kf_dis_th", 1.0);
    cfg.kf_angle_th = Param<int>(nh, "mapping/kf_angle_th", 10) * M_PI / 180.0;

    LoadCameras(nh, cfg);

    cfg.blind = Param<double>(nh, "preprocess/blind", 0.01);
    cfg.lidar_type = Param<int>(nh, "preprocess/lidar_type", static_cast<int>(LidarType::mid360));
    cfg.point_filter_num = Param<int>(nh, "preprocess/point_filter_num", 1);
    cfg.scan_line = Param<int>(nh, "preprocess/scan_line", 1);
    cfg.height_max = Param<double>(nh, "preprocess/height_max", 0.01);
    cfg.height_min = Param<double>(nh, "preprocess/height_min", 0.01);
    cfg.dense_pcd_save_en = Param<bool>(nh, "pcd_save/dense_pcd_save_en", false);

    cfg.ivox_grid_resolution = Param<double>(nh, "fasterlio/ivox_grid_resolution", 2.0);
    cfg.plane_icp_weight = Param<double>(nh, "fasterlio/plane_icp_weight", 2.0);
    cfg.ivox_nearby_type = Param<int>(nh, "fasterlio/ivox_nearby_type", 0);
    cfg.ivox_capacity = Param<int>(nh, "fasterlio/ivox_capacity", 1000000);
    cfg.enable_icp_part = Param<bool>(nh, "fasterlio/enable_icp_part", true);
    cfg.proj_kfs = Param<bool>(nh, "fasterlio/proj_kfs", true);
    cfg.max_layer = Param<int>(nh, "lio/max_layer", 2);
    cfg.voxel_size = Param<double>(nh, "lio/voxel_size", 0.5);
    cfg.min_eigen_value = Param<double>(nh, "lio/min_eigen_value", 0.0025);
    cfg.sigma_num = Param<double>(nh, "lio/sigma_num", 3.0);
    cfg.beam_err = Param<double>(nh, "lio/beam_err", 0.05);
    cfg.dept_err = Param<double>(nh, "lio/dept_err", 0.02);
    cfg.max_points_num = Param<int>(nh, "lio/max_points_num", 50);
    cfg.layer_init_num = Param<std::vector<int>>(nh, "lio/layer_init_num", {5, 5, 5, 5, 5});
    cfg.voxel_lru_capacity = Param<int>(nh, "lio/lru_capacity", 100000);
    cfg.use_imu_filter = Param<bool>(nh, "imu_filter/use_imu_filter", true);
    cfg.imu_reset_level = Param<int>(nh, "fasterlio/imu_reset_level", 2);
    cfg.min_pts = Param<int>(nh, "fasterlio/min_pts", 2);
    cfg.use_degeneracy = Param<bool>(nh, "fasterlio/use_degeneracy", true);
    cfg.max_init_count = Param<int>(nh, "fasterlio/max_init_count", 10);
    cfg.T_body_lio = LoadBodyLevel(nh);
}

void LoadOnlineLoop(const rclcpp::Node::SharedPtr &nh, OnlineLoop::Options &opt)
{
    LoadStdesc(nh, opt.stdesc);
    opt.use_height_factor = Param<bool>(nh, "use_height_factor", true);
    opt.filter_size_map = Param<double>(nh, "map/filter_size_map", 0.2);
    opt.filter_size_src = Param<double>(nh, "map/filter_size_src", 0.2);
    opt.point_filter_num = Param<int>(nh, "point_filter_num", 5);
    opt.update_graph_freq = Param<double>(nh, "loop/update_graph_freq", 1.0);
    opt.pcd_path = Param<std::string>(nh, "system/pcd_path", "");
    opt.loop_map_name = Param<std::string>(nh, "system/loop_map_name", std::string("global_loop_map.pcd"));
    opt.map_path = Param<std::string>(nh, "system/map_path", "");
    opt.bbs_min_level_res = Param<double>(nh, "lidar_loc/bbs_min_level_res", 1.0);
    opt.bbs_max_level = Param<int>(nh, "lidar_loc/bbs_max_level", 6);
    opt.use_enu_results = Param<bool>(nh, "pcd_save/enu_results", true);
    const std::vector<double> t_default{-1, 0, -0.1};
    const std::vector<double> e_default{-0.8611351114, -9.9666036744, -0.5294170198};
    opt.T_body_wrt_lio = Param<std::vector<double>>(nh, "common/T_body_wrt_lio", t_default);
    opt.E_body_wrt_lio = Param<std::vector<double>>(nh, "common/E_body_wrt_lio", e_default);
    opt.odom_rot_sigma_deg = Param<double>(nh, "loop/odom_rot_sigma_deg", 0.2);
    opt.odom_trans_sigma = Param<double>(nh, "loop/odom_trans_sigma", 0.03);
}

void LoadPub(const rclcpp::Node::SharedPtr &nh, PubROSTopic::Options &opt, bool mapping)
{
    if (mapping)
    {
        opt.pcd_path = Param<std::string>(nh, "system/pcd_path", "");
        opt.dense_map_name = Param<std::string>(nh, "system/dense_map_name", "");
        opt.down_map_name = Param<std::string>(nh, "system/down_map_name", "");
        opt.rgb_img_en = Param<bool>(nh, "publish/rgb_img_en", false);
        opt.rgb_cloud_en = Param<bool>(nh, "publish/rgb_cloud_en", false);
    }
    opt.pub_path_num = Param<int>(nh, "publish/pub_path_num", 10);
}

void LoadLidarLoc(const rclcpp::Node::SharedPtr &nh, loc::LidarLoc::Options &opt)
{
    opt.map_option_.max_pts_in_dyn_chunk_ =
        static_cast<size_t>(Param<int>(nh, "maps/max_pts_dyn_chunk", 500));
    opt.update_kf_dis_ = Param<double>(nh, "lidar_loc/update_kf_dis", 1.0);
    opt.update_lidar_loc_score_ = Param<double>(nh, "lidar_loc/update_lidar_loc_score", 0.8);
    opt.min_init_confidence_ = Param<double>(nh, "lidar_loc/min_init_confidence", 0.5);
    opt.filter_z_max_ = Param<double>(nh, "lidar_loc/filter_z_max", 2.0);
    opt.lidar_loc_odom_th_ = Param<double>(nh, "lidar_loc/lidar_loc_odom_th", 0.5);
    opt.init_with_fp_ = Param<bool>(nh, "lidar_loc/init_with_fp", true);
    opt.init_with_recover_ = Param<bool>(nh, "lidar_loc/init_with_recover", true);
    opt.enable_icp_adjust_ = Param<bool>(nh, "lidar_loc/enable_icp_adjust", false);
    opt.try_self_extrap_ = Param<bool>(nh, "lidar_loc/try_self_extrap", true);
    opt.add_recover_to_index_ = Param<bool>(nh, "lidar_loc/add_recover_to_index", false);
    opt.use_3dbbs_just_reset_ = Param<bool>(nh, "lidar_loc/use_3dbbs_just_reset", true);
    opt.use_3d_bbs_ = Param<bool>(nh, "lidar_loc/use_3d_bbs", true);
    opt.bbs_backend_ = Param<std::string>(nh, "lidar_loc/bbs_backend", std::string("auto"));
    opt.bbs_min_level_res_ = Param<double>(nh, "lidar_loc/bbs_min_level_res", 1.0);
    opt.bbs_max_level_ = Param<int>(nh, "lidar_loc/bbs_max_level", 6);
    opt.bbs_score_threshold_percentage_ = Param<double>(nh, "lidar_loc/bbs_score_threshold_percentage", 0.7);
    opt.bbs_timeout_msec_ = Param<int>(nh, "lidar_loc/bbs_timeout_msec", 0);
    opt.bbs_map_leaf_size_ = Param<double>(nh, "lidar_loc/bbs_map_leaf_size", 0.2);
    opt.bbs_src_leaf_size_ = Param<double>(nh, "lidar_loc/bbs_src_leaf_size", 0.2);
    opt.bbs_min_scan_range_ = Param<double>(nh, "lidar_loc/bbs_min_scan_range", 0.2);
    opt.bbs_max_scan_range_ = Param<double>(nh, "lidar_loc/bbs_max_scan_range", 0.2);
    opt.bbs_min_z_ = Param<double>(nh, "lidar_loc/bbs_min_z", -2.0);
    opt.bbs_max_z_ = Param<double>(nh, "lidar_loc/bbs_max_z", 2.0);
    opt.bbs_use_gicp_ = Param<bool>(nh, "lidar_loc/bbs_use_gicp", false);
    opt.bbs_use_voxelmap_coord_ = Param<bool>(nh, "lidar_loc/bbs_use_voxelmap_coord", false);
    opt.ndt_omp_th_ = Param<int>(nh, "lidar_loc/ndt_omp_th", 1);
    opt.with_height_ = Param<bool>(nh, "loop_closing/with_height", false);
    opt.update_dynamic_cloud_ = Param<bool>(nh, "lidar_loc/update_dynamic_cloud", true);
    opt.force_2d_ = Param<bool>(nh, "lidar_loc/force_2d", true);
    opt.enable_debug_log_ = Param<bool>(nh, "system/enable_debug_log", false);

    std::vector<double> bbs_min_rpy;
    if (ReadParam(nh, "lidar_loc/bbs_min_rpy", bbs_min_rpy) && bbs_min_rpy.size() == 3)
        opt.bbs_min_rpy_ = Eigen::Vector3d(bbs_min_rpy[0], bbs_min_rpy[1], bbs_min_rpy[2]);
    std::vector<double> bbs_max_rpy;
    if (ReadParam(nh, "lidar_loc/bbs_max_rpy", bbs_max_rpy) && bbs_max_rpy.size() == 3)
        opt.bbs_max_rpy_ = Eigen::Vector3d(bbs_max_rpy[0], bbs_max_rpy[1], bbs_max_rpy[2]);

    const std::string map_path = Param<std::string>(nh, "system/map_path", "");
    opt.map_option_.map_path_ = map_path;
    opt.map_option_.global_loop_map_name_ = Param<std::string>(nh, "system/global_loop_map_name", "");
    opt.bbs_target_clouds_ = map_path + "VoxelMap/" + opt.map_option_.global_loop_map_name_;
    opt.recover_pose_path_ = map_path + "recover.txt";

    const std::string map_policy = Param<std::string>(nh, "maps/dyn_cloud_policy", "");
    if (map_policy == "short")
        opt.map_option_.policy_ = FullMap::DynamicCloudPolicy::SHORT;
    else if (map_policy == "long")
        opt.map_option_.policy_ = FullMap::DynamicCloudPolicy::LONG;
    else if (map_policy == "persistent")
        opt.map_option_.policy_ = FullMap::DynamicCloudPolicy::PERSISTENT;
    opt.map_option_.load_dyn_cloud_ = Param<bool>(nh, "maps/load_dyn_cloud", true);
    opt.map_option_.save_dyn_when_quit_ = Param<bool>(nh, "maps/save_dyn_when_quit", true);

    lidar_loc::grid_search_angle_step = Param<int>(nh, "lidar_loc/grid_search_angle_step", 1);
    lidar_loc::grid_search_angle_range = Param<double>(nh, "lidar_loc/grid_search_angle_range", 10.0);
}

}  // namespace

SlamOptions LoadSlamOptions(const rclcpp::Node::SharedPtr &nh)
{
    SlamOptions opt;
    opt.lio_ctor.is_in_slam_mode_ = true;
    LoadLaserMappingConfig(nh, opt.lio);
    LoadOnlineLoop(nh, opt.loop);
    LoadPub(nh, opt.pub, true);
    opt.with_loop_closing = Param<bool>(nh, "system/with_loop_closing", false);
    opt.lidar_topic = Param<std::string>(nh, "common/lid_topic", std::string("/livox/lidar"));
    opt.imu_topic = Param<std::string>(nh, "common/imu_topic", std::string("/livox/imu"));
    opt.image_topics = LoadImageTopics(nh);
    opt.lidar_type = Param<int>(nh, "preprocess/lidar_type", static_cast<int>(LidarType::mid360));
    return opt;
}

LocOptions LoadLocOptions(const rclcpp::Node::SharedPtr &nh)
{
    LocOptions opt;
    opt.lio_ctor.is_in_slam_mode_ = false;
    LoadLaserMappingConfig(nh, opt.lio);
    const Eigen::Isometry3d T = opt.lio.T_body_lio;
    opt.lio_ctor.T_body_wrt_lio_ = T.translation();
    opt.lio_ctor.Q_body_wrt_lio_ = Eigen::Quaterniond(T.linear());
    LoadLidarLoc(nh, opt.lidar_loc);
    LoadPub(nh, opt.pub, false);

    opt.localization.online_mode_ = true;
    opt.localization.enable_lidar_loc_skip_ = Param<bool>(nh, "system/enable_lidar_loc_skip", false);
    opt.localization.lidar_loc_skip_num_ = Param<int>(nh, "system/lidar_loc_skip_num", 4);
    opt.localization.enable_lidar_odom_skip_ = Param<bool>(nh, "system/enable_lidar_odom_skip", false);
    opt.localization.lidar_odom_skip_num_ = Param<int>(nh, "system/lidar_odom_skip_num", 2);
    opt.localization.loc_on_kf_ = Param<bool>(nh, "lidar_loc/loc_on_kf", true);
    opt.localization.preprocess_blind_ = Param<double>(nh, "preprocess/blind", 0.01);
    opt.localization.preprocess_lidar_type_ = Param<int>(nh, "preprocess/lidar_type", static_cast<int>(LidarType::mid360));
    opt.localization.preprocess_point_filter_num_ = Param<int>(nh, "preprocess/point_filter_num", 1);
    opt.localization.preprocess_scan_line_ = Param<int>(nh, "preprocess/scan_line", 1);
    opt.localization.preprocess_height_max_ = Param<double>(nh, "preprocess/height_max", 0.01);
    opt.localization.preprocess_height_min_ = Param<double>(nh, "preprocess/height_min", 0.01);

    auto &sys = opt.system;
    sys.lidar_topic = Param<std::string>(nh, "common/lid_topic", std::string("/livox/lidar"));
    sys.imu_topic = Param<std::string>(nh, "common/imu_topic", std::string("/livox/imu"));
    sys.image_topics = LoadImageTopics(nh);
    sys.visual_enabled = Param<bool>(nh, "visual/enable", false);
    return opt;
}

}  // namespace lightning
