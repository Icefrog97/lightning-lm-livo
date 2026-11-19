#include "system/slam.h"
#include <glog/logging.h>
#include <opencv2/core.hpp>

int main(int argc, char **argv)
{
    cv::setNumThreads(1);

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.allow_undeclared_parameters(true);
    node_options.automatically_declare_parameters_from_overrides(true);
    auto node = std::make_shared<rclcpp::Node>("run_mapping", node_options);
    std::string log_path;
    int log_level = 1;
    if (node->has_parameter("system.log_path"))
        log_path = node->get_parameter("system.log_path").as_string();
    if (node->has_parameter("system.log_level"))
        log_level = static_cast<int>(node->get_parameter("system.log_level").as_int());

    google::InitGoogleLogging(argv[0]);
    google::SetLogFilenameExtension("_mapping.log");
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = log_level;
    FLAGS_alsologtostderr = true;
    FLAGS_log_prefix = false;
    google::SetLogDestination(google::GLOG_INFO, log_path.c_str());
    google::SetLogDestination(google::GLOG_WARNING, "");
    google::SetLogDestination(google::GLOG_ERROR, ""); 

    LOG(INFO) << lightning::get_time_us() << "[INFO][run_mapping::" << __FUNCTION__ << "] " << "Starting LiDAR SLAM...";

    lightning::SlamSystem::Options options;
    options.online_mode_ = true;
    lightning::SlamSystem slam(options);
    const lightning::SlamOptions slam_opt = lightning::LoadSlamOptions(node);

    if (!slam.Init(node, slam_opt))
    {
        LOG(ERROR) << lightning::get_time_us() << "[ERROR][run_mapping::" << __FUNCTION__ << "] " << "failed to init slam";
        return -1;
    }

    slam.StartSLAM();
    if (slam.VisualEnabled())
    {
        // FAST-LIVO2 scheduling: spinOnce, then one LIVO step (LIO or VIO, not both).
        rclcpp::executors::SingleThreadedExecutor exec;
        exec.add_node(node);
        rclcpp::Rate rate(5000);
        while (rclcpp::ok())
        {
            exec.spin_some();
            slam.ProcessOnce();
            rate.sleep();
        }
    }
    else
    {
        // Preserve the existing pure-LIO callback-driven path.
        rclcpp::spin(node);
    }

    slam.FinishLoop();
    slam.SaveMap();

    rclcpp::shutdown();

    LOG(INFO) << lightning::get_time_us() << "[INFO][run_mapping::" << __FUNCTION__ << "] " << "slam mapping down";

    return 0;
}