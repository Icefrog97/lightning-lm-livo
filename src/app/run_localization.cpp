#include <glog/logging.h>
#include <opencv2/core.hpp>
#include "system/loc_system.h"

int main(int argc, char **argv)
{
    cv::setNumThreads(1);

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.allow_undeclared_parameters(true);
    node_options.automatically_declare_parameters_from_overrides(true);
    auto node = std::make_shared<rclcpp::Node>("run_localization", node_options);
    std::string log_path;
    int log_level = 1;
    if (node->has_parameter("system.log_path"))
        log_path = node->get_parameter("system.log_path").as_string();
    if (node->has_parameter("system.log_level"))
        log_level = static_cast<int>(node->get_parameter("system.log_level").as_int());

    google::InitGoogleLogging(argv[0]);
    google::SetLogFilenameExtension("_localization.log");
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = log_level;
    FLAGS_alsologtostderr = false;
    FLAGS_log_prefix = false;
    FLAGS_logbufsecs = 0;
    google::SetLogDestination(google::GLOG_INFO, log_path.c_str());
    google::SetLogDestination(google::GLOG_WARNING, ""); 
    google::SetLogDestination(google::GLOG_ERROR, "");  

    lightning::LocSystem::Options opt;
    lightning::LocSystem loc(opt);
    const lightning::LocOptions loc_opt = lightning::LoadLocOptions(node);

    if (!loc.Init(node, loc_opt))
        LOG(ERROR) << lightning::get_time_us() << "[ERROR][run_localization::" << __FUNCTION__ << "] " << "failed to init loc";

    loc.SetInitPose(lightning::SE3());
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}