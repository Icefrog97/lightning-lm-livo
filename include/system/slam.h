#ifndef LIGHTNING_SLAM_H
#define LIGHTNING_SLAM_H

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <string>
#include <vector>
#include "common/eigen_types.h"
#include "common/keyframe.h"
#include "common/timer.h"
#include "ros/param_loader.h"

#include <thread>
#include <mutex>
namespace lightning
{

    class LaserMapping; //  lio 前端
    class OnlineLoop;   // 在线 STD 回环
    class PubROSTopic;

    /**
     * SLAM 系统调用接口
     */
    class SlamSystem
    {
    public:
        struct Options
        {
            Options() {}
            bool online_mode_ = true; // 在线模式，在线模式下会起一些子线程来做异步处理

            bool with_cc_ = false;           // 是否需要带交叉验证
            bool with_loop_closing_ = false; // 是否需要回环检测
        };

        SlamSystem(Options options);

        /// 初始化。opt 由 LoadSlamOptions 填好。
        bool Init(const rclcpp::Node::SharedPtr &nh, const SlamOptions &opt);

        /// 对外部交互接口
        /// 开始建图，输入地图名称
        void StartSLAM();

        bool VisualEnabled() const;
        bool ProcessOnce();

        /// 保存地图，默认保存至./data/地图名/ 下方
        void SaveMap();

        /// 在线回环线程排空，并把优化后的点云写成 PCD。未开启时无效果。
        void FinishLoop();

        void ImuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &msg);

        void LidarCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
        void LivoxLidarCallback(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg);

        void ImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg, int cam_id);

    private:
        Options options_;
        std::atomic_bool running_{false};

        std::shared_ptr<LaserMapping> lio_ = nullptr; // lio 前端
        std::shared_ptr<OnlineLoop> lc_ = nullptr;    // 在线回环，with_loop_closing 才创建

        Keyframe::Ptr cur_kf_ = nullptr;

        /// node, subscribers
        std::string imu_topic_;
        std::string lidar_topic_;
        std::vector<std::string> image_topics_;

        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
        rclcpp::SubscriptionBase::SharedPtr lidar_sub_;
        std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> image_subs_;

        std::shared_ptr<PubROSTopic> pub_ = nullptr;

    private:
        int lidar_type_;
    };
} // namespace lightning

#endif // LIGHTNING_SLAM_H
