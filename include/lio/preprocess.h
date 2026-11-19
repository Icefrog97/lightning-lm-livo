#include "ros/ros2_types.h"
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "common/point_def.h"
#include "common/eigen_types.h"

namespace lightning
{

    enum LidarType
    {
        ROBOSENSE = 1,
        VELO16 = 2,
        OUST64 = 3,
        mid360 = 4,
        LIVOX_CUSTOM = 5
    }; //{1, 2, 3}

    namespace mid360_ros
    {
        struct EIGEN_ALIGN16 Point
        {
            float x;           /**< X axis, Unit:m */
            float y;           /**< Y axis, Unit:m */
            float z;           /**< Z axis, Unit:m */
            float intensity;   /**< Reflectivity   */
            std::uint8_t tag;  /**< Livox point tag   */
            std::uint8_t line; /**< Laser line id     */
            double timestamp;
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        };
    }

    struct PointRobotSense {
        PCL_ADD_POINT4D
        PCL_ADD_INTENSITY
        std::uint16_t ring = 0;
        double timestamp = 0;
        PointRobotSense() {}
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    namespace velodyne_ros
    {
        struct EIGEN_ALIGN16 Point
        {
            PCL_ADD_POINT4D;
            float intensity;
            float time;
            std::uint16_t ring;
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        };
    } // namespace velodyne_ros

    namespace ouster_ros
    {
        struct EIGEN_ALIGN16 Point
        {
            PCL_ADD_POINT4D;
            float intensity;
            std::uint32_t t;
            std::uint16_t reflectivity;
            std::uint8_t ring;
            std::uint16_t ambient;
            std::uint32_t range;
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        };
    } // namespace ouster_ros

    class PointCloudPreprocess
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        void Process(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg, PointCloudType::Ptr &pcl_out);
        void Process(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg, PointCloudType::Ptr &pcl_out);
        void SetLidarType(LidarType lt) { lidar_type_ = lt; }

        void SetHeightROI(double height_max, double height_min)
        {
            height_max_ = height_max;
            height_min_ = height_min;
        }

        // sensor_msgs::msg::PointCloud2::ConstSharedPtr pointcloud;
        PointCloudType pl_full_, pl_surf_;

        int lidar_type_, point_filter_num_, N_SCANS_;
        double blind_;
        bool given_offset_time_;

        double height_max_;
        double height_min_;


    private:
        void Mid360Handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
        void LivoxCustomHandler(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg);
        void Oust64Handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
        void VeclodyneHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
        void RoboSenseHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
    };

} // namespace

POINT_CLOUD_REGISTER_POINT_STRUCT(lightning::mid360_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint8_t, tag, tag)(std::uint8_t, line, line)(double, timestamp, timestamp))

POINT_CLOUD_REGISTER_POINT_STRUCT(lightning::PointRobotSense,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                                                          intensity)(std::uint16_t, ring, ring)(double, timestamp, timestamp))

POINT_CLOUD_REGISTER_POINT_STRUCT(lightning::velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, time, time)(std::uint16_t, ring, ring))

POINT_CLOUD_REGISTER_POINT_STRUCT(lightning::ouster_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint32_t, t, t)(std::uint16_t, reflectivity, reflectivity)(std::uint8_t, ring, ring)(std::uint16_t, ambient, ambient)(std::uint32_t, range, range))