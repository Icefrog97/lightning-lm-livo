#pragma once

#include <livox_ros_driver/CustomPoint.h>
#include <std_msgs/Header.h>

#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>
#include <ros/serialization.h>
#include <ros/types.h>

#include <memory>
#include <ostream>
#include <vector>

namespace livox_ros_driver
{
template <class ContainerAllocator>
struct CustomMsg_
{
    using Type = CustomMsg_<ContainerAllocator>;

    CustomMsg_() { rsvd.assign(0); }
    explicit CustomMsg_(const ContainerAllocator& alloc) : header(alloc), points(alloc) { rsvd.assign(0); }

    std_msgs::Header_<ContainerAllocator> header;
    uint64_t timebase = 0;
    uint32_t point_num = 0;
    uint8_t lidar_id = 0;
    boost::array<uint8_t, 3> rsvd;
    std::vector<CustomPoint_<ContainerAllocator>,
                typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<CustomPoint_<ContainerAllocator>>>
        points;

    using Ptr = boost::shared_ptr<CustomMsg_<ContainerAllocator>>;
    using ConstPtr = boost::shared_ptr<const CustomMsg_<ContainerAllocator>>;
};

using CustomMsg = CustomMsg_<std::allocator<void>>;
using CustomMsgPtr = boost::shared_ptr<CustomMsg>;
using CustomMsgConstPtr = boost::shared_ptr<const CustomMsg>;

template <typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const CustomMsg_<ContainerAllocator>& v)
{
    ros::message_operations::Printer<CustomMsg_<ContainerAllocator>>::stream(s, "", v);
    return s;
}
} // namespace livox_ros_driver

namespace ros
{
namespace message_traits
{
template <class ContainerAllocator>
struct IsMessage<lightning::msg::CustomMsg_<ContainerAllocator>> : TrueType {};
template <class ContainerAllocator>
struct IsMessage<const lightning::msg::CustomMsg_<ContainerAllocator>> : TrueType {};
template <class ContainerAllocator>
struct IsFixedSize<lightning::msg::CustomMsg_<ContainerAllocator>> : FalseType {};
template <class ContainerAllocator>
struct IsFixedSize<const lightning::msg::CustomMsg_<ContainerAllocator>> : FalseType {};
template <class ContainerAllocator>
struct HasHeader<lightning::msg::CustomMsg_<ContainerAllocator>> : TrueType {};
template <class ContainerAllocator>
struct HasHeader<const lightning::msg::CustomMsg_<ContainerAllocator>> : TrueType {};

template <class ContainerAllocator>
struct MD5Sum<lightning::msg::CustomMsg_<ContainerAllocator>>
{
    static const char* value() { return "e4d6829bdfe657cb6c21a746c86b21a6"; }
    static const char* value(const lightning::msg::CustomMsg_<ContainerAllocator>&) { return value(); }
    static const uint64_t static_value1 = 0xe4d6829bdfe657cbULL;
    static const uint64_t static_value2 = 0x6c21a746c86b21a6ULL;
};

template <class ContainerAllocator>
struct DataType<lightning::msg::CustomMsg_<ContainerAllocator>>
{
    static const char* value() { return "livox_ros_driver/CustomMsg"; }
    static const char* value(const lightning::msg::CustomMsg_<ContainerAllocator>&) { return value(); }
};

template <class ContainerAllocator>
struct Definition<lightning::msg::CustomMsg_<ContainerAllocator>>
{
    static const char* value()
    {
        return "# Livox publish pointcloud msg format.\n"
               "\n"
               "Header header             # ROS standard message header\n"
               "uint64 timebase           # The time of first point\n"
               "uint32 point_num          # Total number of pointclouds\n"
               "uint8  lidar_id           # Lidar device id number\n"
               "uint8[3]  rsvd            # Reserved use\n"
               "CustomPoint[] points      # Pointcloud data\n"
               "\n";
    }
    static const char* value(const lightning::msg::CustomMsg_<ContainerAllocator>&) { return value(); }
};
} // namespace message_traits

namespace serialization
{
template <class ContainerAllocator>
struct Serializer<lightning::msg::CustomMsg_<ContainerAllocator>>
{
    template <typename Stream, typename T>
    inline static void allInOne(Stream& stream, T m)
    {
        stream.next(m.header);
        stream.next(m.timebase);
        stream.next(m.point_num);
        stream.next(m.lidar_id);
        stream.next(m.rsvd);
        stream.next(m.points);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
};
} // namespace serialization

namespace message_operations
{
template <class ContainerAllocator>
struct Printer<lightning::msg::CustomMsg_<ContainerAllocator>>
{
    template <typename Stream>
    static void stream(Stream& s, const std::string& indent, const lightning::msg::CustomMsg_<ContainerAllocator>& v)
    {
        s << indent << "header:\n";
        Printer<std_msgs::Header_<ContainerAllocator>>::stream(s, indent + "  ", v.header);
        s << indent << "timebase: " << v.timebase << "\n";
        s << indent << "point_num: " << v.point_num << "\n";
        s << indent << "lidar_id: " << static_cast<int>(v.lidar_id) << "\n";
        s << indent << "points[] size: " << v.points.size() << "\n";
    }
};
} // namespace message_operations
} // namespace ros
