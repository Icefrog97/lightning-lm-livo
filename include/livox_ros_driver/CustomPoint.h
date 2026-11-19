#pragma once

#include <boost/shared_ptr.hpp>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>
#include <ros/serialization.h>
#include <ros/types.h>

#include <cstdint>
#include <memory>
#include <ostream>

namespace livox_ros_driver
{
template <class ContainerAllocator>
struct CustomPoint_
{
    using Type = CustomPoint_<ContainerAllocator>;

    CustomPoint_() = default;
    explicit CustomPoint_(const ContainerAllocator& alloc) { (void)alloc; }

    uint32_t offset_time = 0;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    uint8_t reflectivity = 0;
    uint8_t tag = 0;
    uint8_t line = 0;

    using Ptr = boost::shared_ptr<CustomPoint_<ContainerAllocator>>;
    using ConstPtr = boost::shared_ptr<const CustomPoint_<ContainerAllocator>>;
};

using CustomPoint = CustomPoint_<std::allocator<void>>;
using CustomPointPtr = boost::shared_ptr<CustomPoint>;
using CustomPointConstPtr = boost::shared_ptr<const CustomPoint>;

template <typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const CustomPoint_<ContainerAllocator>& v)
{
    ros::message_operations::Printer<CustomPoint_<ContainerAllocator>>::stream(s, "", v);
    return s;
}
} // namespace livox_ros_driver

namespace ros
{
namespace message_traits
{
template <class ContainerAllocator>
struct IsMessage<livox_ros_driver::CustomPoint_<ContainerAllocator>> : TrueType {};
template <class ContainerAllocator>
struct IsMessage<const livox_ros_driver::CustomPoint_<ContainerAllocator>> : TrueType {};
template <class ContainerAllocator>
struct IsFixedSize<livox_ros_driver::CustomPoint_<ContainerAllocator>> : TrueType {};
template <class ContainerAllocator>
struct IsFixedSize<const livox_ros_driver::CustomPoint_<ContainerAllocator>> : TrueType {};
template <class ContainerAllocator>
struct HasHeader<livox_ros_driver::CustomPoint_<ContainerAllocator>> : FalseType {};
template <class ContainerAllocator>
struct HasHeader<const livox_ros_driver::CustomPoint_<ContainerAllocator>> : FalseType {};

template <class ContainerAllocator>
struct MD5Sum<livox_ros_driver::CustomPoint_<ContainerAllocator>>
{
    static const char* value() { return "109a3cc548bb1f96626be89a5008bd6d"; }
    static const char* value(const livox_ros_driver::CustomPoint_<ContainerAllocator>&) { return value(); }
    static const uint64_t static_value1 = 0x109a3cc548bb1f96ULL;
    static const uint64_t static_value2 = 0x626be89a5008bd6dULL;
};

template <class ContainerAllocator>
struct DataType<livox_ros_driver::CustomPoint_<ContainerAllocator>>
{
    static const char* value() { return "livox_ros_driver/CustomPoint"; }
    static const char* value(const livox_ros_driver::CustomPoint_<ContainerAllocator>&) { return value(); }
};

template <class ContainerAllocator>
struct Definition<livox_ros_driver::CustomPoint_<ContainerAllocator>>
{
    static const char* value()
    {
        return "# Livox costom pointcloud format.\n"
               "\n"
               "uint32 offset_time      # offset time relative to the base time\n"
               "float32 x               # X axis, unit:m\n"
               "float32 y               # Y axis, unit:m\n"
               "float32 z               # Z axis, unit:m\n"
               "uint8 reflectivity      # reflectivity, 0~255\n"
               "uint8 tag               # livox tag\n"
               "uint8 line              # laser number in lidar\n"
               "\n";
    }
    static const char* value(const livox_ros_driver::CustomPoint_<ContainerAllocator>&) { return value(); }
};
} // namespace message_traits

namespace serialization
{
template <class ContainerAllocator>
struct Serializer<livox_ros_driver::CustomPoint_<ContainerAllocator>>
{
    template <typename Stream, typename T>
    inline static void allInOne(Stream& stream, T m)
    {
        stream.next(m.offset_time);
        stream.next(m.x);
        stream.next(m.y);
        stream.next(m.z);
        stream.next(m.reflectivity);
        stream.next(m.tag);
        stream.next(m.line);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
};
} // namespace serialization

namespace message_operations
{
template <class ContainerAllocator>
struct Printer<livox_ros_driver::CustomPoint_<ContainerAllocator>>
{
    template <typename Stream>
    static void stream(Stream& s, const std::string& indent, const livox_ros_driver::CustomPoint_<ContainerAllocator>& v)
    {
        s << indent << "offset_time: " << v.offset_time << "\n";
        s << indent << "x: " << v.x << "\n";
        s << indent << "y: " << v.y << "\n";
        s << indent << "z: " << v.z << "\n";
        s << indent << "reflectivity: " << static_cast<int>(v.reflectivity) << "\n";
        s << indent << "tag: " << static_cast<int>(v.tag) << "\n";
        s << indent << "line: " << static_cast<int>(v.line) << "\n";
    }
};
} // namespace message_operations
} // namespace ros
