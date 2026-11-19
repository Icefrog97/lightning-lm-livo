#pragma once

#include <cmath>
#include <string>

#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/rclcpp.hpp>

namespace lightning {

inline std::string RosParamName(std::string key) {
    for (char &ch : key) {
        if (ch == '/')
            ch = '.';
    }
    return key;
}

inline builtin_interfaces::msg::Time StampFromSec(double t) {
    builtin_interfaces::msg::Time stamp;
    if (!std::isfinite(t) || t < 0.0)
        t = 0.0;
    const int64_t ns = static_cast<int64_t>(std::llround(t * 1e9));
    stamp.sec = static_cast<int32_t>(ns / 1000000000LL);
    stamp.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
    return stamp;
}

inline double SecFromStamp(const builtin_interfaces::msg::Time &stamp) {
    return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

inline rclcpp::QoS CloudQos(size_t depth = 5) {
    return rclcpp::SensorDataQoS().keep_last(depth);
}

inline rclcpp::QoS ReliableQos(size_t depth) {
    return rclcpp::QoS(rclcpp::KeepLast(depth));
}

inline rclcpp::QoS MapQos() {
    return rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
}

}  // namespace lightning
