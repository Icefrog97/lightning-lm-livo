import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def deep_merge(base, extra):
    merged = dict(base)
    for key, value in extra.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def ros_param_value(value):
    # Humble stores an empty sequence with no element type, then refuses to load it.
    if isinstance(value, dict):
        return {
            key: ros_param_value(item)
            for key, item in value.items()
            if not (isinstance(item, (list, tuple)) and len(item) == 0)
        }
    return value


def package_root(share):
    # ROS1 $(find lidar_slam) was the source package, where MAP/PCD/Log live.
    source = os.path.abspath(os.path.join(share, "..", "..", "..", "..", "src", "lightning-lm2"))
    if os.path.isfile(os.path.join(source, "package.xml")):
        return source
    return share


def generate_launch_description():
    share = get_package_share_directory("lightning")
    root = package_root(share)
    params = {}
    for name in ("config_mid360.yaml", "camera_fisheye.yaml"):
        params = deep_merge(params, load_yaml(os.path.join(share, "config", name)))
    bbs = load_yaml(os.path.join(share, "config", "3dbbs.yaml"))
    params["lidar_loc"] = deep_merge(params.get("lidar_loc", {}), bbs)

    params.setdefault("system", {})
    params["system"]["map_path"] = os.path.join(root, "MAP") + "/"
    params["system"]["log_path"] = os.path.join(root, "Log") + "/"
    params["system"]["enable_debug_log"] = False

    params.setdefault("dynamic_filter", {})
    params["dynamic_filter"]["model_path"] = os.path.join(root, "model", "bestc++.engine")

    params.setdefault("common", {})
    params["common"]["imu_topic"] = "/livox/imu_192_168_1_130"
    params["common"]["lid_topic"] = "/livox/lidar_192_168_1_130"

    params.setdefault("publish", {})
    params["publish"]["pub_path_num"] = 100

    params["use_sim_time"] = False

    rviz = LaunchConfiguration("rviz")
    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(
            package="lightning",
            executable="run_localization",
            name="run_localization",
            output="screen",
            parameters=[ros_param_value(params)],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz_lio",
            arguments=["-d", os.path.join(share, "rviz_cfg", "localization.rviz")],
            condition=IfCondition(rviz),
        ),
    ])
