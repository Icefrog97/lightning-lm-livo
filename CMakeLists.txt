cmake_minimum_required(VERSION 3.5)
project(lightning)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -fexceptions")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")

# Executables live in lib/lightning, shared libraries in lib/.
# Also record OpenCV 4.2 and /usr/local so launch does not need LD_LIBRARY_PATH.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(APPEND CMAKE_INSTALL_RPATH
    "$ORIGIN"
    "$ORIGIN/.."
    "$ENV{HOME}/deps/opencv-4.2.0-install/lib"
    "/usr/local/lib"
)
# DT_RPATH is searched for dependencies of libgtsam, such as libmetis-gtsam.
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")

set(OpenCV_DIR "$ENV{HOME}/deps/opencv-4.2.0-install/lib/cmake/opencv4" CACHE PATH "" FORCE)
find_package(OpenCV 4.2.0 REQUIRED EXACT)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)

set(OpenCV_DIR "$ENV{HOME}/deps/opencv-4.2.0-install/lib/cmake/opencv4" CACHE PATH "" FORCE)

find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(GTSAM REQUIRED QUIET)
find_package(OpenMP QUIET)
find_package(Ceres REQUIRED)
find_package(TBB REQUIRED)

add_subdirectory(src/vikit_common)

# Optional CUDA for GPU 3D-BBS. OFF or missing toolkit → CPU 3D-BBS only; package still builds.
option(WITH_CUDA "Build GPU 3D-BBS when CUDA is available" ON)
set(LIDAR_SLAM_HAS_CUDA OFF)
if (WITH_CUDA)
    find_package(CUDA QUIET)
    if (CUDA_FOUND)
        set(LIDAR_SLAM_HAS_CUDA ON)
        message(STATUS "CUDA: ENABLED for 3D-BBS GPU (${CUDA_VERSION})")
    else()
        message(WARNING "WITH_CUDA=ON but CUDA not found; 3D-BBS uses CPU only")
    endif()
else()
    message(STATUS "CUDA: DISABLED (WITH_CUDA=OFF); 3D-BBS uses CPU only")
endif()

# Optional TensorRT for YOLO dynamic filter (from IQIS). OFF → stub detector, package still builds.
option(BUILD_DYNAMIC_FILTER_TRT "Build YOLO TensorRT dynamic filter" ON)
# PC (default): TensorRT 10 IO-tensor API in yolo_trt_detector.cpp
# Jetson: set ON to compile yolo_trt_detector_jetson.cpp (TRT 8 Binding API)
option(YOLO_TRT_JETSON "Use Jetson TensorRT 8.x Binding API instead of PC TensorRT 10" OFF)
set(LIDAR_SLAM_HAS_TENSORRT OFF)
if (BUILD_DYNAMIC_FILTER_TRT)
    find_package(CUDA QUIET)
    find_library(NVINFER_LIB NAMES nvinfer)
    find_library(NVONNXPARSER_LIB NAMES nvonnxparser)
    if (CUDA_FOUND AND NVINFER_LIB AND NVONNXPARSER_LIB)
        set(LIDAR_SLAM_HAS_TENSORRT ON)
        if (YOLO_TRT_JETSON)
            message(STATUS "DynamicFilter TensorRT: ENABLED (Jetson TRT8 Binding API)")
        else()
            message(STATUS "DynamicFilter TensorRT: ENABLED (PC TRT10 IO-tensor API)")
        endif()
    else()
        message(WARNING "DynamicFilter TensorRT requested but nvinfer/nvonnxparser/CUDA not found; building stub")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

include_directories(BEFORE SYSTEM ${OpenCV_INCLUDE_DIRS})

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIRS}
    ${glog_INCLUDE_DIRS}
    include
    src/bbs3d/include
    src/vikit_common/include
)

add_definitions("-DPCL_NO_PRECOMPILE")

add_subdirectory(3rdparty/livox_ros_driver)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/3rdparty/livox_ros_driver/rosidl_generator_cpp)

if (YOLO_TRT_JETSON)
    set(YOLO_TRT_SRC src/perception/yolo_trt_detector_jetson.cpp)
else()
    set(YOLO_TRT_SRC src/perception/yolo_trt_detector.cpp)
endif()

add_library(${PROJECT_NAME}_core SHARED
    src/LIO/eskf.cpp
    src/LIO/laserMapping.cpp
    src/LIO/voxel_map.cpp
    src/LIO/preprocess.cpp

    src/pub/pub_ROS1.cpp

    src/maps/full_map.cpp
    src/maps/tiled_map_chunk.cpp

    src/localization/pose_graph/pgo.cc
    src/localization/pose_graph/pose_extrapolator.cc
    src/localization/pose_graph/pgo_impl.cc

    src/bbs3d/src/run_cpu.cpp
    src/bbs3d/src/bbs3d_runner.cpp

    src/localization/lidar_loc.cpp
    src/localization/localization_result.cpp
    src/localization/localization.cpp

    src/system/slam.cpp
    src/system/loc_system.cpp
    src/ros/param_loader.cpp

    src/loopclosing/online_loop.cpp
    include/STDesc/STDesc.cpp

    src/vio/frame.cpp
    src/vio/vio.cpp
    src/vio/visual_point.cpp

    ${YOLO_TRT_SRC}
    src/perception/dynamic_filter.cpp

    include/common/options.cpp
    include/common/nav_state.cpp
    include/localization/pclomp/ndt_omp.cpp
    include/localization/pclomp/voxel_grid_covariance_omp.cpp
    include/utils/timer.cc
    
)

target_link_libraries(${PROJECT_NAME}_core PUBLIC
    livox_interfaces2__rosidl_typesupport_cpp
)
ament_target_dependencies(${PROJECT_NAME}_core PUBLIC
    rclcpp std_msgs geometry_msgs nav_msgs sensor_msgs visualization_msgs
    cv_bridge pcl_conversions tf2 tf2_ros tf2_eigen
)

include_directories(${PROJECT_SOURCE_DIR}/src/miao)
add_subdirectory(src/miao)

add_subdirectory(src/bbs3d)

target_link_libraries(${PROJECT_NAME}_core
    PUBLIC
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    glog::glog
    vikit_plugin
    miao.core
    miao.utils
    cpu_bbs3d
    gtsam
    ${CERES_LIBRARIES}
)

if (LIDAR_SLAM_HAS_CUDA)
    target_compile_definitions(${PROJECT_NAME}_core PUBLIC LIDAR_SLAM_USE_CUDA=1)
    target_include_directories(${PROJECT_NAME}_core PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME}_core PUBLIC
        gpu_bbs3d
        ${CUDA_LIBRARIES}
    )
endif()

if (LIDAR_SLAM_HAS_TENSORRT)
    cuda_add_library(yolo_letterbox SHARED src/perception/yolo_letterbox.cu)
    target_include_directories(yolo_letterbox PUBLIC include ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME}_core PUBLIC LIDAR_SLAM_USE_TENSORRT=1)
    target_include_directories(${PROJECT_NAME}_core PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME}_core PUBLIC
        yolo_letterbox
        ${CUDA_LIBRARIES}
        ${NVINFER_LIB}
        ${NVONNXPARSER_LIB}
    )
endif()

add_executable(run_mapping src/app/run_mapping.cpp)
target_link_libraries(run_mapping
    ${PROJECT_NAME}_core
    ${OpenCV_LIBRARIES}
)
ament_target_dependencies(run_mapping rclcpp)

add_executable(run_localization src/app/run_localization.cpp)
target_link_libraries(run_localization
    ${PROJECT_NAME}_core
    ${OpenCV_LIBRARIES}
)
ament_target_dependencies(run_localization rclcpp)

set(LIGHTNING_INSTALL_TARGETS ${PROJECT_NAME}_core run_mapping run_localization
    miao.core miao.utils cpu_bbs3d vikit_plugin)
if (LIDAR_SLAM_HAS_CUDA)
    list(APPEND LIGHTNING_INSTALL_TARGETS gpu_bbs3d)
endif()
install(TARGETS ${LIGHTNING_INSTALL_TARGETS}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config rviz_cfg model DESTINATION share/${PROJECT_NAME})
ament_export_dependencies(rclcpp)
ament_package()

