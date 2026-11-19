## 使用说明：

1. 将 MAP/ 下的 example_* 复制并重命名为 *。
2. 安装依赖

本仓库在 [lightning-lm](https://github.com/gaoxiang12/lightning-lm) 的激光建图与定位上继续开发。前端仍是 FasterLIO / ESKF，定位仍是激光匹配加 PGO。

用到的项目：

- [lightning-lm](https://github.com/gaoxiang12/lightning-lm)：激光建图、定位和 iVox 前端
- [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2)：VoxelMap 和视觉里程计
- [STD](https://github.com/hku-mars/STD)：回环用的三角形描述子
- [3D-BBS](https://github.com/KOKIAOKI/3d_bbs)：定位全局初始化
- [GTSAM](https://github.com/borglab/gtsam)：回环位姿图，用的是 ISAM2
- [rpg_vikit](https://github.com/uzh-rpg/rpg_vikit)：相机模型，源码在 `src/vikit_common/`
- [Ultralytics](https://github.com/ultralytics/ultralytics)：动态物检测模型的训练
- [Ceres](https://github.com/ceres-solver/ceres-solver)

### 建图前端可以三选一

lightning-lm 的纯激光地图是 iVox。这里另外接了 FAST-LIVO2 的 VoxelMap。VoxelMap 不等于必须开相机。

1. 纯激光，用原来的 iVox

   `config/config_mid360.yaml` 里：

   - `visual/enable: false`
   - `lio/use_voxel_map: false`

   当前帧降采样用 `common/filter_size_surf`。往地图里加点的体素门限用 `common/filter_size_map`。

2. 纯激光，用 VoxelMap，不用 iVox

   - `visual/enable: false`
   - `lio/use_voxel_map: true`

   仍然只吃雷达和 IMU，不跑视觉更新。地图改成按体素存平面，参数在 `lio:` 下面，包括 `voxel_size`、`max_layer`、`max_points_num`、`lru_capacity`。`lru_capacity` 是根体素个数上限，写 `0` 表示不淘汰。当前帧降采样改用 `common/livo_filter_size_surf`，默认 0.4，对应 FAST-LIVO2 的 avia 配置。这一档用来单独看 VoxelMap 的激光里程计，不把视觉误差混进来。

3. 激光加视觉（LIVO）

   - `visual/enable: true`

   打开视觉后会强制 `use_voxel_map: true`，不能再退回 iVox。需要图像话题 `common/image_topics`，以及 `camera`、`vio` 里的内外参。相机模型编在 `src/vikit_common/`，目标名 `vikit_plugin`，不再单独作为 ROS 包，也不依赖 `vikit_ros`。视觉点的缓存大小由 `visual/lru_capacity` 和 `visual/max_points_per_voxel` 控制，写 `0` 表示不限制。每个视觉点最多保留 15 次观测。法向更新足够小且观测超过 7 次就标成收敛，不再继续堆观测。

### 动态物剔除

`dynamic_filter/enable: true` 时，点云去畸变之后、匹配之前，用图像检测结果把人、车等框掉。需要图像话题和相机内外参，可以和上面第 2 种纯激光 VoxelMap 一起开，不必打开 `visual/enable`。有 TensorRT 时走 YOLO；没有 TensorRT 时编一个空的检测器，纯激光照常编译、运行。被删掉的点可以发到 `/dynamic_filter/removed` 和 `/dynamic_filter/removed_body`。

仓库里的 `model/` 可以直接用，但不建议在自己的场景里完全沿用。需要换场景时，自己采集图像，用 [Ultralytics](https://github.com/ultralytics/ultralytics) 训练 YOLO，再导出成 TensorRT engine，把路径填到 launch 的模型参数里。有 CUDA 和 TensorRT 时，前处理在 GPU 上做 letterbox，推理用 CUDA Graph 录一次、之后每帧重放。Jetson 走另一份 TensorRT 8 的实现，由 `YOLO_TRT_JETSON` 选择。

### 回环

lightning-lm 按距离找历史关键帧，用 NDT 验证，再用 miao 做位姿图。这里换成 [STD](https://github.com/hku-mars/STD) 加 [GTSAM](https://github.com/borglab/gtsam) ISAM2。建图时 LIO 只把关键帧送进 `OnlineLoop`，描述子和图优化在单独线程里跑。每 `sub_frame_num` 帧点云合成一个子图，提三角形描述子再检索；跳过最近 `skip_near_num` 个子图，避免和刚走过的帧配上。检索到候选后用平面几何 ICP 修正相对位姿，再加 Between 因子。参数在 `config/config_loop.yaml`。`system/with_loop_closing` 为 true 时和建图一起开，优化后的点云写到源码包的 `PCD/`，同时写出 3D-BBS 用的体素地图。

新关键帧相对上一帧的里程计边用上一帧优化后的位姿，不用优化前的 LIO 位姿。结束时打印每条回环边的平移残差（厘米）、旋转残差（度）和分数，以及均值、最大值。`use_height_factor` 是高度因子，默认打开：每来一个关键帧就加一条一元约束，把该帧的 z 拉向第一帧的高度，标准差 0.1 m，用来压住单层里的高度漂移。多楼层场景把它设为 false。

### 定位与 3D-BBS

没有初值时用 [3D-BBS](https://github.com/KOKIAOKI/3d_bbs) 做全局搜索。`config/3dbbs.yaml` 的 `bbs_backend`：`auto` 是编进了 CUDA 并且机器上有显卡就用 GPU，否则用 CPU；`gpu` 是要求 GPU，没有设备时退回 CPU；`cpu` 是始终用 CPU。CPU 和 GPU 使用同一段高度范围 `bbs_min_z`、`bbs_max_z`，搜索时丢掉与这段高度不相交的分支。CMake 选项 `WITH_CUDA` 默认打开；找不到 CUDA 时整个工程仍然能编过，只是 3D-BBS 只有 CPU。定位时若不要 3D-BBS，把 `lidar_loc/use_3d_bbs` 设为 `false`。

`lidar_loc/bbs_timeout_msec` 大于 0 时，搜索到时间就停。0 表示不限时。定位可以读回环存下来的体素图；没有这份文件时，用当前加载的点云现算。

进定位匹配前，当前帧先按 0.2 m 做体素降采样，再送进 NDT。初始化时先攒 3 帧拼成一帧，同样按 0.2 m 降采样，这个尺寸按 Mid360 取的。功能点或 3D-BBS 第一次没对上时，粗 NDT 的分辨率改成 2.0 m，再试一轮。`min_init_confidence` 不再减半，避免走廊里得分偏低时误初始化。连续匹配失败会计数，次数够了把状态打成 FAIL，再重置 LIO。`lidar_loc/add_recover_to_index` 为 true 时，启动把 `MAP` 里的恢复位姿加进功能点，当前配置是 false。动态图层更新前对点云做 z 向直通，下限 0.5 m，上限 `lidar_loc/filter_z_max`。

### 雷达、iVox 与调平

`preprocess/lidar_type`：1 RoboSense，2 Velodyne，3 Ouster，4 Mid360 的 `PointCloud2`，5 Livox `CustomMsg`。建图和定位用同一个值。

iVox 每个体素最多留 10 个点，满了删最旧的，避免小范围来回走时一个格子里的点越堆越多。格子总数上限是 `fasterlio/ivox_capacity`，近邻类型是 `fasterlio/ivox_nearby_type`（0、6、18、26）。IMU 初始化帧数是 `fasterlio/max_init_count`，默认 10。定位发散后会重置 LIO。

退化检测只看平移信息矩阵 `H_tt` 的特征值，旋转块不抑制。开关是 `fasterlio/use_degeneracy`，比例是 `fasterlio/degeneracy_threshold_ratio`。这种做法只按特征值比例压平移，旋转和平移还耦合在一起。更精确的退化模型推荐 [DCReg](https://github.com/JokerJohn/DCReg)：用 Schur 补把旋转和平移的可观性拆开，再只对弱方向做处理。

雷达倾斜安装时用 `common/T_body_wrt_lio` 和 `common/E_body_wrt_lio` 调平。前者是平移，后者是 yaw、pitch、roll，单位是度，表示后轴中心相对 LIO/IMU 的位姿。发布的轨迹、点云和回环位姿都左乘它的逆。水平地面上可以用 `scripts/calc_imu_tilt.py` 估 pitch 和 roll。

## 编译

### 环境

Ubuntu 22.04，ROS 2 Humble。

### 依赖

- PCL
- Eigen
- Ceres
- yaml-cpp
- glog
- geographiclib
- opencv
- gflags
- pcl_conversions
- OpenMP
- CUDA（可选。没有 CUDA 时 3D-BBS 用 CPU，动态物剔除不走 TensorRT）

sudo apt-get install libpcl-dev pcl-tools libyaml-cpp-dev ros-humble-pcl-conversions libfmt-dev

编译 Ceres

​    cd 3rdparty/ceres

​    mkdir build
​    cd build

​    cmake ..

​    make -j6

​    sudo make install

编译 geographiclib

​    cd 3rdparty/geographiclib

​    mkdir build

​    cd build

​    cmake ..

​    make -j6

​    sudo make install

编译 gtsam

​    cd 3rdparty/gtsam-4.2.0

​    mkdir build

​    cd build

​    cmake ..

​    make -j6

​    sudo make install

GTSAM 要对着系统里的 Eigen 编译。Ubuntu 22.04 的系统 Eigen 是 3.4。`3rdparty/gtsam-4.2.0/cmake/HandleEigen.cmake` 里 `GTSAM_USE_SYSTEM_EIGEN` 默认已经是 ON。如果以前用别的 Eigen 装过 GTSAM，先把 `build` 目录删掉再 cmake，否则会沿用旧的缓存。

编译 opencv

工程不用系统自带的 OpenCV。系统里一般是 4.5，这个工程的 CMakeLists 指定要 4.2.0，并且只到下面这个目录里找：

​    $HOME/deps/opencv-4.2.0-install

`$HOME` 是当前用户的主目录。用户名是 `abc` 时，这个路径就是 `/home/abc/deps/opencv-4.2.0-install`。下面的 `CMAKE_INSTALL_PREFIX` 就是在告诉 OpenCV：编译完成后，头文件和库安装到这个文件夹，不要装进 `/usr`。装完之后，这个目录里会有 `lib/cmake/opencv4`，工程靠它找到 `libopencv_core.so.4.2`。目录名和 CMakeLists 里写的不一致时，运行会报找不到 `libopencv_core.so.4.2`。

​    cd 3rdparty/opencv-4.2.0

​    mkdir build
​    cd build

​    cmake -D CMAKE_INSTALL_PREFIX=$HOME/deps/opencv-4.2.0-install ..

​    make -j6

​    make install

这里不要加 sudo。`$HOME/deps` 在自己的主目录下，用 sudo 会把库装成 root 的家目录。

vikit 编在 `src/vikit_common/`，目标 `vikit_plugin`，不依赖单独的 `vikit_ros`。

CUDA 自己装。不装也能编译：3D-BBS 走 CPU。装了 CUDA 但不想编 GPU 库时，在下面的 colcon 命令里加上 `-DWITH_CUDA=OFF`。

### 编译

在工作空间根目录执行。这个仓库的工作空间根目录是源码包的上一级，例如 `/home/abc/lyx/lightning-lm2`，不是 `src/lightning-lm2`。在源码包里面执行的话，`log/` 和 `install/` 会写进源码目录。

​    source /opt/ros/humble/setup.bash

​    MAKEFLAGS="-j4" colcon build --packages-select lightning --cmake-args -DCMAKE_BUILD_TYPE=Release

​    source install/setup.bash

`MAKEFLAGS="-j4"` 是编译时最多 4 个线程。

### 流程：

1. 实时建图: ros2 launch lightning mapping.launch.py

   - 在线回环与 LIO 同时运行，在源码包的 PCD/ 下生成 pcd 格式点云，将 global_loop_map.pcd 放到 MAP 下作为 localization 的静态地图，MAP/VoxelMap/ 作为 3d-bbs 的先验体素地图。
   - 将新的 MAP 下点云及 MAP/VoxelMap 重命名为同名文件并写入 yaml 中的 system/global_loop_map_name，否则下次运行将被覆盖。
2. 实时定位: ros2 launch lightning localization.launch.py

## 辅助脚本

脚本的参数说明和使用流程详见 [`scripts/README.md`](scripts/README.md)：

- `calc_imu_tilt.py`
  - **什么时候做：**设备首次安装，或 IMU/雷达安装位置发生调整后，在水平地面静止录制一段 IMU 数据时使用；
  - **为什么做：**估计传感器相对地面的 Roll/Pitch，写入 `common/E_body_wrt_lio`，避免安装倾角把地图建斜。
