## 使用说明：

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

### 1. 建图前端可以三选一

lightning-lm 的纯激光地图是 iVox。这里另外接了 FAST-LIVO2 的 VoxelMap。VoxelMap 可以单独当纯激光用。

### 2. 相对 FAST-LIVO2 的 LRU

FAST-LIVO2 的 VoxelMap 和视觉特征图会一直保留。这里两张图都按最近使用淘汰，长轨迹不会把内存一直堆上去。

### 3. 动态物剔除

点云去畸变之后、匹配之前，用图像检测结果把人、车等框掉。

**检测器**

- 有 TensorRT 时走 YOLO
- 没有 TensorRT 时编一个空检测器，纯激光照常编译、运行
- 被删掉的点发到 `/dynamic_filter/removed` 和 `/dynamic_filter/removed_body`

**模型**

- `model/` 可以直接用，换场景时不建议完全沿用
- 换场景：自己采集图像，用 [Ultralytics](https://github.com/ultralytics/ultralytics) 训练 YOLO，导出 TensorRT engine，路径填到 launch 的模型参数

### 4. 回环

lightning-lm 按距离找历史关键帧，用 NDT 验证，再用 miao 做位姿图。这里换成 [STD](https://github.com/hku-mars/STD) 加 [GTSAM](https://github.com/borglab/gtsam) ISAM2。

**高度因子**

- `use_height_factor` 默认打开
- 每个关键帧加一条一元约束，把 z 拉向第一帧高度，标准差 0.1 m
- 用来压住单层里的高度漂移

### 5. 重定位

没有初值时用 [3D-BBS](https://github.com/KOKIAOKI/3d_bbs) 做全局搜索。

### 6. 重置

定位连续 10 帧激光匹配失败后，状态打成 FAIL，并重置定位、PGO 和 LIO。位姿回到最近一次还匹配得上时记下的恢复位姿，速度清零。IMU、雷达、图像缓冲清空，iVox 重建，VoxelMap 和视觉特征图清空，下一帧按首帧处理。

### 7. 退化

- 只看平移信息矩阵 `H_tt` 的特征值，旋转块不抑制
- 开关：`fasterlio/use_degeneracy`
- 比例：`fasterlio/degeneracy_threshold_ratio`
- 这种做法按特征值比例压平移，旋转和平移还耦合在一起
- 更精确的模型推荐 [DCReg](https://github.com/JokerJohn/DCReg)：用 Schur 补把旋转和平移的可观性拆开，再只对弱方向做处理

### 8. 调平

雷达斜着安装时，用它把轨迹和点云拉平。

- `common/T_body_wrt_lio`：平移
- `common/E_body_wrt_lio`：yaw、pitch、roll，单位是度，表示后轴中心相对 LIO/IMU 的位姿
- 发布的轨迹、点云和回环位姿都左乘它的逆
- 水平地面上用 `scripts/calc_imu_tilt.py` 估 pitch 和 roll

## 编译

### 环境

Ubuntu 22.04，ROS 2 Humble。

### 依赖

1. 将 MAP/ 下的 example_* 复制并重命名为 *。
2. 安装依赖

- PCL、Eigen、Ceres、yaml-cpp、glog、geographiclib、OpenCV、gflags、pcl_conversions、OpenMP
- CUDA 可选。没有 CUDA 时 3D-BBS 用 CPU，动态物剔除不走 TensorRT

```bash
sudo apt-get install libpcl-dev pcl-tools libyaml-cpp-dev ros-humble-pcl-conversions libfmt-dev
```

**Ceres**

```bash
cd 3rdparty/ceres
mkdir build && cd build
cmake ..
make -j6
sudo make install
```

**GeographicLib**

```bash
cd 3rdparty/geographiclib
mkdir build && cd build
cmake ..
make -j6
sudo make install
```

**GTSAM**

GTSAM 要对着系统里的 Eigen 编译。Ubuntu 22.04 的系统 Eigen 是 3.4。`3rdparty/gtsam-4.2.0/cmake/HandleEigen.cmake` 里 `GTSAM_USE_SYSTEM_EIGEN` 默认已经是 ON。如果以前用别的 Eigen 装过 GTSAM，先把 `build` 目录删掉再 cmake，否则会沿用旧的缓存。

```bash
cd 3rdparty/gtsam-4.2.0
mkdir build && cd build
cmake ..
make -j6
sudo make install
```

**OpenCV 4.2.0**

工程不用系统自带的 OpenCV。系统里一般是 4.5，这个工程的 CMakeLists 指定要 4.2.0，并且只到下面这个目录里找：

`$HOME/deps/opencv-4.2.0-install`

`$HOME` 是当前用户的主目录。用户名是 `abc` 时，这个路径就是 `/home/abc/deps/opencv-4.2.0-install`。`CMAKE_INSTALL_PREFIX` 是告诉 OpenCV：头文件和库安装到这个文件夹。装完之后，这个目录里会有 `lib/cmake/opencv4`，工程靠它找到 `libopencv_core.so.4.2`。目录名和 CMakeLists 里写的不一致时，运行会报找不到 `libopencv_core.so.4.2`。

`make install` 不要加 sudo。`$HOME/deps` 在自己的主目录下，用 sudo 会把库装进 root 的家目录。

```bash
cd 3rdparty/opencv-4.2.0
mkdir build && cd build
cmake -D CMAKE_INSTALL_PREFIX=$HOME/deps/opencv-4.2.0-install ..
make -j6
make install
```

**vikit 与 CUDA**

- vikit 编在 `src/vikit_common/`，目标 `vikit_plugin`
- CUDA >= 12.0 自己装。不装也能编译，3D-BBS 走 CPU

### 编译工程

```bash
colcon build
source install/setup.bash
```

### 运行

**1. 实时建图**

```bash
ros2 launch lightning mapping.launch.py
```

- 在线回环与 LIO 同时运行
- 在源码包的 `PCD/` 下生成 pcd
- 把 `global_loop_map.pcd` 放到 `MAP/` 下，作为定位的静态地图
- `MAP/VoxelMap/` 作为 3D-BBS 的先验体素地图
- 把新的 `MAP` 下点云和 `MAP/VoxelMap` 重命名为同名文件，并写入 yaml 的 `system/global_loop_map_name`，否则下次运行会被覆盖

**2. 实时定位**

```bash
ros2 launch lightning localization.launch.py
```

## 辅助脚本

参数说明和使用流程见 [`scripts/README.md`](scripts/README.md)。

**`calc_imu_tilt.py`**

- 什么时候做：设备首次安装，或 IMU/雷达安装位置调整后，在水平地面静止录一段 IMU
- 为什么做：估计传感器相对地面的 roll、pitch，写入 `common/E_body_wrt_lio`，避免安装倾角把地图建斜
