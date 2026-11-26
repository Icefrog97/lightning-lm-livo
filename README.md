[中文](README.zh-CN.md)

## Usage

This repository continues development on the lidar mapping and localization of [lightning-lm](https://github.com/gaoxiang12/lightning-lm). The frontend is still FasterLIO / ESKF, and localization is still lidar matching plus PGO.

Projects used:

- [lightning-lm](https://github.com/gaoxiang12/lightning-lm): lidar mapping, localization, and the iVox frontend
- [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2): VoxelMap and visual odometry
- [STD](https://github.com/hku-mars/STD): triangle descriptors for loop closure
- [3D-BBS](https://github.com/KOKIAOKI/3d_bbs): global initialization for localization
- [GTSAM](https://github.com/borglab/gtsam): loop-closure pose graph, using ISAM2
- [rpg_vikit](https://github.com/uzh-rpg/rpg_vikit): camera models; source is in `src/vikit_common/`
- [Ultralytics](https://github.com/ultralytics/ultralytics): training the dynamic-object detection model
- [Ceres](https://github.com/ceres-solver/ceres-solver)

### 1. The mapping frontend can be one of three

The pure-lidar map in lightning-lm is iVox. VoxelMap from FAST-LIVO2 is also connected here. VoxelMap can be used on its own as pure lidar.

### 2. LRU relative to FAST-LIVO2

FAST-LIVO2 keeps its VoxelMap and visual feature map indefinitely. Here both maps evict by least recently used, so a long trajectory does not keep stacking memory.

### 3. Dynamic object removal

After point-cloud undistortion and before matching, image detections are used to mask out people, vehicles, and similar objects.

**Detector**

- Use YOLO when TensorRT is available
- Without TensorRT, build an empty detector; pure lidar still compiles and runs
- Removed points are published to `/dynamic_filter/removed` and `/dynamic_filter/removed_body`

**Model**

- `model/` can be used directly; fully reusing it on a new scene is not recommended
- New scene: collect your own images, train YOLO with [Ultralytics](https://github.com/ultralytics/ultralytics), export a TensorRT engine, and fill the path into the model parameter of the launch file

### 4. Loop closure

lightning-lm finds historical keyframes by distance, verifies them with NDT, then builds the pose graph with miao. This is replaced by [STD](https://github.com/hku-mars/STD) plus [GTSAM](https://github.com/borglab/gtsam) ISAM2.

**Height factor**

- `use_height_factor` is on by default
- Each keyframe adds a unary constraint that pulls z toward the first-frame height, with a standard deviation of 0.1 m
- Used to suppress height drift within a single floor

### 5. Relocalization

When there is no initial guess, [3D-BBS](https://github.com/KOKIAOKI/3d_bbs) does a global search.

### 6. Reset

After localization fails lidar matching for 10 consecutive frames, the state is set to FAIL, and localization, PGO, and LIO are reset. The pose returns to the recovery pose recorded at the last time matching still succeeded, and velocity is cleared. IMU, lidar, and image buffers are cleared, iVox is rebuilt, VoxelMap and the visual feature map are cleared, and the next frame is treated as the first frame.

### 7. Degeneracy

- Only the eigenvalues of the translational information matrix `H_tt` are considered; the rotation block is not suppressed
- Switch: `fasterlio/use_degeneracy`
- Ratio: `fasterlio/degeneracy_threshold_ratio`
- This approach suppresses translation by the eigenvalue ratio; rotation and translation remain coupled
- For a more accurate model, [DCReg](https://github.com/JokerJohn/DCReg) is recommended: use the Schur complement to separate the observability of rotation and translation, then handle only the weak directions

### 8. Leveling

When the lidar is mounted at an angle, this pulls the trajectory and point cloud level.

- `common/T_body_wrt_lio`: translation
- `common/E_body_wrt_lio`: yaw, pitch, roll, in degrees; the pose of the rear-axle center relative to LIO/IMU
- Published trajectories, point clouds, and loop-closure poses are all left-multiplied by its inverse
- On level ground, use `scripts/calc_imu_tilt.py` to estimate pitch and roll

## Build

### Environment

Ubuntu 22.04, ROS 2 Humble.

### Dependencies

1. Copy the example_* under MAP/ and rename them to *.
2. Install dependencies

- PCL, Eigen, Ceres, yaml-cpp, glog, geographiclib, OpenCV, gflags, pcl_conversions, OpenMP
- CUDA is optional. Without CUDA, 3D-BBS uses the CPU, and dynamic-object removal does not use TensorRT

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

GTSAM must be compiled against the system Eigen. The system Eigen on Ubuntu 22.04 is 3.4. In `3rdparty/gtsam-4.2.0/cmake/HandleEigen.cmake`, `GTSAM_USE_SYSTEM_EIGEN` is already ON by default. If GTSAM was previously installed against a different Eigen, delete the `build` directory before cmake, otherwise the old cache will be reused.

```bash
cd 3rdparty/gtsam-4.2.0
mkdir build && cd build
cmake ..
make -j6
sudo make install
```

**OpenCV 4.2.0**

The project does not use the system OpenCV. The system usually has 4.5. This project's CMakeLists requires 4.2.0 and looks only in the following directory:

`$HOME/deps/opencv-4.2.0-install`

`$HOME` is the current user's home directory. When the username is `abc`, this path is `/home/abc/deps/opencv-4.2.0-install`. `CMAKE_INSTALL_PREFIX` tells OpenCV to install headers and libraries into this folder. After installation, this directory contains `lib/cmake/opencv4`, which the project uses to find `libopencv_core.so.4.2`. If the directory name does not match what is written in CMakeLists, the run will report that `libopencv_core.so.4.2` cannot be found.

Do not add sudo to `make install`. `$HOME/deps` is under your own home directory; using sudo installs the libraries into root's home directory.

```bash
cd 3rdparty/opencv-4.2.0
mkdir build && cd build
cmake -D CMAKE_INSTALL_PREFIX=$HOME/deps/opencv-4.2.0-install ..
make -j6
make install
```

**vikit and CUDA**

- vikit is built in `src/vikit_common/`, target `vikit_plugin`
- Install CUDA >= 12.0 yourself. It still compiles without it; 3D-BBS uses the CPU

### Build the project

```bash
colcon build
source install/setup.bash
```

### Run

**1. Real-time mapping**

```bash
ros2 launch lightning mapping.launch.py
```

- Online loop closure runs together with LIO
- Generates pcd under `PCD/` of the source package
- Put `global_loop_map.pcd` under `MAP/` as the static map for localization
- `MAP/VoxelMap/` is the prior voxel map for 3D-BBS
- Rename the point clouds under the new `MAP` and `MAP/VoxelMap` to the same filenames, and write them into `system/global_loop_map_name` in the yaml; otherwise they will be overwritten the next time it runs

**2. Real-time localization**

```bash
ros2 launch lightning localization.launch.py
```

## Helper scripts

Parameter descriptions and the usage flow are in [`scripts/README.md`](scripts/README.md).

**`calc_imu_tilt.py`**

- When: on first installation of the device, or after the IMU/lidar mounting position is adjusted, record a segment of IMU while stationary on level ground
- Why: estimate the sensor's roll and pitch relative to the ground, write them into `common/E_body_wrt_lio`, and avoid the mounting tilt skewing the map
