add_library(cpu_bbs3d
  SHARED
  src/cpu_bbs3d/bbs3d.cpp
  src/cpu_bbs3d/voxelmaps.cpp
  src/cpu_bbs3d/voxelmaps_io.cpp
)

target_include_directories(cpu_bbs3d
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if (LIDAR_SLAM_HAS_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARY_DIRS})

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        --expt-relaxed-constexpr
        -Xcudafe "--diag_suppress=esa_on_defaulted_function_ignored"
        -Xcudafe "--diag_suppress=20011"
        -Xcudafe "--diag_suppress=20012"
        -Xcudafe "--diag_suppress=20014"
        -Xcudafe "--diag_suppress=20015"
        -Xcompiler=-Wno-deprecated-copy,-Wno-cpp,-Wno-expansion-to-defined,-Wno-attributes,-Wno-literal-suffix,-w
    )

    cuda_add_library(gpu_bbs3d
    SHARED
        src/gpu_bbs3d/bbs3d.cu
        src/gpu_bbs3d/calc_score.cu
        src/gpu_bbs3d/voxelmaps.cu
        src/gpu_bbs3d/stream_manager/check_error.cu
        src/gpu_bbs3d/voxelmaps_io.cu
        src/run_gpu.cpp
    )

    target_include_directories(gpu_bbs3d
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${PROJECT_SOURCE_DIR}/include
        ${CUDA_INCLUDE_DIRS}
    )
    message(STATUS "3D-BBS: GPU library enabled")
else()
    message(STATUS "3D-BBS: CPU only")
endif()
