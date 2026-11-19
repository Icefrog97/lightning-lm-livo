# Vendored like FAST-LIVO2: only vikit_common, no catkin / vikit_ros.
file(GLOB VIKIT_COMMON_SRC src/*.cpp)

add_library(vikit_plugin SHARED ${VIKIT_COMMON_SRC})

target_include_directories(vikit_plugin
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vikit_plugin
    PUBLIC
    ${OpenCV_LIBS}
    yaml-cpp
    pthread
)
