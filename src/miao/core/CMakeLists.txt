set(glog_DIR "/usr/local/lib/cmake/glog")
find_package(glog REQUIRED CONFIG NO_DEFAULT_PATH)

# 2. 检查头文件是否来自 /usr/local
include_directories(/usr/local/include)

add_library(miao.core SHARED
        graph/graph.cc
        graph/optimizer.cc

        common/string_tools.cc

        math/marginal_covariance_cholesky.cc

        opti_algo/optimization_algorithm.cc
        opti_algo/opti_algo_gauss_newton.cc
        opti_algo/opti_algo_lm.cc
        opti_algo/opti_algo_dogleg.cc

        solver/solver.cc
)

target_include_directories(miao.core PRIVATE
        ${PROJECT_SOURCE_DIR}/src/miao
)

target_link_libraries(
        miao.core
        glog::glog tbb
)