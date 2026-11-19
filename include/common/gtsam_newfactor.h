#include <gtsam/geometry/Pose3.h>          // 提供 Pose3 类
#include <gtsam/nonlinear/NonlinearFactor.h> // 提供 NoiseModelFactor1 基类
#include <gtsam/base/Matrix.h>              // 提供 Matrix 和 Vector 类型

namespace gtsam {

/**
 * 只约束 Pose3 的 Z 轴高度的一元因子
 * 误差公式: e = pose.z() - target_z
 */
class HeightFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
    double measured_z_;
public:
    HeightFactor(gtsam::Key j, double z, const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, j), measured_z_(z) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose, 
                                boost::optional<gtsam::Matrix&> H = boost::none) const override {
        if (H) {
            // 这是最关键的一步：
            // Pose3 的平移对位姿李代数的导数是一个 3x6 矩阵
            // 我们只需要其中代表 Z 的那一行（第三行）
            gtsam::Matrix H_all;
            pose.translation(H_all); 
            *H = H_all.row(2); // 只取 Z 轴的导数
        }
        return (gtsam::Vector(1) << pose.z() - measured_z_).finished();
    }
};

} // namespace gtsam