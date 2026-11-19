//
// Created by xiang on 2022/2/15.
//

#include "lio/eskf.hpp"

#include "lightning_math.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>

namespace {

using CovType = lightning::ESKF::CovType;

// 狭窄巷道表现不好
void SymmetrizeAndFloorCovariance(CovType& P, double min_cov_diag) {
    P = 0.5 * (P + P.transpose()).eval();

    for (int i = 0; i < P.rows(); ++i) {
        if (P(i, i) < min_cov_diag) {
            P(i, i) = min_cov_diag;
        } else if (P(i, i) > 100.0) {
            P(i, i) = 100.0;
        }

        for (int j = 0; j < P.cols(); ++j) {
            if (std::isnan(P(i, j)) || std::isinf(P(i, j))) {
                LOG(WARNING) << "find nan or inf in P: " << P(i, j);
                P(i, j) = 1.0;
            }
        }
    }
}

}

namespace lightning {

void ESKF::Predict(const double& dt, const ESKF::ProcessNoiseType& Q, const Vec3d& gyro, const Vec3d& acce) {
    NavState::FullVectState f_ = x_.get_f(gyro, acce);  // 调用get_f 获取 速度 角速度 加速度
    Eigen::Matrix<double, NavState::full_dim, state_dim_> f_x_ = x_.df_dx(acce);

    Eigen::Matrix<double, NavState::full_dim, process_noise_dim_> f_w_ = x_.df_dw();
    Eigen::Matrix<double, state_dim_, process_noise_dim_> f_w_final =
        Eigen::Matrix<double, state_dim_, process_noise_dim_>::Zero();;

    NavState x_before = x_;
    x_.oplus(f_, dt);

    F_x1_ = CovType::Identity();

    // set f_x_final
    CovType f_x_final = CovType::Zero();;  // 23x23
    for (auto st : x_.vect_states_) {
        int idx = st.idx_;
        int dim = st.dim_;
        int dof = st.dof_;

        for (int i = 0; i < state_dim_; i++) {
            for (int j = 0; j < dof; j++) {
                f_x_final(idx + j, i) = f_x_(dim + j, i);
            }
        }

        for (int i = 0; i < process_noise_dim_; i++) {
            for (int j = 0; j < dof; j++) {
                f_w_final(idx + j, i) = f_w_(dim + j, i);
            }
        }
    }

    Mat3d res_temp_SO3;
    Vec3d seg_SO3;
    for (auto st : x_.SO3_states_) {
        int idx = st.idx_;
        int dim = st.dim_;
        for (int i = 0; i < 3; i++) {
            seg_SO3(i) = -1 * f_(dim + i) * dt;
        }

        F_x1_.block<3, 3>(idx, idx) = math::exp(seg_SO3, 0.5).matrix();

        res_temp_SO3 = math::A_matrix(seg_SO3);
        for (int i = 0; i < state_dim_; i++) {
            f_x_final.template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_.block<3, 1>(dim, i));
        }

        for (int i = 0; i < process_noise_dim_; i++) {
            f_w_final.template block<3, 1>(idx, i) = res_temp_SO3 * (f_w_.block<3, 1>(dim, i));
        }
    }

    Eigen::Matrix<double, 2, 3> res_temp_S2;
    Vec3d seg_S2;
    for (auto st : x_.S2_states_) {
        int idx = st.idx_;
        int dim = st.dim_;
        for (int i = 0; i < 3; i++) {
            seg_S2(i) = f_(dim + i) * dt;
        }

        SO3 res = math::exp(seg_S2, 0.5f);

        Vec2d vec = Vec2d::Zero();
        Eigen::Matrix<double, 2, 3> Nx = x_.grav_.S2_Nx_yy();
        Eigen::Matrix<double, 3, 2> Mx = x_before.grav_.S2_Mx(vec);

        F_x1_.block<2, 2>(idx, idx) = Nx * res.matrix() * Mx;

        Eigen::Matrix<double, 3, 3> x_before_hat = x_before.grav_.S2_hat();
        res_temp_S2 = -Nx * res.matrix() * x_before_hat * math::A_matrix(seg_S2).transpose();

        for (int i = 0; i < state_dim_; i++) {
            f_x_final.block<2, 1>(idx, i) = res_temp_S2 * (f_x_.block<3, 1>(dim, i));
        }
        for (int i = 0; i < process_noise_dim_; i++) {
            f_w_final.block<2, 1>(idx, i) = res_temp_S2 * (f_w_.block<3, 1>(dim, i));
        }
    }

    F_x1_ += f_x_final * dt;
    P_ = (F_x1_)*P_ * (F_x1_).transpose() + (dt * f_w_final) * Q * (dt * f_w_final).transpose();

    // FAST ImuProcess: if (exposure_estimate_en) cov_w(6,6) = cov_inv_expo * dt * dt
    // NavState has no expo column in df_dw (12-dim IMU noise). Add the same scalar Q here.
    // Invoked on kf_ (mapping thread) and kf_imu_ (IMU callback / mapping, mtx_buffer_ held).
    // This only writes this instance's P_; do not take mtx_buffer_ inside Predict.
    if (options_.exposure_estimate_en_ && dt > 0.0 && options_.inv_expo_cov_ > 0.0)
    {
        P_(NavState::inv_expo_idx, NavState::inv_expo_idx) += options_.inv_expo_cov_ * dt * dt;
    }

    // double s = options_.predict_cov_inflation_;
    // P_.block<3, 3>(0, 0) *= s;    // pos
    // P_.block<3, 3>(3, 3) *= s;    // rot
    // P_.block<3, 3>(12, 12) *= s;  // vel
    
    // SymmetrizeAndFloorCovariance(P_, options_.min_cov_diag_);
}

/**
 * 原版的迭代过程中，收敛次数大于1才会结果，所以需要两次收敛。
 * 在未收敛时，实际上不会计算最近邻，也就回避了一次ObsModel的计算
 * 如果这边对每次迭代都计算最近邻的话，时间明显会变长一些，并不是非常合理。。
 *
 * @param obs
 * @param R
 */
void ESKF::Update(ESKF::ObsType obs, const double& R) {
    custom_obs_model_.valid_ = true;
    custom_obs_model_.converge_ = true;

    CovType P_propagated = P_;

    Eigen::Matrix<double, state_dim_, 1> K_r;
    Eigen::Matrix<double, state_dim_, state_dim_> K_H;

    StateVecType dx_current = StateVecType::Zero();  // 本轮迭代的dx

    NavState start_x = x_;  // 迭代的起点
    NavState last_x = x_;

    int converged_times = 0;
    double last_lidar_res = 0;

    double init_res = 0.0;
    static double iterated_num = 0;
    static double update_num = 0;
    update_num += 1;
    const bool is_lidar_obs = (obs == ObsType::LIDAR || obs == ObsType::WHEEL_SPEED_AND_LIDAR);
    for (int i = -1; i < maximum_iter_; i++) {
        custom_obs_model_.valid_ = true;

        /// 计算observation function，主要是residual_, h_x_, s_
        /// x_ 在每次迭代中都是更新的，线性化点也会更新
        if (obs == ObsType::LIDAR || obs == ObsType::WHEEL_SPEED_AND_LIDAR) {
            lidar_obs_func_(x_, custom_obs_model_);
        } else if (obs == ObsType::WHEEL_SPEED) {
            wheelspeed_obs_func_(x_, custom_obs_model_);
        } else if (obs == ObsType::ACC_AS_GRAVITY) {
            acc_as_gravity_obs_func_(x_, custom_obs_model_);
        } else if (obs == ObsType::GPS) {
            gps_obs_func_(x_, custom_obs_model_);
        } else if (obs == ObsType::BIAS) {
            bias_obs_func_(x_, custom_obs_model_);
        } else if (obs == ObsType::STATIONARY_BIAS_AND_ACC) {
            stationary_bias_and_acc_obs_func_(x_, custom_obs_model_);
        }

        if (custom_obs_model_.valid_ == false) {
            x_ = last_x;
            P_ = P_propagated;
            return;
        }

        if (use_aa_ && i > -1 && (obs == ObsType::LIDAR || obs == ObsType::WHEEL_SPEED_AND_LIDAR) &&
            custom_obs_model_.lidar_residual_mean_ >= last_lidar_res * 1.01) {
            x_ = last_x;
            break;
        }
        iterated_num += 1;

        if (!custom_obs_model_.valid_) {
            continue;
        }

        if (i == -1) {
            init_res = custom_obs_model_.lidar_residual_mean_;
            if (init_res < 1e-9) {
                init_res = 1e-9;  // 可能有零
            }
        }

        iterations_ = i + 2;  // i从-1开始计
        final_res_ = custom_obs_model_.lidar_residual_mean_ / init_res;

        StateVecType dx = x_.boxminus(start_x);  // 当前x与起点之间的dx
        dx_current = dx;                         //

        P_ = P_propagated;

        for (auto it : x_.SO3_states_) {
            int idx = it.idx_;
            Vec3d seg_SO3 = dx.block<3, 1>(idx, 0);
            Mat3d res_temp_SO3 = math::A_matrix(seg_SO3).transpose();  // 小块的J阵, SO3上的雅可比？

            dx_current.block<3, 1>(idx, 0) = res_temp_SO3 * dx.block<3, 1>(idx, 0);

            /// P 上面有SO3的行 进行转换
            for (int j = 0; j < state_dim_; j++) {
                P_.block<3, 1>(idx, j) = res_temp_SO3 * (P_.block<3, 1>(idx, j));
            }
            /// P 上面有SO3的列 进行转换
            for (int j = 0; j < state_dim_; j++) {
                P_.block<1, 3>(j, idx) = (P_.block<1, 3>(j, idx)) * res_temp_SO3.transpose();
            }
        }

        for (auto it : x_.S2_states_) {
            int idx = it.idx_;

            Vec2d seg_S2 = dx.block<2, 1>(idx, 0);

            Eigen::Matrix<double, 2, 3> Nx = x_.grav_.S2_Nx_yy();
            Eigen::Matrix<double, 3, 2> Mx = start_x.grav_.S2_Mx(seg_S2);
            Mat2d res_temp_S2 = Nx * Mx;

            dx_current.block<2, 1>(idx, 0) = res_temp_S2 * dx.block<2, 1>(idx, 0);

            for (int j = 0; j < state_dim_; j++) {
                P_.block<2, 1>(idx, j) = res_temp_S2 * (P_.block<2, 1>(idx, j));
            }

            for (int j = 0; j < state_dim_; j++) {
                P_.block<1, 2>(j, idx) = (P_.block<1, 2>(j, idx)) * res_temp_S2.transpose();
            }
        }

        CovType HTH = custom_obs_model_.HTH_;
        StateVecType HTr = custom_obs_model_.HTr_;

        CovType HTH_eff = HTH;
        StateVecType HTr_eff = HTr;

        // 当前只对 h_tt 做退化检测，留有旋转块 H_rr 不抑制（避免平移/旋转量级混比误判）
        if (is_lidar_obs && options_.use_degeneracy_) {
            Mat6d HTH_pose = HTH.block<pose_obs_dim_, pose_obs_dim_>(0, 0);
            Mat6d HTH_pose_sym = 0.5 * (HTH_pose + HTH_pose.transpose());

            // 仅对平移块 H_tt 做退化检测；旋转块 H_rr 不抑制（避免平移/旋转量级混比误判）
            Mat3d H_tt = HTH_pose_sym.block<3, 3>(0, 0);
            Mat3d H_tt_sym = 0.5 * (H_tt + H_tt.transpose());

            Eigen::SelfAdjointEigenSolver<Mat3d> eigen_solver_tt(H_tt_sym);
            if (eigen_solver_tt.info() != Eigen::Success) {
                LOG(WARNING) << "Failed to decompose ESKF H_tt observation information matrix.";
                continue;
            }

            const Vec3d eigen_values_tt = eigen_solver_tt.eigenvalues();
            const Mat3d eigen_vectors_tt = eigen_solver_tt.eigenvectors();
            const double max_eigen_value_tt = std::max(1e-12, eigen_values_tt.maxCoeff());
            const double degeneracy_threshold_tt = max_eigen_value_tt * options_.degeneracy_threshold_ratio_;

            const Vec3d eigen_ratios_tt = eigen_values_tt / max_eigen_value_tt;

            Vec3d trans_mask = Vec3d::Ones();
            for (int k = 0; k < 3; ++k) {
                if (eigen_values_tt(k) <= degeneracy_threshold_tt && eigen_values_tt(k) <= 8000) {
                    trans_mask(k) = 0;
                    LOG(WARNING) << "eigen values of H_tt: " << eigen_values_tt.transpose()
                      << " max ratio: " << eigen_ratios_tt.transpose();
                }
            }

            const Mat3d trans_projector =
                eigen_vectors_tt * trans_mask.asDiagonal() * eigen_vectors_tt.transpose();

            Mat6d pose_projector = Mat6d::Identity();
            pose_projector.block<3, 3>(0, 0) = trans_projector;

            HTH_eff.block<pose_obs_dim_, pose_obs_dim_>(0, 0) =
                pose_projector * HTH_pose_sym * pose_projector;
            HTr_eff.head<pose_obs_dim_>() = pose_projector * HTr.head<pose_obs_dim_>();
        }

        CovType P_temp = (P_ / R).inverse();  // P阵上面已经更新

        P_temp += HTH_eff;
        CovType Q_inv = P_temp.inverse();     // Q inv
        K_r = Q_inv * HTr_eff;
        K_H = Q_inv * HTH_eff;

        dx_current = K_r + (K_H - Eigen::Matrix<double, state_dim_, state_dim_>::Identity()) * dx_current;

        // check nan
        for (int j = 0; j < state_dim_; ++j) {
            if (std::isnan(dx_current(j, 0))) {
                return;
            }
        }

        const double dx_translation = dx_current.head<3>().norm();
        const double dx_rotation_deg = dx_current.segment<3>(3).norm() * 180.0 / M_PI;
        if (dx_translation > options_.max_update_translation_step_ ||
            dx_rotation_deg > options_.max_update_rotation_step_deg_) {
            LOG(ERROR) << "Reject ESKF iter update, dtrans: " << dx_translation << ", drot_deg: " << dx_rotation_deg;
                    //    << ", dvel: " << dx_current.segment<NavState::kBlockDim>(NavState::kVelIdx).norm();
            x_ = start_x;
            P_ = P_propagated;
            return;
        }

        if (!use_aa_) {
            x_ = x_.boxplus(dx_current);
        } else {
            // 转到起点的线性空间
            x_ = x_.boxplus(dx_current);

            if (i == -1) {
                aa_.init(dx_current);  // 初始化AA
            } else {
                // 利用AA计算dx from start
                auto dx_all = x_.boxminus(start_x);
                auto new_dx_all = aa_.compute(dx_all);
                x_ = start_x.boxplus(new_dx_all);
            }
        }

        last_x = x_;

        // update last res
        last_lidar_res = custom_obs_model_.lidar_residual_mean_;
        custom_obs_model_.converge_ = true;

        for (int j = 0; j < state_dim_; j++) {
        // for (auto j = 0; j < pose_obs_dim_; j++) {
            if (std::fabs(dx_current[j]) > limit_[j]) {
                custom_obs_model_.converge_ = false;
                break;
            }
        }

        if (custom_obs_model_.converge_) {
            converged_times++;
        }

        if (!converged_times && i == maximum_iter_ - 2) {
            custom_obs_model_.converge_ = true;
        }

        if (converged_times > 0 || i == maximum_iter_ - 1) {
            /// 结束条件：已经收敛
            /// 更新P阵, using (45)
            L_ = P_;
            Mat3d res_temp_SO3;
            Vec3d seg_SO3;
            for (auto it : x_.SO3_states_) {
                int idx = it.idx_;
                for (int j = 0; j < 3; j++) {
                    seg_SO3(j) = dx_current(j + idx);
                }

                res_temp_SO3 = math::A_matrix(seg_SO3).transpose();
                for (int j = 0; j < state_dim_; j++) {
                    L_.block<3, 1>(idx, j) = res_temp_SO3 * (P_.block<3, 1>(idx, j));
                }

                for (int j = 0; j < state_dim_; j++) {
                    K_H.block<3, 1>(idx, j) = res_temp_SO3 * (K_H.block<3, 1>(idx, j));
                }

                for (int j = 0; j < state_dim_; j++) {
                    L_.block<1, 3>(j, idx) = (L_.block<1, 3>(j, idx)) * res_temp_SO3.transpose();
                    P_.block<1, 3>(j, idx) = (P_.block<1, 3>(j, idx)) * res_temp_SO3.transpose();
                }
            }

            Mat2d res_temp_S2;
            Vec2d seg_S2;
            for (auto it : x_.S2_states_) {
                int idx = it.idx_;

                for (int j = 0; j < 2; j++) {
                    seg_S2(j) = dx_current(j + idx);
                }

                Eigen::Matrix<double, 2, 3> Nx = x_.grav_.S2_Nx_yy();
                Eigen::Matrix<double, 3, 2> Mx = start_x.grav_.S2_Mx(seg_S2);
                res_temp_S2 = Nx * Mx;

                for (auto j = 0; j < state_dim_; j++) {
                    L_.block<2, 1>(idx, j) = res_temp_S2 * (P_.block<2, 1>(idx, j));
                }

                for (auto j = 0; j < state_dim_; j++) {
                    K_H.block<2, 1>(idx, j) = res_temp_S2 * (K_H.block<2, 1>(idx, j));
                }

                for (int j = 0; j < state_dim_; j++) {
                    L_.block<1, 2>(j, idx) = (L_.block<1, 2>(j, idx)) * res_temp_S2.transpose();
                    P_.block<1, 2>(j, idx) = (P_.block<1, 2>(j, idx)) * res_temp_S2.transpose();
                }
            }

            P_ = L_ - K_H * P_;

            break;
        }
    }

    // SymmetrizeAndFloorCovariance(P_, options_.min_cov_diag_);
}

}  // namespace lightning