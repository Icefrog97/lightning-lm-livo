#pragma once

#ifndef IMU_PROCESSING_H
#define IMU_PROCESSING_H

#include <cmath>
#include <deque>
#include <fstream>

#include "common/eigen_types.h"
#include "common/measure_group.h"
#include "common/point_def.h"
#include "lio/eskf.hpp"
#include "lio/pose6d.h"
#include "common/timer.h"
#include "lio/imu_filter.h"

namespace lightning
{
    enum class ImuResetLevel
    {
        INIT,          // 初始化
        DYNAMIC_RESET, // 动态 reset
        STATIC_RESET,  // 静态 reset
    };

    class ImuProcess
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ImuProcess();
        ~ImuProcess();

        void Reset();
        void setExtrinsic(const Vec3d &transl, const Mat3d &rot);
        void setGyrCov(const Vec3d &scaler);
        void setAccCov(const Vec3d &scaler);
        void setGyrBiasCov(const Vec3d &b_g);
        void setAccBiasCov(const Vec3d &b_a);
        void setMaxInitCount(int count) { max_init_count_ = count; }

        void Process(const MeasureGroup &meas, ESKF &kf_state, CloudPtr &pcl_un_);

        bool IsIMUInited() const { return imu_need_init_ == false; }
        void SetUseIMUFilter(bool b) { use_imu_filter_ = b; }

        Eigen::Matrix<double, 12, 12> Q_;
        Vec3d cov_acc_;
        Vec3d cov_gyr_;
        Vec3d cov_acc_scale_;
        Vec3d cov_gyr_scale_;
        Vec3d cov_bias_gyr_;
        Vec3d cov_bias_acc_;

        bool flag_degen_{false};

        ImuResetLevel reset_level_ = ImuResetLevel::INIT;

    private:
        void IMUInit(const MeasureGroup &meas, ESKF &kf_state, int &N);
        void UndistortPcl(const MeasureGroup &meas, ESKF &kf_state, CloudPtr &pcl_in_out);

        int max_init_count_ = 20;

        lightning::IMUPtr last_imu_ = nullptr;
        std::vector<Pose6D> imu_pose_;
        Mat3d R_lidar_imu_ = Mat3d ::Identity();
        Vec3d t_lidar_imu_ = Vec3d ::Zero();
        Vec3d mean_acc_ = Vec3d::Zero();
        Vec3d mean_gyr_ = Vec3d::Zero();
        Vec3d angvel_last_ = Vec3d ::Zero();
        Vec3d acc_s_last_ = Vec3d ::Zero();

        double last_lidar_end_time_ = 0;
        int init_iter_num_ = 1;
        bool b_first_frame_ = true;
        bool imu_need_init_ = true;
        S2 static_grav_;
        Vec3d static_bg_;

        bool use_imu_filter_ = true;
        IMUFilter filter_;
    };

    inline ImuProcess::ImuProcess() : b_first_frame_(true), imu_need_init_(true)
    {
        init_iter_num_ = 1;
        Q_.setZero();
        Q_.diagonal() << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5;
        cov_acc_ = Vec3d(0.1, 0.1, 0.1);
        cov_gyr_ = Vec3d(0.1, 0.1, 0.1);
        cov_bias_gyr_ = Vec3d(0.0001, 0.0001, 0.0001);
        cov_bias_acc_ = Vec3d(0.0001, 0.0001, 0.0001);
        mean_acc_ = Vec3d(0, 0, -1.0);
        mean_gyr_ = Vec3d(0, 0, 0);

        last_imu_.reset(new lightning::IMU());
    }

    inline ImuProcess::~ImuProcess() {}

    inline void ImuProcess::Reset()
    {
        mean_acc_ = Vec3d::Zero();
        mean_gyr_ = Vec3d::Zero();
        angvel_last_.setZero();

        imu_need_init_ = true;
        init_iter_num_ = 1;
        imu_pose_.clear();
        last_imu_.reset(new lightning::IMU());
    }

    inline void ImuProcess::setExtrinsic(const Vec3d &transl, const Mat3d &rot)
    {
        t_lidar_imu_ = transl;
        R_lidar_imu_ = rot;
    }

    inline void ImuProcess::setGyrCov(const Vec3d &scaler)
    {
        cov_gyr_scale_ = scaler;
    }

    inline void ImuProcess::setAccCov(const Vec3d &scaler)
    {
        cov_acc_scale_ = scaler;
    }

    inline void ImuProcess::setGyrBiasCov(const Vec3d &b_g)
    {
        cov_bias_gyr_ = b_g;
    }

    inline void ImuProcess::setAccBiasCov(const Vec3d &b_a)
    {
        cov_bias_acc_ = b_a;
    }

    inline void ImuProcess::IMUInit(const MeasureGroup &meas, ESKF &kf_state, int &N)
    {
        /** 1. initializing the gravity, gyro bias, acc and gyro covariance
         ** 2. normalize the acceleration measurenments to unit gravity **/

        Vec3d cur_acc, cur_gyr;

        if (b_first_frame_)
        {
            Reset();
            N = 1;
            b_first_frame_ = false;
            const auto &imu_acc = meas.imu_.front()->linear_acceleration;
            const auto &gyr_acc = meas.imu_.front()->angular_velocity;
            mean_acc_ = imu_acc;
            mean_gyr_ = gyr_acc;
        }

        for (const auto &imu : meas.imu_)
        {
            const auto &imu_acc = imu->linear_acceleration;
            const auto &gyr_acc = imu->angular_velocity;
            cur_acc = imu_acc;
            cur_gyr = gyr_acc;

            mean_acc_ += (cur_acc - mean_acc_) / N;
            mean_gyr_ += (cur_gyr - mean_gyr_) / N;

            cov_acc_ = cov_acc_ * (N - 1.0) / N +
                       (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_) * (N - 1.0) / (N * N);
            cov_gyr_ = cov_gyr_ * (N - 1.0) / N +
                       (cur_gyr - mean_gyr_).cwiseProduct(cur_gyr - mean_gyr_) * (N - 1.0) / (N * N);

            N++;
        }
        NavState init_state = kf_state.GetX();

        init_state.timestamp_ = meas.imu_.back()->timestamp;
        init_state.grav_ = S2(-mean_acc_ / mean_acc_.norm() * G_m_s2);
        init_state.bg_ = mean_gyr_;
        init_state.offset_t_lidar_ = t_lidar_imu_;
        init_state.offset_R_lidar_ = R_lidar_imu_;
        init_state.ba_ = Vec3d::Zero();

        auto init_P = kf_state.GetP();
        init_P.setIdentity();
        init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
        init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
        init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
        init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
        init_P(21, 21) = init_P(22, 22) = 0.00001;

        if (reset_level_ == ImuResetLevel::DYNAMIC_RESET)
        {
            init_state.bg_ = static_bg_;
            init_state.grav_ = static_grav_;
            init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
            init_P(21, 21) = init_P(22, 22) = 0.00001;
        }
        else
        {
            static_grav_ = init_state.grav_;
            static_bg_ = init_state.bg_;
        }

        kf_state.ChangeX(init_state);
        kf_state.ChangeP(init_P);
        last_imu_ = meas.imu_.back();
    }

    inline void ImuProcess::UndistortPcl(const MeasureGroup &meas, ESKF &kf_state, CloudPtr &pcl_out)
    {
        /*** add the imu of the last frame-tail to the of current frame-head ***/
        auto v_imu = meas.imu_;
        v_imu.push_front(last_imu_);
        const double &imu_end_time = v_imu.back()->timestamp;
        const double &pcl_beg_time = meas.lidar_begin_time_;
        const double &pcl_end_time = meas.lidar_end_time_;

        /*** Initialize IMU pose ***/
        NavState imu_state = kf_state.GetX();
        imu_pose_.clear();
        imu_pose_.emplace_back(0.0, acc_s_last_, angvel_last_, imu_state.vel_, imu_state.pos_, imu_state.rot_.matrix());

        /*** forward propagation at each imu point ***/
        Vec3d angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
        Mat3d R_imu;

        double dt = 0;
        Vec3d acc = Vec3d::Zero();
        Vec3d gyro = Vec3d::Zero();

        if (use_imu_filter_) {
            for (auto &imu : v_imu) {
                auto imu_f = filter_.Filter(*imu);
                *imu = imu_f;
            }
        }

        for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
        {
            auto &&head = *(it_imu);
            auto &&tail = *(it_imu + 1);

            if (tail->timestamp < last_lidar_end_time_)
                continue;

            // TODO: 四阶龙格库塔
            // TODO: wheel 融合方案一：线速度代替 imu 加速度积分直接 predict 到状态中，需要考虑外参变换
            angvel_avr = .5 * (head->angular_velocity + tail->angular_velocity);
            acc_avr = .5 * (head->linear_acceleration + tail->linear_acceleration);

            acc_avr = acc_avr * G_m_s2 / mean_acc_.norm(); // - state_inout.ba;

            if (head->timestamp < last_lidar_end_time_)
                dt = tail->timestamp - last_lidar_end_time_;
            else
                dt = tail->timestamp - head->timestamp;

            acc = acc_avr;
            gyro = angvel_avr;

            Q_.block<3, 3>(0, 0).diagonal() = cov_gyr_;
            Q_.block<3, 3>(3, 3).diagonal() = cov_acc_;
            Q_.block<3, 3>(6, 6).diagonal() = cov_bias_gyr_;
            Q_.block<3, 3>(9, 9).diagonal() = cov_bias_acc_;
            kf_state.Predict(dt, Q_, gyro, acc);

            /* save the poses at each IMU measurements */
            imu_state = kf_state.GetX();
            angvel_last_ = angvel_avr - imu_state.bg_;
            acc_s_last_ = imu_state.rot_ * (acc_avr - imu_state.ba_);
            for (int i = 0; i < 3; i++)
                acc_s_last_[i] += imu_state.grav_[i];

            double &&offs_t = tail->timestamp - pcl_beg_time;
            imu_pose_.emplace_back(
                Pose6D(offs_t, acc_s_last_, angvel_last_, imu_state.vel_, imu_state.pos_, imu_state.rot_.matrix()));
        }

        /*** calculated the pos and attitude prediction at the frame-end ***/
        double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
        dt = note * (pcl_end_time - imu_end_time);
        kf_state.Predict(dt, Q_, gyro, acc);

        // yxliang 2026.2.5
        // 删除前版 wheel 融合逻辑，逻辑不通
        // 没有将时间戳对齐到 wheel 上，没有对每个 wheel 消息进行观测处理，没有外参变换

        imu_state = kf_state.GetX();
        last_imu_ = meas.imu_.back();
        last_lidar_end_time_ = pcl_end_time;

        /*** sort point clouds by offset time ***/
        pcl_out = meas.scan_;
        std::sort(pcl_out->points.begin(), pcl_out->points.end(),
                  [](const PointType &p1, const PointType &p2)
                  { return p1.time < p2.time; });
        /*** undistort each lidar point (backward propagation) ***/
        if (pcl_out->empty())
            return;
        auto it_pcl = pcl_out->points.end() - 1;
        for (auto it_kp = imu_pose_.end() - 1; it_kp != imu_pose_.begin(); it_kp--)
        {
            auto head = it_kp - 1;
            auto tail = it_kp;
            R_imu = (head->rot);
            vel_imu = (head->vel);
            pos_imu = (head->pos);
            acc_imu = (tail->acc);
            angvel_avr = (tail->gyr);

            for (; it_pcl->time / double(1000) > head->offset_time && it_pcl != pcl_out->points.begin(); it_pcl--)
            {
                dt = it_pcl->time / double(1000) - head->offset_time;

                /// dt 有时候存在非法数据
                if (dt < 0 || dt > lo::lidar_time_interval)
                {
                    // LOG(WARNING) << "find abnormal dt in cloud: " << dt;
                    continue;
                }

                Mat3d R_i(R_imu * lightning::math::Exp(angvel_avr, dt));

                Vec3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
                Vec3d T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos_);
                Vec3d P_compensate =
                    imu_state.offset_R_lidar_.inverse() *
                    (imu_state.rot_.inverse() *
                         (R_i * (imu_state.offset_R_lidar_ * P_i + imu_state.offset_t_lidar_) + T_ei) -
                     imu_state.offset_t_lidar_); // not accurate!

                // save Undistorted points and their rotation
                it_pcl->x = P_compensate(0);
                it_pcl->y = P_compensate(1);
                it_pcl->z = P_compensate(2);
            }
        }
    }

    inline void ImuProcess::Process(const MeasureGroup &meas, ESKF &kf_state, CloudPtr &cur_pcl_un)
    {
        if (meas.imu_.empty())
            return;

        if (imu_need_init_)
        {
            /// The very first lidar frame
            IMUInit(meas, kf_state, init_iter_num_);

            imu_need_init_ = true;

            last_imu_ = meas.imu_.back();

            NavState imu_state = kf_state.GetX();

            if (reset_level_ == ImuResetLevel::INIT || reset_level_ == ImuResetLevel::STATIC_RESET)
            {
                if (init_iter_num_ > max_init_count_)
                {
                    cov_acc_ *= pow(G_m_s2 / mean_acc_.norm(), 2);
                    imu_need_init_ = false;

                    cov_acc_ = cov_acc_scale_;
                    cov_gyr_ = cov_gyr_scale_;
                    LOG(INFO) << get_time_us() << "[INFO][ImuProcess::" << __FUNCTION__ << "] "
                              << "IMU Initial Done, bg: " << imu_state.bg_.transpose() << ", ba: " << imu_state.ba_.transpose();
                }
                else
                    LOG(INFO) << get_time_us() << "[INFO][ImuProcess::" << __FUNCTION__ << "] " << "waiting for imu init ..." << init_iter_num_;

                return;
            }
            else
                imu_need_init_ = false;
        }

        UndistortPcl(meas, kf_state, cur_pcl_un);
    }
}

#endif