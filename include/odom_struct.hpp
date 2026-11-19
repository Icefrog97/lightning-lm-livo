#ifndef ODOM_STRUCT_HPP
#define ODOM_STRUCT_HPP

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <iostream>

#include "lidar_slam_math.hpp"

struct OdomStructf
{
    double timestamp = 0.0;
    Eigen::Vector3f T; // translation
    Eigen::Quaternionf Q;
    Eigen::Matrix3f R() { return Q.normalized().toRotationMatrix(); } // rotation
    Eigen::Matrix4f M() { return lightning::RT_to_SE3(R(), T); }     // se3
    OdomStructf(Eigen::Vector3f T, Eigen::Quaternionf Q)
    {
        this->T = T;
        this->Q = Q;
    }
    OdomStructf(Eigen::Matrix4f m)
    {
        this->Q = Eigen::Quaternionf(m.block<3,3>(0,0));
        this->T = m.block<3, 1>(0, 3);
    }
    OdomStructf()
    {
        this->Q = Eigen::Quaternionf::Identity();
        this->T = Eigen::Vector3f::Zero();
    }
    void setFromSE3(Eigen::Matrix4f m)
    {
        Q = Eigen::Quaternionf(m.block<3,3>(0,0));
        T = m.block<3, 1>(0, 3);
    }
    void print()
    {
        std::cout << "T: " << T.transpose() << "\n"
                  << "Q: " << Q.coeffs().transpose() << "\n\n";
    }
};

struct OdomStructd
{
    double timestamp = 0.0;
    Eigen::Vector3d T; // translation
    Eigen::Quaterniond Q;
    Eigen::Matrix3d R() { return Q.normalized().toRotationMatrix(); } // rotation
    Eigen::Matrix4d M() { return lightning::RT_to_SE3(R(), T); }     // se3
    OdomStructd(Eigen::Vector3d T, Eigen::Quaterniond Q)
    {
        this->T = T;
        this->Q = Q;
    }
    OdomStructd(Eigen::Matrix4d m)
    {
        this->Q = Eigen::Quaterniond(m.block<3,3>(0,0));
        this->T = m.block<3, 1>(0, 3);
    }
    OdomStructd()
    {
        this->Q = Eigen::Quaterniond::Identity();
        this->T = Eigen::Vector3d::Zero();
    }
    void setFromSE3(Eigen::Matrix4d m)
    {
        Q = Eigen::Quaterniond(m.block<3,3>(0,0));
        T = m.block<3, 1>(0, 3);
    }
    void print()
    {
        std::cout << "T: " << T.transpose() << "\n"
                  << "Q: " << Q.coeffs().transpose() << "\n\n";
    }
};

#endif