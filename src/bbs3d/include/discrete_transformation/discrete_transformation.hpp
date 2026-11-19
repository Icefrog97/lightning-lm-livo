#pragma once
#include <Eigen/Core>

template <typename T>
using Vector3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using Matrix4 = Eigen::Matrix<T, 4, 4>;

template <typename T>
class DiscreteTransformation {
public:
  DiscreteTransformation() : score(0), level(0), x(0), y(0), z(0), roll(0), pitch(0), yaw(0) {}

  DiscreteTransformation(int score) : score(score), level(0), x(0), y(0), z(0), roll(0), pitch(0), yaw(0) {}

  DiscreteTransformation(int score, int level, int x, int y, int z, int roll, int pitch, int yaw)
  : score(score),
    level(level),
    x(x),
    y(y),
    z(z),
    roll(roll),
    pitch(pitch),
    yaw(yaw) {}

  ~DiscreteTransformation() {}

  bool operator<(const DiscreteTransformation& rhs) const { return score < rhs.score; }

  bool is_leaf() const { return level == 0; }

  Matrix4<T> create_matrix(const T trans_res, const Vector3<T>& rpy_res, const Vector3<T>& min_rpy) {
    Eigen::Translation<T, 3> translation(x * trans_res, y * trans_res, z * trans_res);
    Eigen::AngleAxis<T> rollAngle(roll * rpy_res.x() + min_rpy.x(), Vector3<T>::UnitX());
    Eigen::AngleAxis<T> pitchAngle(pitch * rpy_res.y() + min_rpy.y(), Vector3<T>::UnitY());
    Eigen::AngleAxis<T> yawAngle(yaw * rpy_res.z() + min_rpy.z(), Vector3<T>::UnitZ());
    return (translation * yawAngle * pitchAngle * rollAngle).matrix();
  }

  void branch(std::vector<DiscreteTransformation>& b, const int child_level, const int v_rate, const Eigen::Vector3i& num_division,
              bool limit_z = false, T child_res = T(0), T min_z = T(0), T max_z = T(0)) {
    for (int i = 0; i < v_rate; i++) {
      for (int j = 0; j < v_rate; j++) {
        for (int k = 0; k < v_rate; k++) {
          const int child_z = z * v_rate + k;
          if (limit_z) {
            const T z0 = static_cast<T>(child_z) * child_res;
            const T z1 = z0 + child_res;
            if (!(z1 > min_z && z0 <= max_z)) {
              continue;
            }
          }
          for (int l = 0; l < num_division.x(); l++) {
            for (int m = 0; m < num_division.y(); m++) {
              for (int n = 0; n < num_division.z(); n++) {
                b.emplace_back(DiscreteTransformation(
                  0,
                  child_level,
                  x * v_rate + i,
                  y * v_rate + j,
                  child_z,
                  roll * num_division.x() + l,
                  pitch * num_division.y() + m,
                  yaw * num_division.z() + n));
              }
            }
          }
        }
      }
    }
  }

  std::vector<DiscreteTransformation> branch(const int child_level, const int v_rate, const Eigen::Vector3i& num_division) {
    std::vector<DiscreteTransformation> b;
    b.reserve(v_rate * v_rate * v_rate * num_division.x() * num_division.y() * num_division.z());
    for (int i = 0; i < v_rate; i++) {
      for (int j = 0; j < v_rate; j++) {
        for (int k = 0; k < v_rate; k++) {
          for (int l = 0; l < num_division.x(); l++) {
            for (int m = 0; m < num_division.y(); m++) {
              for (int n = 0; n < num_division.z(); n++) {
                b.emplace_back(DiscreteTransformation(
                  0,
                  child_level,
                  x * v_rate + i,
                  y * v_rate + j,
                  z * v_rate + k,
                  roll * num_division.x() + l,
                  pitch * num_division.y() + m,
                  yaw * num_division.z() + n));
              }
            }
          }
        }
      }
    }
    return b;
  }

public:
  int score;
  int level;
  int x;
  int y;
  int z;
  int roll;
  int pitch;
  int yaw;
};
