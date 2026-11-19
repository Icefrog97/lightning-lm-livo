#pragma once

#include <cstdint>
#include <functional>

namespace lightning {

constexpr int64_t kVoxelMapHashP = 116101;
constexpr int64_t kVoxelMapMaxN = 10000000000;

class VoxelLocation {
   public:
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;

    VoxelLocation(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

    bool operator==(const VoxelLocation& other) const { return x == other.x && y == other.y && z == other.z; }
};

}  // namespace lightning

namespace std {

template <>
struct hash<lightning::VoxelLocation> {
    size_t operator()(const lightning::VoxelLocation& loc) const {
        const int64_t hash_value =
            ((((loc.z) * lightning::kVoxelMapHashP) % lightning::kVoxelMapMaxN + loc.y) *
                 lightning::kVoxelMapHashP) %
                lightning::kVoxelMapMaxN +
            loc.x;
        return static_cast<size_t>(hash_value);
    }
};

}  // namespace std
