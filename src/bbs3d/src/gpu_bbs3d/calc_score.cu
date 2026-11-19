#include <gpu_bbs3d/bbs3d.cuh>
#include <gpu_bbs3d/voxelmaps.cuh>
#include <gpu_bbs3d/stream_manager/check_error.cuh>
#include <discrete_transformation/discrete_transformation.hpp>

namespace lightning::gpu {
__global__ void calc_scores_kernel(
  Eigen::Vector4i* const* multi_buckets_ptrs,
  const VoxelMapInfo* voxelmap_info_ptr,
  const AngularInfoGpu* d_ang_info_vec_ptr,
  DiscreteTransformation<float>* trans_ptr,
  int num_levels,
  size_t index_size,
  const Point3fGpu* points_ptr,
  size_t num_points) {
  const size_t pose_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (pose_index >= index_size) {
    return;
  }

  DiscreteTransformation<float>& trans = trans_ptr[pose_index];
  const int level = trans.level;
  // Guard invalid level before any per-level array indexing
  if (level < 0 || level >= num_levels) {
    trans.score = 0;
    return;
  }
  const VoxelMapInfo& voxelmap_info = voxelmap_info_ptr[level];
  const AngularInfoGpu& ang_info = d_ang_info_vec_ptr[level];
  const Eigen::Vector4i* buckets = multi_buckets_ptrs[level];
  // Safety guards: if voxel map has no buckets or pointer is null, skip scoring to avoid illegal memory access
  if (voxelmap_info.num_buckets <= 0 || buckets == nullptr) {
    trans.score = 0;
    return;
  }

  int score = 0;
  for (size_t i = 0; i < num_points; i++) {
    const Point3fGpu& point = points_ptr[i];

    const float tx = trans.x * voxelmap_info.res;
    const float ty = trans.y * voxelmap_info.res;
    const float tz = trans.z * voxelmap_info.res;
    const float roll = trans.roll * ang_info.rpy_res_x + ang_info.min_rpy_x;
    const float pitch = trans.pitch * ang_info.rpy_res_y + ang_info.min_rpy_y;
    const float yaw = trans.yaw * ang_info.rpy_res_z + ang_info.min_rpy_z;
    const float cr = cosf(roll);
    const float sr = sinf(roll);
    const float cp = cosf(pitch);
    const float sp = sinf(pitch);
    const float cyaw = cosf(yaw);
    const float sy = sinf(yaw);
    const float rx = (cyaw * cp) * point.x + (cyaw * sp * sr - sy * cr) * point.y + (cyaw * sp * cr + sy * sr) * point.z;
    const float ry = (sy * cp) * point.x + (sy * sp * sr + cyaw * cr) * point.y + (sy * sp * cr - cyaw * sr) * point.z;
    const float rz = (-sp) * point.x + (cp * sr) * point.y + (cp * cr) * point.z;
    const float transed_x = rx + tx;
    const float transed_y = ry + ty;
    const float transed_z = rz + tz;

    // coord to hash
    const int coord_x = static_cast<int>(floorf(transed_x * voxelmap_info.inv_res));
    const int coord_y = static_cast<int>(floorf(transed_y * voxelmap_info.inv_res));
    const int coord_z = static_cast<int>(floorf(transed_z * voxelmap_info.inv_res));
    const std::uint32_t cx = static_cast<std::uint32_t>(coord_x);
    const std::uint32_t cy = static_cast<std::uint32_t>(coord_y);
    const std::uint32_t cz = static_cast<std::uint32_t>(coord_z);
    const std::uint32_t hash = (cx * 73856093u) ^ (cy * 19349669u) ^ (cz * 83492791u);

    // open addressing
    for (int j = 0; j < voxelmap_info.max_bucket_scan_count; j++) {
      const std::uint32_t bucket_index = (hash + j) % voxelmap_info.num_buckets;
      const Eigen::Vector4i bucket = buckets[bucket_index];

      if (bucket.w() == 0) {
        break;
      }

      if (bucket.x() != coord_x || bucket.y() != coord_y || bucket.z() != coord_z) {
        continue;
      }

      score++;
      break;
    }
  }
  trans.score = score;
}

std::vector<DiscreteTransformation<float>> BBS3D::calc_scores(
  const std::vector<DiscreteTransformation<float>>& h_transset,
  thrust::device_vector<AngularInfoGpu>& d_ang_info_vec) {
  size_t transset_size = h_transset.size();
  thrust::device_vector<DiscreteTransformation<float>> d_transset(h_transset.begin(), h_transset.end());

  // Readiness guards: if any prerequisite buffers are empty, skip GPU launch to avoid race-time illegal access
  if (transset_size == 0 ||
      voxelmaps_ptr_ == nullptr ||
      voxelmaps_ptr_->d_voxelmaps_info_.size() == 0 ||
      voxelmaps_ptr_->d_multi_buckets_ptrs_.size() == 0 ||
      src_points_.size() == 0) {
    return std::vector<DiscreteTransformation<float>>();
  }

  const size_t block_size = 32;
  const size_t num_blocks = (transset_size + (block_size - 1)) / block_size;

  calc_scores_kernel<<<num_blocks, block_size, 0, stream>>>(
    thrust::raw_pointer_cast(voxelmaps_ptr_->d_multi_buckets_ptrs_.data()),
    thrust::raw_pointer_cast(voxelmaps_ptr_->d_voxelmaps_info_.data()),
    thrust::raw_pointer_cast(d_ang_info_vec.data()),
    thrust::raw_pointer_cast(d_transset.data()),
    static_cast<int>(voxelmaps_ptr_->d_voxelmaps_info_.size()),
    transset_size,
    thrust::raw_pointer_cast(d_src_points_.data()),
    src_points_.size());

  const cudaError_t launch_error = cudaPeekAtLastError();
  if (launch_error != cudaSuccess) {
    std::cout << "[GPU Debug] calc_scores_kernel launch error: " << cudaGetErrorString(launch_error) << std::endl;
  }

  std::vector<DiscreteTransformation<float>> h_output(transset_size);
  check_error << cudaMemcpyAsync(
    h_output.data(),
    thrust::raw_pointer_cast(d_transset.data()),
    sizeof(DiscreteTransformation<float>) * transset_size,
    cudaMemcpyDeviceToHost,
    stream);

  check_error << cudaStreamSynchronize(stream);
  const cudaError_t sync_error = cudaGetLastError();
  if (sync_error != cudaSuccess) {
    std::cout << "[GPU Debug] calc_scores_kernel sync error: " << cudaGetErrorString(sync_error) << std::endl;
  }
  return h_output;
}
}  // namespace gpu