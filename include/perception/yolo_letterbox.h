#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

// Letterbox the packed BGR uint8 image into a YOLO NCHW float buffer (RGB, /255, gray 114).
// src is a device pointer, tightly packed (step = src_w * 3). dst is device NCHW.
void LaunchYoloLetterbox(const uint8_t *src, int src_w, int src_h, float *dst, int dst_size, int pad_w,
                         int pad_h, int resized_w, int resized_h, cudaStream_t stream);
