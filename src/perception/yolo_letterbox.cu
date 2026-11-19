#include "perception/yolo_letterbox.h"

namespace {

__device__ __forceinline__ float SampleBilinear(const uint8_t *src, int w, int h, float x, float y, int c) {
    x = fminf(fmaxf(x, 0.f), static_cast<float>(w - 1));
    y = fminf(fmaxf(y, 0.f), static_cast<float>(h - 1));
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = min(x0 + 1, w - 1);
    const int y1 = min(y0 + 1, h - 1);
    const float ax = x - static_cast<float>(x0);
    const float ay = y - static_cast<float>(y0);
    const float v00 = static_cast<float>(src[(y0 * w + x0) * 3 + c]);
    const float v01 = static_cast<float>(src[(y0 * w + x1) * 3 + c]);
    const float v10 = static_cast<float>(src[(y1 * w + x0) * 3 + c]);
    const float v11 = static_cast<float>(src[(y1 * w + x1) * 3 + c]);
    const float v0 = v00 + (v01 - v00) * ax;
    const float v1 = v10 + (v11 - v10) * ax;
    return (v0 + (v1 - v0) * ay) * (1.f / 255.f);
}

__global__ void LetterboxBgrToNchwKernel(const uint8_t *src, int src_w, int src_h, float *dst, int dst_size,
                                         int pad_w, int pad_h, int resized_w, int resized_h) {
    const int n = dst_size * dst_size;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    const int x = idx % dst_size;
    const int y = idx / dst_size;
    const int lx = x - pad_w;
    const int ly = y - pad_h;
    constexpr float kPad = 114.f / 255.f;
    float r = kPad;
    float g = kPad;
    float b = kPad;
    if (src_w > 0 && src_h > 0 && resized_w > 0 && resized_h > 0 && lx >= 0 && ly >= 0 && lx < resized_w &&
        ly < resized_h) {
        // align_corners=False: pixel center (i + 0.5) maps by src_size / resized_size.
        const float sx = (static_cast<float>(lx) + 0.5f) * static_cast<float>(src_w) / static_cast<float>(resized_w) -
                         0.5f;
        const float sy = (static_cast<float>(ly) + 0.5f) * static_cast<float>(src_h) / static_cast<float>(resized_h) -
                         0.5f;
        b = SampleBilinear(src, src_w, src_h, sx, sy, 0);
        g = SampleBilinear(src, src_w, src_h, sx, sy, 1);
        r = SampleBilinear(src, src_w, src_h, sx, sy, 2);
    }
    dst[idx] = r;
    dst[n + idx] = g;
    dst[2 * n + idx] = b;
}

}  // namespace

void LaunchYoloLetterbox(const uint8_t *src, int src_w, int src_h, float *dst, int dst_size, int pad_w, int pad_h,
                         int resized_w, int resized_h, cudaStream_t stream) {
    if (dst == nullptr || dst_size <= 0)
        return;
    const int n = dst_size * dst_size;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    LetterboxBgrToNchwKernel<<<blocks, threads, 0, stream>>>(src, src_w, src_h, dst, dst_size, pad_w, pad_h,
                                                             resized_w, resized_h);
}
