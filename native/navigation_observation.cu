#include "navigation_observation.h"
#include <cuda_runtime.h>
__global__ void ai2_observation_kernel(const double *p, const double *s, const double *platform,
                                      float *out, int *errors, int lanes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lanes) return;
    errors[i] = ai2_obs_pack(p + i * 32, s + i * 12, platform + i * 4, out + i * 64);
    if (errors[i]) for (int j = 0; j < 64; ++j) out[i * 64 + j] = NAN;
}
extern "C" int ai2_observation_cuda(const double *p, const double *s, const double *platform,
                                   float *out, int *errors, size_t lanes, void *stream) {
    if (!p || !s || !platform || !out || !errors || lanes == 0 || lanes > 4096) return (int)cudaErrorInvalidValue;
    ai2_observation_kernel<<<(lanes + 127) / 128, 128, 0, (cudaStream_t)stream>>>(p, s, platform, out, errors, (int)lanes);
    return (int)cudaGetLastError();
}
