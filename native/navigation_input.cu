#include <cuda_runtime.h>
#include <stddef.h>
#include "navigation_input.h"

/* Low-level device-pointer API: caller owns contiguous float64 [N,10] actions,
 * int32 [N,35] before/after, int32 [N] status, lifetimes and current-device
 * selection. before/after may coincide exactly; other overlap is unsupported.
 * The return value reports launch errors; status reports per-world input errors.
 * No host copies, device allocation, or stream synchronization happen here. */
__global__ static void ai2_input_kernel(const double *actions, const int32_t *before,
                                        int32_t *after, int32_t *status, int lanes) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;
    status[lane] = ai2_input_step_core(
        actions + (size_t)lane * AI2_INPUT_BUTTONS,
        before + (size_t)lane * AI2_INPUT_WIDTH,
        after + (size_t)lane * AI2_INPUT_WIDTH);
}

extern "C" int ai2_input_step_cuda(const double *actions, const int32_t *before,
                                    int32_t *after, int32_t *status, int lanes,
                                    cudaStream_t stream) {
    if (lanes < 0) return (int)cudaErrorInvalidValue;
    if (lanes == 0) return (int)cudaSuccess;
    if (!actions || !before || !after || !status) return (int)cudaErrorInvalidValue;
    ai2_input_kernel<<<1 + (lanes - 1) / 128, 128, 0, stream>>>(
        actions, before, after, status, lanes);
    return (int)cudaGetLastError();
}
