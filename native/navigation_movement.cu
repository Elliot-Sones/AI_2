#include "navigation_movement.h"

#include <cuda_runtime.h>

__global__ static void ai2_movement_kernel(const AI2MovementConfig config,
                                           const AI2MovementState *before,
                                           const AI2MovementInput *input,
                                           AI2MovementResult *result,
                                           int32_t *errors,
                                           int lanes) {
    const int lane = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (lane >= lanes) return;
    AI2MovementResult local;
    const int status = ai2_movement_step_core(&config, &before[lane], &input[lane], &local);
    errors[lane] = status;
    if (status == 0) {
        result[lane] = local;
    }
}

extern "C" int ai2_movement_step_cuda(const AI2MovementConfig *config,
                                       const AI2MovementState *before,
                                       const AI2MovementInput *input,
                                       AI2MovementResult *result,
                                       int32_t *errors,
                                       int lanes,
                                       void *stream) {
    if (!config || !before || !input || !result || !errors || lanes <= 0) {
        return AI2_MOVEMENT_ERR_INVALID;
    }
    const int threads = 256;
    const int blocks = (lanes + threads - 1) / threads;
    ai2_movement_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(*config, before, input, result, errors, lanes);
    return (int)cudaGetLastError();
}
