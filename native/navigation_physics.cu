#include "navigation_physics.h"

#include <cuda_runtime.h>
#include <stdint.h>

__global__ void navigation_physics_step_kernel(
    const double *states,
    double *out,
    int lanes,
    double gravity_x,
    double gravity_y,
    double damping,
    double dt,
    int *errors
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) {
        return;
    }
    const int offset = lane * (int)AI2_PHYSICS_BODY_STATE_FIELDS;
    const int result = ai2_physics_step_body_math(
        states + offset,
        gravity_x,
        gravity_y,
        damping,
        dt,
        out + offset
    );
    errors[lane] = result;
}

extern "C" int navigation_physics_step_batch_cuda(
    const double *states,
    double *out,
    int lanes,
    double gravity_x,
    double gravity_y,
    double damping,
    double dt,
    int *errors,
    void *stream_ptr
) {
    if (states == 0 || out == 0 || errors == 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (lanes < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (lanes == 0) {
        return (int)cudaSuccess;
    }

    const int threads = 256;
    const int blocks = (lanes + threads - 1) / threads;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    navigation_physics_step_kernel<<<blocks, threads, 0, stream>>>(
        states,
        out,
        lanes,
        gravity_x,
        gravity_y,
        damping,
        dt,
        errors
    );
    return (int)cudaGetLastError();
}
