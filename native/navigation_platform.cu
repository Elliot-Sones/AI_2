#include "navigation_platform.h"

#include <cuda_runtime.h>
#include <stdint.h>

__global__ void navigation_platform_step_kernel(
    const double *positions_xy,
    const double *waypoint1_xy,
    const double *waypoint2_xy,
    const uint8_t *moving_to_w2,
    double *velocities_xy,
    uint8_t *next_moving_to_w2,
    int lanes
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) {
        return;
    }
    const int offset = lane * 2;
    ai2_navigation_platform_step_math(
        positions_xy[offset],
        positions_xy[offset + 1],
        waypoint1_xy[offset],
        waypoint1_xy[offset + 1],
        waypoint2_xy[offset],
        waypoint2_xy[offset + 1],
        moving_to_w2[lane],
        &velocities_xy[offset],
        &velocities_xy[offset + 1],
        &next_moving_to_w2[lane]
    );
}

__global__ void navigation_platform_step_kinematic_kernel(
    const double *positions_xy,
    const double *waypoint1_xy,
    const double *waypoint2_xy,
    const uint8_t *moving_to_w2,
    double dt,
    double *next_positions_xy,
    double *velocities_xy,
    uint8_t *next_moving_to_w2,
    int lanes
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) {
        return;
    }
    const int offset = lane * 2;
    ai2_navigation_platform_step_kinematic_math(
        positions_xy[offset],
        positions_xy[offset + 1],
        waypoint1_xy[offset],
        waypoint1_xy[offset + 1],
        waypoint2_xy[offset],
        waypoint2_xy[offset + 1],
        moving_to_w2[lane],
        dt,
        &next_positions_xy[offset],
        &next_positions_xy[offset + 1],
        &velocities_xy[offset],
        &velocities_xy[offset + 1],
        &next_moving_to_w2[lane]
    );
}

extern "C" int navigation_platform_step_cuda(
    const double *positions_xy,
    const double *waypoint1_xy,
    const double *waypoint2_xy,
    const uint8_t *moving_to_w2,
    double *velocities_xy,
    uint8_t *next_moving_to_w2,
    int lanes,
    void *stream_ptr
) {
    if (lanes < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (lanes == 0) {
        return (int)cudaSuccess;
    }

    const int threads = 256;
    const int blocks = (lanes + threads - 1) / threads;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    navigation_platform_step_kernel<<<blocks, threads, 0, stream>>>(
        positions_xy,
        waypoint1_xy,
        waypoint2_xy,
        moving_to_w2,
        velocities_xy,
        next_moving_to_w2,
        lanes
    );
    return (int)cudaGetLastError();
}

extern "C" int navigation_platform_step_kinematic_cuda(
    const double *positions_xy,
    const double *waypoint1_xy,
    const double *waypoint2_xy,
    const uint8_t *moving_to_w2,
    double dt,
    double *next_positions_xy,
    double *velocities_xy,
    uint8_t *next_moving_to_w2,
    int lanes,
    void *stream_ptr
) {
    if (lanes < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (lanes == 0) {
        return (int)cudaSuccess;
    }

    const int threads = 256;
    const int blocks = (lanes + threads - 1) / threads;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    navigation_platform_step_kinematic_kernel<<<blocks, threads, 0, stream>>>(
        positions_xy,
        waypoint1_xy,
        waypoint2_xy,
        moving_to_w2,
        dt,
        next_positions_xy,
        velocities_xy,
        next_moving_to_w2,
        lanes
    );
    return (int)cudaGetLastError();
}
