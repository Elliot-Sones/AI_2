/* trace.c  -  the smallest possible kernel, with every call printed.
 * build and run:   make trace
 */
#include "cuda_shim.h"
#include <stdio.h>

/* THE KERNEL. One function. It does not know how many times it will be called. */
__global__ void hello(int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;     /* who am I? */
    if (i >= n) return;                                  /* am I past the end? */
    out[i] = i * 10;                                     /* do my one piece of work */
    printf("  block %u, thread %u  ->  i = %d   (I own out[%d])\n", blockIdx.x, threadIdx.x, i, i);
}

int main(void) {
    int out[8] = {0};
    printf("launch hello<<<2 blocks, 4 threads>>> for n = 7\n");
    LAUNCH(hello, 2, 4, out, 7);
    printf("result: ");
    for (int i = 0; i < 8; ++i) printf("%d ", out[i]);
    printf("\n");
    return 0;
}
