/* test_kernels.c  -  run:  make test
 * Each test launches a kernel from kernels.h the way the real code does:
 * compute blocks from n and a chosen block size, LAUNCH, then check results.
 */
#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int failures = 0, shown = 0;
/* print at most 5 failures per test so a TODO kernel does not flood the terminal */
#define CHECK(cond, ...) do { if (!(cond)) { failures++; if (shown < 5) { printf("  FAIL: " __VA_ARGS__); printf("\n"); } else if (shown == 5) printf("  ... more failures hidden\n"); shown++; } } while (0)
#define BEGIN(name) do { shown = 0; printf(name "\n"); } while (0)
static int blocks_for(int n, int threads) { return (n + threads - 1) / threads; }

static void test_multiply(void) {
    BEGIN("1. multiply_elements");
    enum { N = 4096, T = 64 };
    float *a, *r, *p; DEVICE_ALLOC(a, N * sizeof(float)); DEVICE_ALLOC(r, N * sizeof(float)); DEVICE_ALLOC(p, N * sizeof(float));
    float ha[N], hr[N], hp[N];
    for (int i = 0; i < N; ++i) { ha[i] = (float)i; hr[i] = 0.5f; }
    TO_DEVICE(a, ha, sizeof ha); TO_DEVICE(r, hr, sizeof hr);
    LAUNCH(multiply_elements, blocks_for(N, T), T, a, r, p, N);
    TO_HOST(hp, p, sizeof hp);
    for (int i = 0; i < N; ++i) CHECK(hp[i] == ha[i] * 0.5f, "element %d: got %g want %g", i, hp[i], ha[i] * 0.5f);
    printf("  launched %d blocks x %d threads for %d elements\n", blocks_for(N, T), T, N);
    DEVICE_FREE(a); DEVICE_FREE(r); DEVICE_FREE(p);
}

static void test_clamp(void) {
    BEGIN("2. clamp_elements");
    enum { N = 1000, T = 128 };   /* 1000 is not a multiple of 128: the guard matters */
    float *in, *out; DEVICE_ALLOC(in, N * sizeof(float)); DEVICE_ALLOC(out, N * sizeof(float));
    float hin[N], hout[N];
    for (int i = 0; i < N; ++i) hin[i] = (float)i / 500.0f;   /* 0.0 .. 2.0 */
    TO_DEVICE(in, hin, sizeof hin);
    LAUNCH(clamp_elements, blocks_for(N, T), T, in, out, 0.8f, 1.2f, N);
    TO_HOST(hout, out, sizeof hout);
    for (int i = 0; i < N; ++i) {
        float want = hin[i] < 0.8f ? 0.8f : hin[i] > 1.2f ? 1.2f : hin[i];
        CHECK(fabsf(hout[i] - want) < 1e-6f, "element %d: got %g want %g", i, hout[i], want);
    }
    printf("  %d blocks x %d threads = %d threads for %d elements; %d threads hit the guard\n",
           blocks_for(N, T), T, blocks_for(N, T) * T, N, blocks_for(N, T) * T - N);
    DEVICE_FREE(in); DEVICE_FREE(out);
}

static void test_sum_actions(void) {
    BEGIN("3. sum_actions_per_player");
    enum { G = 2048, T = 64 };
    float *acts, *out; DEVICE_ALLOC(acts, G * 20 * sizeof(float)); DEVICE_ALLOC(out, G * 2 * sizeof(float));
    static float h[G * 20], hout[G * 2];
    for (int g = 0; g < G; ++g) for (int p = 0; p < 2; ++p) for (int k = 0; k < 10; ++k) h[g * 20 + p * 10 + k] = (float)(p + 1);  /* p0 sums to 10, p1 to 20 */
    TO_DEVICE(acts, h, sizeof h);
    LAUNCH(sum_actions_per_player, blocks_for(G, T), T, acts, out, G);
    TO_HOST(hout, out, sizeof hout);
    for (int g = 0; g < G; ++g) { CHECK(hout[g * 2] == 10.0f, "game %d player 0: got %g want 10", g, hout[g * 2]); CHECK(hout[g * 2 + 1] == 20.0f, "game %d player 1: got %g want 20", g, hout[g * 2 + 1]); }
    printf("  %d games -> %d blocks of %d, i.e. %d warps\n", G, blocks_for(G, T), T, blocks_for(G, T) * T / 32);
    DEVICE_FREE(acts); DEVICE_FREE(out);
}

static void run_fighters(const char *label, int *hstate, int n) {
    int *state, *hits; float *x, *y;
    DEVICE_ALLOC(state, n * sizeof(int)); DEVICE_ALLOC(hits, n * sizeof(int)); DEVICE_ALLOC(x, n * sizeof(float)); DEVICE_ALLOC(y, n * sizeof(float));
    int *hh = calloc(n, sizeof(int)); float *hx = calloc(n, sizeof(float)); float *hy = calloc(n, sizeof(float));
    int *orig = malloc(n * sizeof(int)); memcpy(orig, hstate, n * sizeof(int));
    TO_DEVICE(state, hstate, n * sizeof(int)); TO_DEVICE(hits, hh, n * sizeof(int)); TO_DEVICE(x, hx, n * sizeof(float)); TO_DEVICE(y, hy, n * sizeof(float));
    shim_reset();
    LAUNCH(fighter_step, blocks_for(n, 64), 64, state, x, y, hits, n);
    shim_report(label);
    TO_HOST(hstate, state, n * sizeof(int)); TO_HOST(hh, hits, n * sizeof(int)); TO_HOST(hx, x, n * sizeof(float)); TO_HOST(hy, y, n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        switch (orig[i]) {
            case 0: CHECK(hx[i] == 0 && hy[i] == 0 && hh[i] == 0 && hstate[i] == 0, "idle %d changed", i); break;
            case 1: CHECK(hx[i] == 1.0f && hstate[i] == 1, "run %d: x=%g state=%d", i, hx[i], hstate[i]); break;
            case 2: CHECK(hy[i] == 2.0f && hstate[i] == 0, "jump %d: y=%g state=%d (should become idle)", i, hy[i], hstate[i]); break;
            case 3: CHECK(hh[i] == 1 && hstate[i] == 3, "attack %d: hits=%d", i, hh[i]); break;
        }
    }
    free(hh); free(hx); free(hy); free(orig);
    DEVICE_FREE(state); DEVICE_FREE(hits); DEVICE_FREE(x); DEVICE_FREE(y);
}

static void test_fighters(void) {
    BEGIN("4. fighter_step (divergence)");
    enum { N = 2048 };
    static int shuffled[N], sorted[N];
    unsigned seed = 7;
    for (int i = 0; i < N; ++i) { seed = seed * 1103515245u + 12345u; shuffled[i] = (int)((seed >> 16) % 4); }
    for (int i = 0; i < N; ++i) sorted[i] = i * 4 / N;   /* 512 idle, 512 run, 512 jump, 512 attack, grouped */
    run_fighters("shuffled states", shuffled, N);
    run_fighters("sorted states  ", sorted, N);
    printf("  same kernel, same total work. Fewer passes per warp = less divergence.\n");
}

static void test_sum(void) {
    BEGIN("5. sum_naive");
    enum { N = 65536, T = 256 };
    float *in, *out; DEVICE_ALLOC(in, N * sizeof(float)); DEVICE_ALLOC(out, sizeof(float));
    static float h[N]; float zero = 0.0f, got;
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    TO_DEVICE(in, h, sizeof h); TO_DEVICE(out, &zero, sizeof zero);
    LAUNCH(sum_naive, blocks_for(N, T), T, in, out, N);
    TO_HOST(&got, out, sizeof got);
    CHECK(got == (float)N, "sum: got %g want %d", got, N);
    printf("  %d threads all wrote out[0]. Passing here does NOT mean correct on a GPU. Read the comment in kernels.h.\n", N);
    DEVICE_FREE(in); DEVICE_FREE(out);
}

int main(void) {
#ifdef __CUDACC__
    printf("backend: CUDA (real GPU)\n\n");
#else
    printf("backend: CPU shim (one core, threads run one after another, warps simulated)\n\n");
#endif
    test_multiply(); test_clamp(); test_sum_actions(); test_fighters(); test_sum();
    printf("\n%s  (%d failing checks)\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
