// Bonsai Q2_0_g128 W2A8 kernel.
//
// A:  int8[M, K]                     per-token symmetric A8
// B:  uint8[N, K/4]                  natural-order 2-bit codes
//                                      code 1/2/3 -> ternary -1/0/+1
// ws: bf16[N, K/128]                 per-output-row, per-K-group absmax
// s:  float[M]                       activation multiplier (127 / absmax)
// C:  bf16[M, N]
//
// Unlike BitNet's public W2A8 kernel, the weight scale is applied inside
// every 128-wide K group before the groups are accumulated.  This matches
// PrismML Bonsai's Q2_0_g128 training/deployment grid exactly.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>

constexpr int T2_GROUP_SIZE = 128;

__device__ __forceinline__ void decode_i2_natural(uint32_t packed,
                                                   signed char *out) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i] = (signed char)((int)((packed >> (2 * i)) & 0x3u) - 2);
    }
}

template <int M, int N, int K, int KBS, int NT>
__global__ void __launch_bounds__(KBS * NT)
t2g128_mfused(int8_t * __restrict__ A,
              int8_t * __restrict__ B,
              __nv_bfloat16 * __restrict__ C,
              float * __restrict__ s,
              __nv_bfloat16 * __restrict__ ws) {
    constexpr int KPL = 16;                 // one packed uint32 per lane
    constexpr int K_PER_ITER = KPL * KBS;
    static_assert(KBS >= 8 && KBS <= 32 && (KBS & (KBS - 1)) == 0,
                  "KBS must be 8, 16, or 32");
    static_assert(K_PER_ITER % T2_GROUP_SIZE == 0,
                  "iteration must cover whole g128 groups");
    static_assert(K % K_PER_ITER == 0, "K must divide the K tile");
    static_assert(KBS * NT <= 1024, "invalid CUDA block size");
    constexpr int GROUPS_PER_ITER = K_PER_ITER / T2_GROUP_SIZE;
    constexpr int OUTER_ITERS = K / K_PER_ITER;

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;
    const int n = (int)blockIdx.x * NT + ty;
    const int8_t *wrow = B + n * (K / 4);
    const __nv_bfloat16 *swrow = ws + n * (K / T2_GROUP_SIZE);
    const int group_in_iter = tx >> 3;       // 8 lanes x 16 = one g128

    float acc[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;

    signed char wdec[KPL];
    signed char alocal[KPL];

    #pragma unroll
    for (int ko = 0; ko < OUTER_ITERS; ++ko) {
        const int ks = ko * K_PER_ITER + tx * KPL;
        const uint32_t pack = *reinterpret_cast<const uint32_t *>(wrow + (ks >> 2));
        decode_i2_natural(pack, wdec);
        const float wg = __bfloat162float(
            swrow[ko * GROUPS_PER_ITER + group_in_iter]);

        #pragma unroll
        for (int m = 0; m < M; ++m) {
            *reinterpret_cast<int4 *>(alocal) =
                *reinterpret_cast<const int4 *>(A + m * K + ks);
            int dot = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                dot = __dp4a(*reinterpret_cast<int *>(alocal + i * 4),
                             *reinterpret_cast<int *>(wdec + i * 4), dot);
            }
            acc[m] += (float)dot * wg;
        }
    }

    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float r = acc[m];
        #pragma unroll
        for (int off = KBS / 2; off > 0; off >>= 1) {
            r += __shfl_xor_sync(0xffffffff, r, off, KBS);
        }
        if (tx == 0) C[m * N + n] = __float2bfloat16(r / s[m]);
    }
}

#define LAUNCH_CFG(M_, N_, K_, KBS_, NT_)                                  \
    do {                                                                     \
        t2g128_mfused<M_, N_, K_, KBS_, NT_>                                \
            <<<dim3((N_) / (NT_), 1, 1), dim3((KBS_), (NT_), 1), 0, stream>>>(\
                A, B, C, s, ws);                                             \
        return;                                                              \
    } while (0)

#define DISPATCH_SHAPES(M_)                                                  \
    do {                                                                     \
        if (K == 2048 && N == 1024)   LAUNCH_CFG(M_, 1024, 2048, 16, 8);    \
        if (K == 2048 && N == 2048)   LAUNCH_CFG(M_, 2048, 2048, 32, 8);    \
        if (K == 2048 && N == 4096)   LAUNCH_CFG(M_, 4096, 2048, 16, 16);   \
        if (K == 2048 && N == 6144)   LAUNCH_CFG(M_, 6144, 2048, 8, 32);    \
        if (K == 2048 && N == 12288)  LAUNCH_CFG(M_, 12288, 2048, 8, 32);   \
        if (K == 6144 && N == 2048)   LAUNCH_CFG(M_, 2048, 6144, 32, 8);    \
        if (K == 2048 && N == 151936) LAUNCH_CFG(M_, 151936, 2048, 8, 32);  \
    } while (0)

#define DISPATCH_SHAPES_4B(M_)                                               \
    do {                                                                     \
        if (K == 2560 && N == 1024)   LAUNCH_CFG(M_, 1024, 2560, 16, 8);    \
        if (K == 2560 && N == 6144)   LAUNCH_CFG(M_, 6144, 2560, 8, 32);    \
        if (K == 2560 && N == 19456)  LAUNCH_CFG(M_, 19456, 2560, 8, 32);   \
        if (K == 4096 && N == 2560)   LAUNCH_CFG(M_, 2560, 4096, 32, 8);    \
        if (K == 9728 && N == 2560)   LAUNCH_CFG(M_, 2560, 9728, 32, 8);    \
        if (K == 2560 && N == 151936) LAUNCH_CFG(M_, 151936, 2560, 8, 32);  \
    } while (0)

extern "C" void t2g128_int8xint2_linear(
    int8_t *A, int8_t *B, __nv_bfloat16 *C,
    float *s, __nv_bfloat16 *ws,
    int M, int N, int K, cudaStream_t stream) {
    // Qwen3-4B fused projection shapes for AR and D=8 draft/verify widths.
    if (M == 1)  { DISPATCH_SHAPES_4B(1);  }
    if (M == 8)  { DISPATCH_SHAPES_4B(8);  }
    if (M == 9)  { DISPATCH_SHAPES_4B(9);  }
    if (M == 10) { DISPATCH_SHAPES_4B(10); }
    if (M == 12) { DISPATCH_SHAPES_4B(12); }
    if (M == 16) { DISPATCH_SHAPES_4B(16); }
    if (M == 17) { DISPATCH_SHAPES_4B(17); }
    if (M == 1)  { DISPATCH_SHAPES(1);  }
    if (M == 4)  { DISPATCH_SHAPES(4);  }
    if (M == 5)  { DISPATCH_SHAPES(5);  }
    if (M == 6)  { DISPATCH_SHAPES(6);  }
    if (M == 7)  { DISPATCH_SHAPES(7);  }
    if (M == 8)  { DISPATCH_SHAPES(8);  }
    if (M == 9)  { DISPATCH_SHAPES(9);  }
    if (M == 10) { DISPATCH_SHAPES(10); }
    if (M == 12) { DISPATCH_SHAPES(12); }
    if (M == 16) { DISPATCH_SHAPES(16); }
    if (M == 17) { DISPATCH_SHAPES(17); }
    fprintf(stderr,
            "[t2g128] unsupported shape M=%d N=%d K=%d\n", M, N, K);
}
