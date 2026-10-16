// Q1_0_g128 (W1A8) kernel dispatch for Bonsai-1.7B.
// Shapes (per layer, after `prepare_merged_weights`):
//   K=2048, N=2048  (o_proj)
//   K=2048, N=4096  (merged QKV: q=2048, k=1024, v=1024 -- varseg)
//   K=2048, N=12288 (merged gate+up: each 6144)
//   K=6144, N=2048  (down_proj)
//   K=2048, N=151936 (lm_head; tied with embed_tokens but we expose a kernel
//                     dispatch anyway for completeness)
//
// M variants per shape:
//   M=1  (AR decode)
//   M=4..9 (SDLM block decode widths; D=4 -> {5..9}, D=8 -> {9})
//   M=10,12,16,17 (D=8 hybrid/spec pending-width set)
//
// Layout reminders:
//   A:  int8[M, K]
//   B:  int8[N, K/8]
//   ws: __nv_bfloat16[N, K/128]
//   s:  float[M]
//   C:  __nv_bfloat16[M, N]

#include "q1g128_kernels.h"
#include <cuda_bf16.h>
#include <stdio.h>


// =====================================================================
// W1A16: reconstruct binary weights from sign bits and group scales, then
// accumulate their products with BF16 activations in FP32 using CUDA-core FMA.
// =====================================================================
template <int M_, int N_, int K_, int KBS, int NT>
__global__ void __launch_bounds__(KBS * NT)
q1g128_kernel_w1a16(const __nv_bfloat16* __restrict__ A, const int8_t* __restrict__ B,
                    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ ws) {
    constexpr int KPL = 32;
    constexpr int K_PER = KPL * KBS;
    constexpr int GROUPS_PER = K_PER / 128;
    constexpr int OUTER = K_ / K_PER;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int n = blockIdx.x * NT + ty;
    const int8_t* w_row = B + n * (K_ / 8);
    const __nv_bfloat16* s_row = ws + n * (K_ / 128);
    float acc[M_];
    #pragma unroll
    for (int m = 0; m < M_; ++m) acc[m] = 0.0f;
    signed char B_dec[KPL]; int B_pack[1];
    const int gin = tx >> 2;
    #pragma unroll
    for (int ko = 0; ko < OUTER; ++ko) {
        const int ks = ko * K_PER + tx * KPL;
        B_pack[0] = *(int*)(w_row + (ks >> 3));
        decode_i1s_to_i8s(B_pack, B_dec);
        const float wg = __bfloat162float(s_row[ko * GROUPS_PER + gin]);
        #pragma unroll
        for (int m = 0; m < M_; ++m) {
            const __nv_bfloat16* a = A + m * K_ + ks;
            float4 a0 = *(const float4*)(a);
            float4 a1 = *(const float4*)(a + 8);
            float4 a2 = *(const float4*)(a + 16);
            float4 a3 = *(const float4*)(a + 24);
            const __nv_bfloat16* ap0 = (const __nv_bfloat16*)&a0;
            const __nv_bfloat16* ap1 = (const __nv_bfloat16*)&a1;
            const __nv_bfloat16* ap2 = (const __nv_bfloat16*)&a2;
            const __nv_bfloat16* ap3 = (const __nv_bfloat16*)&a3;
            float p = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                p += (float)B_dec[i]      * __bfloat162float(ap0[i]);
                p += (float)B_dec[i + 8]  * __bfloat162float(ap1[i]);
                p += (float)B_dec[i + 16] * __bfloat162float(ap2[i]);
                p += (float)B_dec[i + 24] * __bfloat162float(ap3[i]);
            }
            acc[m] += p * wg;
        }
    }
    #pragma unroll
    for (int m = 0; m < M_; ++m) {
        float r = acc[m];
        #pragma unroll
        for (int off = KBS / 2; off > 0; off >>= 1)
            r += __shfl_xor_sync(0xffffffff, r, off, KBS);
        if (tx == 0) C[m * N_ + n] = __float2bfloat16(r);
    }
}

#define W1A16_LAUNCH(M_)                                                     \
    do {                                                                      \
        q1g128_kernel_w1a16<M_, 151936, 2048, 4, 64>                          \
            <<<dim3(151936 / 64, 1, 1), dim3(4, 64, 1), 0, stream>>>(A, B, C, ws); \
        return;                                                               \
    } while (0)

extern "C" void q1g128_w1a16_linear(
    const __nv_bfloat16* A, const int8_t* B, __nv_bfloat16* C,
    const __nv_bfloat16* ws, int M, int N, int K, cudaStream_t stream)
{
    if (K == 2048 && N == 151936) {
        if (M == 1)  W1A16_LAUNCH(1);
        if (M == 4)  W1A16_LAUNCH(4);
        if (M == 5)  W1A16_LAUNCH(5);
        if (M == 6)  W1A16_LAUNCH(6);
        if (M == 7)  W1A16_LAUNCH(7);
        if (M == 8)  W1A16_LAUNCH(8);
        if (M == 9)  W1A16_LAUNCH(9);
        if (M == 10) W1A16_LAUNCH(10);
        if (M == 12) W1A16_LAUNCH(12);
        if (M == 16) W1A16_LAUNCH(16);
        if (M == 17) W1A16_LAUNCH(17);
    }
    fprintf(stderr,
            "[q1g128_w1a16_linear] unsupported (M=%d, N=%d, K=%d)\n", M, N, K);
}


// =====================================================================
// Public dispatch entrypoint
// =====================================================================
extern "C" void q1g128_int8xint1_linear(
    int8_t* A, int8_t* B, __nv_bfloat16* C,
    float* s, __nv_bfloat16* ws,
    int M, int N, int K, cudaStream_t stream)
{
    // --------------------------- M=1 paths ---------------------------
    if (M == 1) {
        // (K=2048, N=2048) q_proj / o_proj
        if (K == 2048 && N == 2048) {
            q1g128_kernel_M1<2048, 2048, 8, 16>
                <<<dim3(128, 1, 1), dim3(8, 16, 1), 0, stream>>>(A, B, C, s, ws);
            return;
        }
        // (K=2048, N=1024) k_proj / v_proj
        if (K == 2048 && N == 1024) {
            q1g128_kernel_M1<1024, 2048, 8, 16>
                <<<dim3(64, 1, 1), dim3(8, 16, 1), 0, stream>>>(A, B, C, s, ws);
            return;
        }
        // (K=2048, N=6144) gate_proj / up_proj
        if (K == 2048 && N == 6144) {
            q1g128_kernel_M1<6144, 2048, 8, 16>
                <<<dim3(384, 1, 1), dim3(8, 16, 1), 0, stream>>>(A, B, C, s, ws);
            return;
        }
        // (K=2048, N=4096) merged QKV
        if (K == 2048 && N == 4096) {
            q1g128_kernel_M1<4096, 2048, 8, 16>
                <<<dim3(256, 1, 1), dim3(8, 16, 1), 0, stream>>>(A, B, C, s, ws);
            return;
        }
        // (K=2048, N=12288) merged gate+up
        if (K == 2048 && N == 12288) {
            q1g128_kernel_M1<12288, 2048, 8, 16>
                <<<dim3(768, 1, 1), dim3(8, 16, 1), 0, stream>>>(A, B, C, s, ws);
            return;
        }
        // (K=6144, N=2048) down_proj
        if (K == 6144 && N == 2048) {
            q1g128_kernel_M1<2048, 6144, 8, 16>
                <<<dim3(128, 1, 1), dim3(8, 16, 1), 0, stream>>>(A, B, C, s, ws);
            return;
        }
        // (K=2048, N=151936) lm_head
        if (K == 2048 && N == 151936) {
            // 151936 / 16 = 9496
            q1g128_kernel_M1<151936, 2048, 8, 16>
                <<<dim3(9496, 1, 1), dim3(8, 16, 1), 0, stream>>>(A, B, C, s, ws);
            return;
        }
    }

    // Dispatch supported block widths by (M, K, N).
    #define DISPATCH_MGT1(M_, N_, K_)                                       \
        do {                                                                \
            q1g128_kernel_Mgt1<M_, N_, K_, 8, 16>                            \
                <<<dim3((N_)/16, (M_), 1), dim3(8, 16, 1), 0, stream>>>(A, B, C, s, ws); \
            return;                                                         \
        } while (0)

    #define DISPATCH_MFUSED(M_, N_, K_)                                     \
        do {                                                                \
            q1g128_kernel_mfused<M_, N_, K_>                                 \
                <<<dim3((N_)/32, 1, 1), dim3(8, 32, 1), 0, stream>>>(A, B, C, s, ws); \
            return;                                                         \
        } while (0)

    // KBS=32 uses a full warp for K reduction; NT=8 determines the output
    // rows per block and the grid size.
    // K must be a multiple of KPL*KBS = 1024 (true for K=2048, 6144) and N a
    // multiple of NT=8 (true for 2048/4096/12288/151936).
    // The output projection uses KBS=4, so each K tile covers one group of
    // 128 weights. NT=64 gives 256 threads per block.
    #define DISPATCH_LMHEAD(M_)                                             \
        do {                                                                \
            q1g128_kernel_mfused_t<M_, 151936, 2048, 4, 64>                  \
                <<<dim3(151936/64, 1, 1), dim3(4, 64, 1), 0, stream>>>(A, B, C, s, ws); \
            return;                                                         \
        } while (0)

    #define DISPATCH_MFUSED_T(M_, N_, K_)                                   \
        do {                                                                \
            q1g128_kernel_mfused_t<M_, N_, K_, 32, 8>                        \
                <<<dim3((N_)/8, 1, 1), dim3(32, 8, 1), 0, stream>>>(A, B, C, s, ws); \
            return;                                                         \
        } while (0)

    // Per-shape (KBS, NT) selection controls the K reduction and output-grid
    // partition. The kernel uses dp4a within each group before scaling.
    #define DISPATCH_MFUSED_T_CFG(M_, N_, K_, KBS_, NT_)                     \
        do {                                                                \
            q1g128_kernel_mfused_t<M_, N_, K_, KBS_, NT_>                    \
                <<<dim3((N_)/(NT_), 1, 1), dim3(KBS_, NT_, 1), 0, stream>>>(A, B, C, s, ws); \
            return;                                                         \
        } while (0)

    // Reuse loaded weight tiles across M rows; select tile parameters by
    // layer shape and use the separate output-projection dispatch.
    #define DISPATCH_LAYER_SHAPES(M_)                              \
        do {                                                       \
            if (K == 2048 && N == 2048)   DISPATCH_MFUSED_T_CFG(M_, 2048, 2048, 32, 8);  /* o_proj */ \
            if (K == 2048 && N == 1024)   DISPATCH_MFUSED_T_CFG(M_, 1024, 2048, 16, 8);  /* k/v unmerged */ \
            if (K == 2048 && N == 6144)   DISPATCH_MFUSED_T_CFG(M_, 6144, 2048, 4, 32);  /* gate/up unmerged */ \
            if (K == 2048 && N == 4096)   DISPATCH_MFUSED_T_CFG(M_, 4096, 2048, 16, 8);  /* merged QKV */ \
            if (K == 2048 && N == 12288)  DISPATCH_MFUSED_T_CFG(M_, 12288, 2048, 4, 32); /* merged gate+up */ \
            if (K == 6144 && N == 2048)   DISPATCH_MFUSED_T_CFG(M_, 2048, 6144, 32, 8);  /* down_proj */ \
            if (K == 2048 && N == 151936) DISPATCH_LMHEAD(M_);            \
        } while (0)

    if (M == 4)  { DISPATCH_LAYER_SHAPES(4);  }
    if (M == 5)  { DISPATCH_LAYER_SHAPES(5);  }
    if (M == 6)  { DISPATCH_LAYER_SHAPES(6);  }
    if (M == 7)  { DISPATCH_LAYER_SHAPES(7);  }
    if (M == 8)  { DISPATCH_LAYER_SHAPES(8);  }
    if (M == 9)  { DISPATCH_LAYER_SHAPES(9);  }
    if (M == 10) { DISPATCH_LAYER_SHAPES(10); }
    if (M == 12) { DISPATCH_LAYER_SHAPES(12); }
    if (M == 16) { DISPATCH_LAYER_SHAPES(16); }
    if (M == 17) { DISPATCH_LAYER_SHAPES(17); }

    fprintf(stderr,
            "[q1g128_int8xint1_linear] unsupported (M=%d, N=%d, K=%d) -- add a "
            "shape instantiation in q1g128_kernels_block8.cu\n", M, N, K);
}
