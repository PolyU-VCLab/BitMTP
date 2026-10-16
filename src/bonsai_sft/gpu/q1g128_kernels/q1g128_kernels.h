// Q1_0_g128 (W1A8) INT8 x packed-INT1 mpGEMM kernels for Bonsai-1.7B.
// =====================================================================
// Adapted from Microsoft BitNet kernels; see licenses/BitNet-MIT.txt.
//
// Differences vs BitNet W2A8:
//   1. Weight packing: 8 sign bits / byte (1 bpw) vs 4 ternary vals / byte.
//      Each loaded int (4 bytes) decodes to 32 INT8 values in {-1, +1}.
//   2. Scale layout: per-128-group BF16 per row, shape (N, K/128), vs
//      per-tensor flat float[ws_num]. Each k-iter applies a different scale.
//   3. Accumulation: int dp4a inside a 128-element group, then convert to
//      float and multiply by ws[n_idx, group_idx]; accumulate across groups
//      in float. Activation scale s is per-row (per-token) FP32.
//
// Layout details:
//   A:  int8[M, K]                                 input activations
//   B:  int8[N, K/8]   (each byte = 8 sign bits)   packed weights
//   ws: __nv_bfloat16[N, K/128]                    per-group weight scales
//   s:  float[M]                                   per-row activation scale
//   C:  __nv_bfloat16[M, N]                        output
//
// Weight bit layout inside one byte (LSB = first weight):
//   bit b = (byte >> b) & 1;  sign = (b == 1) ? +1 : -1
// This matches `freeze_quantize_to_q1_0_g128` packed by `np.packbits(..,
// bitorder='little')`.
//
// The build script selects SM_ARCH. dp4a requires sm_61 or later.

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>


// =====================================================================
// decode_i1s_to_i8s: unpack 32 sign bits (1 int = 4 bytes) -> 32 INT8 in
// {-1, +1}, laid out as 8 ints (out[0..7]) such that
//      out_byte[i] == sign of bit i of i1s, i in [0..31].
// i.e. weight w lives in byte (w & 3) of out[w >> 2].  This makes the
// memory order of decoded weights identical to the natural ascending order
// of A activations -> dp4a(A[k..k+4], B[k..k+4]) computes the right dot.
//
// Bit mapping per nibble: 4 consecutive sign bits are spread to the 4 byte
// lanes of one output int using a multiply-shift, then mapped {0,1}->{-1,+1}
// with __vsubss4 (per-byte signed sub with saturation).
// =====================================================================
template <typename T1, typename T2>
__device__ __forceinline__ void decode_i1s_to_i8s(T1 *_i1s, T2 *_i8s)
{
    uint i1s = *reinterpret_cast<uint *>(_i1s);
    uint *out = reinterpret_cast<uint *>(_i8s);   // 8 ints == 32 INT8
    constexpr uint ONES = 0x01010101;

    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        uint nibble = (i1s >> (4 * g)) & 0xf;
        // Spread the 4 bits of `nibble` into the LSB of 4 bytes:
        //   bit 0 -> byte 0 LSB, bit 1 -> byte 1 LSB, bit 2 -> byte 2 LSB,
        //   bit 3 -> byte 3 LSB.
        uint as_bytes =
              ((nibble       ) & 0x01)         //  bit 0 -> byte 0
            | (((nibble >> 1) & 0x01) <<  8)   //  bit 1 -> byte 1
            | (((nibble >> 2) & 0x01) << 16)   //  bit 2 -> byte 2
            | (((nibble >> 3) & 0x01) << 24);  //  bit 3 -> byte 3
        // Map 0->-1, 1->+1 per byte: (2*x) - 1 with __vsubss4 saturating sub.
        out[g] = __vsubss4(as_bytes << 1, ONES);
    }
}


// =====================================================================
// Common tile configuration
// =====================================================================
// One thread handles KPL=32 weights per K-iteration; KBS threads cover the
// K-reduction dimension; one KPL*KBS=256-element K stride per outer iter.
// For Q1, KPL=32 means each thread loads 4 bytes of packed weight per K-iter
// (== 32 sign bits == 32 INT8 after decode).
constexpr int Q1_GROUP_SIZE = 128;   // matches ggml's Q1_0_g128 group


// =====================================================================
// q1g128_kernel_M1: AR (M=1) decode path.
// Grid:  (N / N_block_size, 1, 1)
// Block: (8 threads x N_block_size threads y, 1) = 128 threads.
// Each (tx, ty) computes one output element (m=0, n = blockIdx.x*N_block_size + ty).
// =====================================================================
template <int N, int K, int K_block_size, int N_block_size>
__global__ void __launch_bounds__(128)
q1g128_kernel_M1(int8_t* __restrict__ A,
                 int8_t* __restrict__ B,
                 __nv_bfloat16* __restrict__ C,
                 float* __restrict__ s,
                 __nv_bfloat16* __restrict__ ws)
{
    constexpr int KPL = 32;                    // weights per thread per K-iter
    constexpr int KBS = K_block_size;          // K reduction width
    constexpr int K_PER_K_ITER = KPL * KBS;    // weights per outer K-iter
    static_assert(K_PER_K_ITER == 256 || K_PER_K_ITER == Q1_GROUP_SIZE * 2,
                  "K_PER_K_ITER expected to be 256 (= 2 groups across all threads)");
    static_assert(K % Q1_GROUP_SIZE == 0, "K must be multiple of 128");
    static_assert(K % K_PER_K_ITER == 0, "K must be multiple of KPL*KBS");

    const int tx = (int)threadIdx.x;        // K reduction lane (0..KBS-1)
    const int ty = (int)threadIdx.y;        // N tile lane     (0..N_block_size-1)
    const int n_idx = ((int)blockIdx.x) * N_block_size + ty;

    // Pointers to this row's packed weight (K/8 bytes) and scale (K/128 bf16).
    const int8_t* w_row = B  + n_idx * (K / 8);
    const __nv_bfloat16* s_row = ws + n_idx * (K / Q1_GROUP_SIZE);

    // Optimized: each thread accumulates its already-scaled partial as float,
    // and a single warp reduction at the very end. This removes 2x shfl_xor
    // per K-iter (was applied to part_g0 and part_g1 separately).
    float acc_scaled = 0.0f;
    constexpr int OUTER_ITERS = K / K_PER_K_ITER;          // K / 256

    signed char A_local[KPL];
    signed char B_dec[KPL];
    int B_pack[1];

    const int group_lane = (tx >= (KBS / 2)) ? 1 : 0;

    #pragma unroll
    for (int k_outer = 0; k_outer < OUTER_ITERS; ++k_outer) {
        const int k_start = k_outer * K_PER_K_ITER + tx * KPL;
        *(int4*)(A_local +  0) = *(int4*)(A + k_start +  0);
        *(int4*)(A_local + 16) = *(int4*)(A + k_start + 16);

        B_pack[0] = *(int*)(w_row + (k_start >> 3));
        decode_i1s_to_i8s(B_pack, B_dec);

        int local_acc = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            local_acc = __dp4a(*(int*)&A_local[i*4], *(int*)&B_dec[i*4], local_acc);
        }

        const int g_base = k_outer * 2;
        const float ws_g = __bfloat162float(s_row[g_base + group_lane]);
        acc_scaled += (float)local_acc * ws_g;
    }

    float r = acc_scaled;
    #pragma unroll
    for (int off = KBS / 2; off > 0; off >>= 1)
        r += __shfl_xor_sync(0xffffffff, r, off, KBS);

    if (tx == 0) {
        C[n_idx] = __float2bfloat16(r / s[0]);
    }
}


// =====================================================================
// q1g128_kernel_Mgt1: SDLM block decode (M > 1), one m-row per blockIdx.y.
// Grid:  (N/N_block_size, M, 1)
// Block: (KBS, N_block_size, 1)
// =====================================================================
template <int M, int N, int K, int K_block_size, int N_block_size>
__global__ void __launch_bounds__(128)
q1g128_kernel_Mgt1(int8_t* __restrict__ A,
                   int8_t* __restrict__ B,
                   __nv_bfloat16* __restrict__ C,
                   float* __restrict__ s,
                   __nv_bfloat16* __restrict__ ws)
{
    constexpr int KPL = 32;
    constexpr int KBS = K_block_size;
    constexpr int K_PER_K_ITER = KPL * KBS;
    static_assert(K % K_PER_K_ITER == 0, "K must be multiple of KPL*KBS");

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;
    const int n_idx = ((int)blockIdx.x) * N_block_size + ty;
    const int m_idx = (int)blockIdx.y;

    const int8_t* w_row = B  + n_idx * (K / 8);
    const __nv_bfloat16* s_row = ws + n_idx * (K / Q1_GROUP_SIZE);
    const int8_t* a_row = A  + m_idx * K;

    // Same optimization as M1/mfused: accumulate scaled per-thread float,
    // single warp reduction at the end.
    float acc_scaled = 0.0f;
    constexpr int OUTER_ITERS = K / K_PER_K_ITER;

    signed char A_local[KPL];
    signed char B_dec[KPL];
    int B_pack[1];

    const int group_lane = (tx >= (KBS / 2)) ? 1 : 0;

    #pragma unroll
    for (int k_outer = 0; k_outer < OUTER_ITERS; ++k_outer) {
        const int k_start = k_outer * K_PER_K_ITER + tx * KPL;
        *(int4*)(A_local +  0) = *(int4*)(a_row + k_start +  0);
        *(int4*)(A_local + 16) = *(int4*)(a_row + k_start + 16);
        B_pack[0] = *(int*)(w_row + (k_start >> 3));
        decode_i1s_to_i8s(B_pack, B_dec);

        int local_acc = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            local_acc = __dp4a(*(int*)&A_local[i*4], *(int*)&B_dec[i*4], local_acc);
        }
        const int g_base = k_outer * 2;
        const float ws_g = __bfloat162float(s_row[g_base + group_lane]);
        acc_scaled += (float)local_acc * ws_g;
    }

    float r = acc_scaled;
    #pragma unroll
    for (int off = KBS / 2; off > 0; off >>= 1)
        r += __shfl_xor_sync(0xffffffff, r, off, KBS);

    if (tx == 0) {
        C[m_idx * N + n_idx] = __float2bfloat16(r / s[m_idx]);
    }
}


// =====================================================================
// q1g128_kernel_M1_varseg: AR decode with N split into 3 segments (Q/K/V).
// Activation scale s[0] shared, but the kernel is otherwise the same as M1
// (no special handling needed because per-group scales already index into
//  the per-row `ws` table).
// Kept as a separate symbol for parity with BitNet's varseg API. (No new
// per-segment behaviour is needed: ws is per-row 2D, so segment boundaries
// are encoded entirely in the row-major layout.)
// =====================================================================
template <int N, int K, int /*seg0_end*/, int /*seg1_end*/, int K_block_size, int N_block_size>
__global__ void __launch_bounds__(128)
q1g128_kernel_M1_varseg(int8_t* __restrict__ A,
                        int8_t* __restrict__ B,
                        __nv_bfloat16* __restrict__ C,
                        float* __restrict__ s,
                        __nv_bfloat16* __restrict__ ws)
{
    // Identical to q1g128_kernel_M1; the per-segment dimension of ws is
    // already baked into the (N, K/128) shape.
    q1g128_kernel_M1<N, K, K_block_size, N_block_size>(A, B, C, s, ws);
}

template <int M, int N, int K, int /*seg0_end*/, int /*seg1_end*/, int K_block_size, int N_block_size>
__global__ void __launch_bounds__(128)
q1g128_kernel_Mgt1_varseg(int8_t* __restrict__ A,
                          int8_t* __restrict__ B,
                          __nv_bfloat16* __restrict__ C,
                          float* __restrict__ s,
                          __nv_bfloat16* __restrict__ ws)
{
    q1g128_kernel_Mgt1<M, N, K, K_block_size, N_block_size>(A, B, C, s, ws);
}


// =====================================================================
// q1g128_kernel_mfused: M rows handled by each thread, weight loaded ONCE
// per K-iter (best for large N where grid.x already gives enough blocks).
// Grid:  (N / 32, 1, 1)  Block: (8, 32, 1) = 256 threads
// =====================================================================
template <int M, int N, int K>
__global__ void __launch_bounds__(256)
q1g128_kernel_mfused(int8_t* __restrict__ A,
                     int8_t* __restrict__ B,
                     __nv_bfloat16* __restrict__ C,
                     float* __restrict__ s,
                     __nv_bfloat16* __restrict__ ws)
{
    // Optimized: avoid per-M warp reduction. Each thread accumulates
    //   acc_scaled[m] += local_acc * (group_lane==0 ? ws_g0 : ws_g1)
    // as a per-thread float, then a single warp-sum at the very end (post K-loop).
    // This matches BitMTP's mfused structure: weight loaded once, M-rows reused,
    // reduction outside the M-loop.
    constexpr int KPL = 32;
    constexpr int KBS = 8;
    constexpr int K_PER_K_ITER = KPL * KBS;     // 256 weights
    static_assert(K % K_PER_K_ITER == 0, "K must be multiple of 256");

    const int tx = (int)threadIdx.x;          // 0..7 (K)
    const int ty = (int)threadIdx.y;          // 0..31 (N)
    const int n_idx = ((int)blockIdx.x) * 32 + ty;

    const int8_t* w_row = B  + n_idx * (K / 8);
    const __nv_bfloat16* s_row = ws + n_idx * (K / Q1_GROUP_SIZE);

    // Per-thread per-m scaled partial sum.
    float acc_scaled[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) acc_scaled[m] = 0.0f;

    signed char A_local[KPL];
    signed char B_dec[KPL];
    int B_pack[1];

    constexpr int OUTER_ITERS = K / K_PER_K_ITER;
    const int group_lane = (tx >= (KBS / 2)) ? 1 : 0;

    #pragma unroll
    for (int k_outer = 0; k_outer < OUTER_ITERS; ++k_outer) {
        const int k_start = k_outer * K_PER_K_ITER + tx * KPL;

        // Load weight ONCE per K-iter (shared across M rows).
        B_pack[0] = *(int*)(w_row + (k_start >> 3));
        decode_i1s_to_i8s(B_pack, B_dec);

        const int g_base = k_outer * 2;
        const float ws_g = __bfloat162float(s_row[g_base + group_lane]);

        #pragma unroll
        for (int m = 0; m < M; ++m) {
            *(int4*)(A_local +  0) = *(int4*)(A + m * K + k_start +  0);
            *(int4*)(A_local + 16) = *(int4*)(A + m * K + k_start + 16);

            int local_acc = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                local_acc = __dp4a(*(int*)&A_local[i*4], *(int*)&B_dec[i*4], local_acc);
            }
            // Per-thread scaled accumulation. No warp sync here.
            acc_scaled[m] += (float)local_acc * ws_g;
        }
    }

    // Single warp reduction at the end of *all* K-iters, per M row.
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float r = acc_scaled[m];
        #pragma unroll
        for (int off = KBS / 2; off > 0; off >>= 1)
            r += __shfl_xor_sync(0xffffffff, r, off, KBS);
        if (tx == 0)
            C[m * N + n_idx] = __float2bfloat16(r / s[m]);
    }
}


// =====================================================================
// q1g128_kernel_mfused_t: tunable (KBS, NT) variant of mfused.
//
// Motivation (Bonsai 1-bit decode profiling): the original mfused uses
// KBS=8 (K-reduction lanes) and NT=32 (N tile) => grid (N/32). For the
// small-N layers (o_proj/qkv/down, N<=4096) this yields only 64-128 grid
// blocks, far below the 78 SMs, leaving the GPU latency-bound at ~1-4% of
// HBM peak. Widening KBS shortens each thread's serial K-chain and shrinking
// NT multiplies the grid so every SM is filled.
//
// With NT and KBS chosen so KBS*NT == 256 and a warp == one N column
// (linear tid = tx + ty*KBS, so KBS|32), the K-reduction is a single
// full/partial warp shuffle. KBS=32, NT=8 makes each warp own exactly one
// output column with a full 32-lane reduction.
//
// Scale indexing: each thread owns KPL=32 contiguous weights, i.e. exactly
// one 128-wide group spans 4 threads -> group_in_iter = tx/4, and there are
// GROUPS_PER_ITER = (KPL*KBS)/128 = KBS/4 groups advanced per outer iter.
// Grid:  (N / NT, 1, 1)   Block: (KBS, NT, 1)
// =====================================================================
template <int M, int N, int K, int KBS, int NT>
__global__ void __launch_bounds__(KBS * NT)
q1g128_kernel_mfused_t(int8_t* __restrict__ A,
                       int8_t* __restrict__ B,
                       __nv_bfloat16* __restrict__ C,
                       float* __restrict__ s,
                       __nv_bfloat16* __restrict__ ws)
{
    constexpr int KPL = 32;
    constexpr int K_PER_K_ITER = KPL * KBS;
    static_assert(K % K_PER_K_ITER == 0, "K must be multiple of KPL*KBS");
    static_assert(K_PER_K_ITER % 128 == 0, "K_PER_K_ITER must be multiple of group");
    constexpr int GROUPS_PER_ITER = K_PER_K_ITER / 128;   // = KBS/4
    constexpr int OUTER_ITERS = K / K_PER_K_ITER;

    const int tx = (int)threadIdx.x;          // 0..KBS-1 (K reduction)
    const int ty = (int)threadIdx.y;          // 0..NT-1  (N tile)
    const int n_idx = ((int)blockIdx.x) * NT + ty;

    const int8_t* w_row = B  + n_idx * (K / 8);
    const __nv_bfloat16* s_row = ws + n_idx * (K / Q1_GROUP_SIZE);

    float acc_scaled[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) acc_scaled[m] = 0.0f;

    signed char A_local[KPL];
    signed char B_dec[KPL];
    int B_pack[1];

    const int group_in_iter = tx >> 2;        // tx/4: 32 weights => one group / 4 threads

    #pragma unroll
    for (int k_outer = 0; k_outer < OUTER_ITERS; ++k_outer) {
        const int k_start = k_outer * K_PER_K_ITER + tx * KPL;

        // Weight loaded ONCE per K-iter (reused across all M rows).
        B_pack[0] = *(int*)(w_row + (k_start >> 3));
        decode_i1s_to_i8s(B_pack, B_dec);

        const float ws_g =
            __bfloat162float(s_row[k_outer * GROUPS_PER_ITER + group_in_iter]);

        #pragma unroll
        for (int m = 0; m < M; ++m) {
            *(int4*)(A_local +  0) = *(int4*)(A + m * K + k_start +  0);
            *(int4*)(A_local + 16) = *(int4*)(A + m * K + k_start + 16);

            int local_acc = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                local_acc = __dp4a(*(int*)&A_local[i*4], *(int*)&B_dec[i*4], local_acc);
            }
            acc_scaled[m] += (float)local_acc * ws_g;
        }
    }

    // Warp reduction over the KBS lanes (threads sharing one n_idx).
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float r = acc_scaled[m];
        #pragma unroll
        for (int off = KBS / 2; off > 0; off >>= 1)
            r += __shfl_xor_sync(0xffffffff, r, off, KBS);
        if (tx == 0)
            C[m * N + n_idx] = __float2bfloat16(r / s[m]);
    }
}


// =====================================================================
// q1g128_kernel_mfused_varseg: same as mfused, but with 3 segments on N
// (Q/K/V). With per-row 2D ws this collapses to the same body.
// =====================================================================
template <int M, int N, int K, int /*seg0_end*/, int /*seg1_end*/>
__global__ void __launch_bounds__(256)
q1g128_kernel_mfused_varseg(int8_t* __restrict__ A,
                            int8_t* __restrict__ B,
                            __nv_bfloat16* __restrict__ C,
                            float* __restrict__ s,
                            __nv_bfloat16* __restrict__ ws)
{
    q1g128_kernel_mfused<M, N, K>(A, B, C, s, ws);
}
