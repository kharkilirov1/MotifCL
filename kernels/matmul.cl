#define TILE 16

__kernel void matmul_naive_f32(__global const float* A,
                               __global const float* B,
                               __global float* C,
                               int M,
                               int N,
                               int K) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

__kernel void matmul_tiled_f32(__global const float* A,
                               __global const float* B,
                               __global float* C,
                               int M,
                               int N,
                               int K) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int local_col = get_local_id(0);
    int local_row = get_local_id(1);

    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];

    float acc = 0.0f;
    int tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        int a_col = t * TILE + local_col;
        int b_row = t * TILE + local_row;
        As[local_row][local_col] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[local_row][local_col] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE; ++k) acc += As[local_row][k] * Bs[k][local_col];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

#define RB_M 32
#define RB_N 32
#define RB_K 16
#define RB_THREAD_M 4
#define RB_THREAD_N 4
#define RB_LX 8
#define RB_LY 8

__kernel void matmul_register_block4_f32(__global const float* A,
                                         __global const float* B,
                                         __global float* C,
                                         int M,
                                         int N,
                                         int K) {
    int lcol = get_local_id(0);
    int lrow = get_local_id(1);
    int lid = lrow * RB_LX + lcol;
    int block_col = get_group_id(0) * RB_N;
    int block_row = get_group_id(1) * RB_M;

    __local float As[RB_M][RB_K];
    __local float Bs[RB_K][RB_N];

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    for (int kt = 0; kt < K; kt += RB_K) {
        for (int idx = lid; idx < RB_M * RB_K; idx += RB_LX * RB_LY) {
            int r = idx / RB_K;
            int kk = idx - r * RB_K;
            int gr = block_row + r;
            int gk = kt + kk;
            As[r][kk] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f;
        }
        for (int idx = lid; idx < RB_K * RB_N; idx += RB_LX * RB_LY) {
            int kk = idx / RB_N;
            int c = idx - kk * RB_N;
            int gk = kt + kk;
            int gc = block_col + c;
            Bs[kk][c] = (gk < K && gc < N) ? B[gk * N + gc] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int rr = lrow * RB_THREAD_M;
        int cc = lcol * RB_THREAD_N;
        for (int kk = 0; kk < RB_K; ++kk) {
            float a0 = As[rr + 0][kk];
            float a1 = As[rr + 1][kk];
            float a2 = As[rr + 2][kk];
            float a3 = As[rr + 3][kk];
            float b0 = Bs[kk][cc + 0];
            float b1 = Bs[kk][cc + 1];
            float b2 = Bs[kk][cc + 2];
            float b3 = Bs[kk][cc + 3];
            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int row0 = block_row + lrow * RB_THREAD_M;
    int col0 = block_col + lcol * RB_THREAD_N;
    if (row0 + 0 < M && col0 + 0 < N) C[(row0 + 0) * N + col0 + 0] = acc00;
    if (row0 + 0 < M && col0 + 1 < N) C[(row0 + 0) * N + col0 + 1] = acc01;
    if (row0 + 0 < M && col0 + 2 < N) C[(row0 + 0) * N + col0 + 2] = acc02;
    if (row0 + 0 < M && col0 + 3 < N) C[(row0 + 0) * N + col0 + 3] = acc03;
    if (row0 + 1 < M && col0 + 0 < N) C[(row0 + 1) * N + col0 + 0] = acc10;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = acc11;
    if (row0 + 1 < M && col0 + 2 < N) C[(row0 + 1) * N + col0 + 2] = acc12;
    if (row0 + 1 < M && col0 + 3 < N) C[(row0 + 1) * N + col0 + 3] = acc13;
    if (row0 + 2 < M && col0 + 0 < N) C[(row0 + 2) * N + col0 + 0] = acc20;
    if (row0 + 2 < M && col0 + 1 < N) C[(row0 + 2) * N + col0 + 1] = acc21;
    if (row0 + 2 < M && col0 + 2 < N) C[(row0 + 2) * N + col0 + 2] = acc22;
    if (row0 + 2 < M && col0 + 3 < N) C[(row0 + 2) * N + col0 + 3] = acc23;
    if (row0 + 3 < M && col0 + 0 < N) C[(row0 + 3) * N + col0 + 0] = acc30;
    if (row0 + 3 < M && col0 + 1 < N) C[(row0 + 3) * N + col0 + 1] = acc31;
    if (row0 + 3 < M && col0 + 2 < N) C[(row0 + 3) * N + col0 + 2] = acc32;
    if (row0 + 3 < M && col0 + 3 < N) C[(row0 + 3) * N + col0 + 3] = acc33;
}

// Decode-specialized dense vector-matrix multiply for M=1.
// The generic register-block kernel is optimized for 2D tiles and launches
// many unused row groups when M is a single token. This path uses exactly one
// work-item per output column and is intentionally simple: for the Gemma4 PLE
// path most dense matmuls are [1,K] x [K,N] with moderate K/N.
__kernel void matmul_f32_m1_f32(__global const float* A,
                                __global const float* B,
                                __global float* C,
                                int N,
                                int K) {
    int col = get_global_id(0);
    if (col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[k] * B[k * N + col];
    }
    C[col] = acc;
}

// Work-group reduction variant for larger K. It assigns one work-group to one
// output column and parallelizes the K reduction. Kept separate from the simple
// kernel so dispatch can choose the lower-overhead path for small K.
__kernel void matmul_f32_m1_wg_f32(__global const float* A,
                                   __global const float* B,
                                   __global float* C,
                                   int N,
                                   int K,
                                   __local float* scratch) {
    int col = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    if (col >= N) return;
    float acc = 0.0f;
    for (int k = lid; k < K; k += local_size) {
        acc += A[k] * B[k * N + col];
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) C[col] = scratch[0];
}


__kernel void rmsnorm_matmul_rb4_f32(__global const float* X,
                                     __global const float* norm_weight,
                                     __global const float* row_inv,
                                     __global const float* B,
                                     __global float* C,
                                     int M,
                                     int N,
                                     int K) {
    int lcol = get_local_id(0);
    int lrow = get_local_id(1);
    int lid = lrow * RB_LX + lcol;
    int block_col = get_group_id(0) * RB_N;
    int block_row = get_group_id(1) * RB_M;

    __local float As[RB_M][RB_K];
    __local float Bs[RB_K][RB_N];

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    for (int kt = 0; kt < K; kt += RB_K) {
        for (int idx = lid; idx < RB_M * RB_K; idx += RB_LX * RB_LY) {
            int r = idx / RB_K;
            int kk = idx - r * RB_K;
            int gr = block_row + r;
            int gk = kt + kk;
            As[r][kk] = (gr < M && gk < K) ? (X[gr * K + gk] * norm_weight[gk] * row_inv[gr]) : 0.0f;
        }
        for (int idx = lid; idx < RB_K * RB_N; idx += RB_LX * RB_LY) {
            int kk = idx / RB_N;
            int c = idx - kk * RB_N;
            int gk = kt + kk;
            int gc = block_col + c;
            Bs[kk][c] = (gk < K && gc < N) ? B[gk * N + gc] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int rr = lrow * RB_THREAD_M;
        int cc = lcol * RB_THREAD_N;
        for (int kk = 0; kk < RB_K; ++kk) {
            float a0 = As[rr + 0][kk];
            float a1 = As[rr + 1][kk];
            float a2 = As[rr + 2][kk];
            float a3 = As[rr + 3][kk];
            float b0 = Bs[kk][cc + 0];
            float b1 = Bs[kk][cc + 1];
            float b2 = Bs[kk][cc + 2];
            float b3 = Bs[kk][cc + 3];
            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int row0 = block_row + lrow * RB_THREAD_M;
    int col0 = block_col + lcol * RB_THREAD_N;
    if (row0 + 0 < M && col0 + 0 < N) C[(row0 + 0) * N + col0 + 0] = acc00;
    if (row0 + 0 < M && col0 + 1 < N) C[(row0 + 0) * N + col0 + 1] = acc01;
    if (row0 + 0 < M && col0 + 2 < N) C[(row0 + 0) * N + col0 + 2] = acc02;
    if (row0 + 0 < M && col0 + 3 < N) C[(row0 + 0) * N + col0 + 3] = acc03;
    if (row0 + 1 < M && col0 + 0 < N) C[(row0 + 1) * N + col0 + 0] = acc10;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = acc11;
    if (row0 + 1 < M && col0 + 2 < N) C[(row0 + 1) * N + col0 + 2] = acc12;
    if (row0 + 1 < M && col0 + 3 < N) C[(row0 + 1) * N + col0 + 3] = acc13;
    if (row0 + 2 < M && col0 + 0 < N) C[(row0 + 2) * N + col0 + 0] = acc20;
    if (row0 + 2 < M && col0 + 1 < N) C[(row0 + 2) * N + col0 + 1] = acc21;
    if (row0 + 2 < M && col0 + 2 < N) C[(row0 + 2) * N + col0 + 2] = acc22;
    if (row0 + 2 < M && col0 + 3 < N) C[(row0 + 2) * N + col0 + 3] = acc23;
    if (row0 + 3 < M && col0 + 0 < N) C[(row0 + 3) * N + col0 + 0] = acc30;
    if (row0 + 3 < M && col0 + 1 < N) C[(row0 + 3) * N + col0 + 1] = acc31;
    if (row0 + 3 < M && col0 + 2 < N) C[(row0 + 3) * N + col0 + 2] = acc32;
    if (row0 + 3 < M && col0 + 3 < N) C[(row0 + 3) * N + col0 + 3] = acc33;
}

__kernel void rmsnorm_matmul_transa_rb4_f32(__global const float* X,
                                            __global const float* norm_weight,
                                            __global const float* row_inv,
                                            __global const float* B,
                                            __global float* C,
                                            int M,
                                            int N,
                                            int K) {
    int lcol = get_local_id(0);
    int lrow = get_local_id(1);
    int lid = lrow * RB_LX + lcol;
    int block_col = get_group_id(0) * RB_N;
    int block_row = get_group_id(1) * RB_M;

    __local float As[RB_M][RB_K];
    __local float Bs[RB_K][RB_N];

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    for (int kt = 0; kt < K; kt += RB_K) {
        for (int idx = lid; idx < RB_M * RB_K; idx += RB_LX * RB_LY) {
            int kk = idx / RB_M;
            int r = idx - kk * RB_M;
            int gr = block_row + r;
            int gk = kt + kk;
            As[r][kk] = (gr < M && gk < K) ? (X[gk * M + gr] * norm_weight[gr] * row_inv[gk]) : 0.0f;
        }
        for (int idx = lid; idx < RB_K * RB_N; idx += RB_LX * RB_LY) {
            int kk = idx / RB_N;
            int c = idx - kk * RB_N;
            int gk = kt + kk;
            int gc = block_col + c;
            Bs[kk][c] = (gk < K && gc < N) ? B[gk * N + gc] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int rr = lrow * RB_THREAD_M;
        int cc = lcol * RB_THREAD_N;
        for (int kk = 0; kk < RB_K; ++kk) {
            float a0 = As[rr + 0][kk];
            float a1 = As[rr + 1][kk];
            float a2 = As[rr + 2][kk];
            float a3 = As[rr + 3][kk];
            float b0 = Bs[kk][cc + 0];
            float b1 = Bs[kk][cc + 1];
            float b2 = Bs[kk][cc + 2];
            float b3 = Bs[kk][cc + 3];
            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int row0 = block_row + lrow * RB_THREAD_M;
    int col0 = block_col + lcol * RB_THREAD_N;
    if (row0 + 0 < M && col0 + 0 < N) C[(row0 + 0) * N + col0 + 0] = acc00;
    if (row0 + 0 < M && col0 + 1 < N) C[(row0 + 0) * N + col0 + 1] = acc01;
    if (row0 + 0 < M && col0 + 2 < N) C[(row0 + 0) * N + col0 + 2] = acc02;
    if (row0 + 0 < M && col0 + 3 < N) C[(row0 + 0) * N + col0 + 3] = acc03;
    if (row0 + 1 < M && col0 + 0 < N) C[(row0 + 1) * N + col0 + 0] = acc10;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = acc11;
    if (row0 + 1 < M && col0 + 2 < N) C[(row0 + 1) * N + col0 + 2] = acc12;
    if (row0 + 1 < M && col0 + 3 < N) C[(row0 + 1) * N + col0 + 3] = acc13;
    if (row0 + 2 < M && col0 + 0 < N) C[(row0 + 2) * N + col0 + 0] = acc20;
    if (row0 + 2 < M && col0 + 1 < N) C[(row0 + 2) * N + col0 + 1] = acc21;
    if (row0 + 2 < M && col0 + 2 < N) C[(row0 + 2) * N + col0 + 2] = acc22;
    if (row0 + 2 < M && col0 + 3 < N) C[(row0 + 2) * N + col0 + 3] = acc23;
    if (row0 + 3 < M && col0 + 0 < N) C[(row0 + 3) * N + col0 + 0] = acc30;
    if (row0 + 3 < M && col0 + 1 < N) C[(row0 + 3) * N + col0 + 1] = acc31;
    if (row0 + 3 < M && col0 + 2 < N) C[(row0 + 3) * N + col0 + 2] = acc32;
    if (row0 + 3 < M && col0 + 3 < N) C[(row0 + 3) * N + col0 + 3] = acc33;
}
__kernel void matmul_flags_f32(__global const float* A,
                               __global const float* B,
                               __global float* C,
                               int M,
                               int N,
                               int K,
                               int transA,
                               int transB) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float av = transA ? A[k * M + row] : A[row * K + k];
        float bv = transB ? B[col * K + k] : B[k * N + col];
        acc += av * bv;
    }
    C[row * N + col] = acc;
}

__kernel void matmul_transa_rb4_f32(__global const float* A,
                                    __global const float* B,
                                    __global float* C,
                                    int M,
                                    int N,
                                    int K) {
    int lcol = get_local_id(0);
    int lrow = get_local_id(1);
    int lid = lrow * RB_LX + lcol;
    int block_col = get_group_id(0) * RB_N;
    int block_row = get_group_id(1) * RB_M;

    __local float As[RB_M][RB_K];
    __local float Bs[RB_K][RB_N];

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    for (int kt = 0; kt < K; kt += RB_K) {
        for (int idx = lid; idx < RB_M * RB_K; idx += RB_LX * RB_LY) {
            int kk = idx / RB_M;
            int r = idx - kk * RB_M;
            int gr = block_row + r;
            int gk = kt + kk;
            As[r][kk] = (gr < M && gk < K) ? A[gk * M + gr] : 0.0f;
        }
        for (int idx = lid; idx < RB_K * RB_N; idx += RB_LX * RB_LY) {
            int kk = idx / RB_N;
            int c = idx - kk * RB_N;
            int gk = kt + kk;
            int gc = block_col + c;
            Bs[kk][c] = (gk < K && gc < N) ? B[gk * N + gc] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int rr = lrow * RB_THREAD_M;
        int cc = lcol * RB_THREAD_N;
        for (int kk = 0; kk < RB_K; ++kk) {
            float a0 = As[rr + 0][kk];
            float a1 = As[rr + 1][kk];
            float a2 = As[rr + 2][kk];
            float a3 = As[rr + 3][kk];
            float b0 = Bs[kk][cc + 0];
            float b1 = Bs[kk][cc + 1];
            float b2 = Bs[kk][cc + 2];
            float b3 = Bs[kk][cc + 3];
            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int row0 = block_row + lrow * RB_THREAD_M;
    int col0 = block_col + lcol * RB_THREAD_N;
    if (row0 + 0 < M && col0 + 0 < N) C[(row0 + 0) * N + col0 + 0] = acc00;
    if (row0 + 0 < M && col0 + 1 < N) C[(row0 + 0) * N + col0 + 1] = acc01;
    if (row0 + 0 < M && col0 + 2 < N) C[(row0 + 0) * N + col0 + 2] = acc02;
    if (row0 + 0 < M && col0 + 3 < N) C[(row0 + 0) * N + col0 + 3] = acc03;
    if (row0 + 1 < M && col0 + 0 < N) C[(row0 + 1) * N + col0 + 0] = acc10;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = acc11;
    if (row0 + 1 < M && col0 + 2 < N) C[(row0 + 1) * N + col0 + 2] = acc12;
    if (row0 + 1 < M && col0 + 3 < N) C[(row0 + 1) * N + col0 + 3] = acc13;
    if (row0 + 2 < M && col0 + 0 < N) C[(row0 + 2) * N + col0 + 0] = acc20;
    if (row0 + 2 < M && col0 + 1 < N) C[(row0 + 2) * N + col0 + 1] = acc21;
    if (row0 + 2 < M && col0 + 2 < N) C[(row0 + 2) * N + col0 + 2] = acc22;
    if (row0 + 2 < M && col0 + 3 < N) C[(row0 + 2) * N + col0 + 3] = acc23;
    if (row0 + 3 < M && col0 + 0 < N) C[(row0 + 3) * N + col0 + 0] = acc30;
    if (row0 + 3 < M && col0 + 1 < N) C[(row0 + 3) * N + col0 + 1] = acc31;
    if (row0 + 3 < M && col0 + 2 < N) C[(row0 + 3) * N + col0 + 2] = acc32;
    if (row0 + 3 < M && col0 + 3 < N) C[(row0 + 3) * N + col0 + 3] = acc33;
}

__kernel void matmul_transb_rb4_f32(__global const float* A,
                                    __global const float* B,
                                    __global float* C,
                                    int M,
                                    int N,
                                    int K) {
    int lcol = get_local_id(0);
    int lrow = get_local_id(1);
    int lid = lrow * RB_LX + lcol;
    int block_col = get_group_id(0) * RB_N;
    int block_row = get_group_id(1) * RB_M;

    __local float As[RB_M][RB_K];
    __local float Bs[RB_K][RB_N];

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    for (int kt = 0; kt < K; kt += RB_K) {
        for (int idx = lid; idx < RB_M * RB_K; idx += RB_LX * RB_LY) {
            int r = idx / RB_K;
            int kk = idx - r * RB_K;
            int gr = block_row + r;
            int gk = kt + kk;
            As[r][kk] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f;
        }
        for (int idx = lid; idx < RB_K * RB_N; idx += RB_LX * RB_LY) {
            int c = idx / RB_K;
            int kk = idx - c * RB_K;
            int gk = kt + kk;
            int gc = block_col + c;
            Bs[kk][c] = (gk < K && gc < N) ? B[gc * K + gk] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int rr = lrow * RB_THREAD_M;
        int cc = lcol * RB_THREAD_N;
        for (int kk = 0; kk < RB_K; ++kk) {
            float a0 = As[rr + 0][kk];
            float a1 = As[rr + 1][kk];
            float a2 = As[rr + 2][kk];
            float a3 = As[rr + 3][kk];
            float b0 = Bs[kk][cc + 0];
            float b1 = Bs[kk][cc + 1];
            float b2 = Bs[kk][cc + 2];
            float b3 = Bs[kk][cc + 3];
            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int row0 = block_row + lrow * RB_THREAD_M;
    int col0 = block_col + lcol * RB_THREAD_N;
    if (row0 + 0 < M && col0 + 0 < N) C[(row0 + 0) * N + col0 + 0] = acc00;
    if (row0 + 0 < M && col0 + 1 < N) C[(row0 + 0) * N + col0 + 1] = acc01;
    if (row0 + 0 < M && col0 + 2 < N) C[(row0 + 0) * N + col0 + 2] = acc02;
    if (row0 + 0 < M && col0 + 3 < N) C[(row0 + 0) * N + col0 + 3] = acc03;
    if (row0 + 1 < M && col0 + 0 < N) C[(row0 + 1) * N + col0 + 0] = acc10;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = acc11;
    if (row0 + 1 < M && col0 + 2 < N) C[(row0 + 1) * N + col0 + 2] = acc12;
    if (row0 + 1 < M && col0 + 3 < N) C[(row0 + 1) * N + col0 + 3] = acc13;
    if (row0 + 2 < M && col0 + 0 < N) C[(row0 + 2) * N + col0 + 0] = acc20;
    if (row0 + 2 < M && col0 + 1 < N) C[(row0 + 2) * N + col0 + 1] = acc21;
    if (row0 + 2 < M && col0 + 2 < N) C[(row0 + 2) * N + col0 + 2] = acc22;
    if (row0 + 2 < M && col0 + 3 < N) C[(row0 + 2) * N + col0 + 3] = acc23;
    if (row0 + 3 < M && col0 + 0 < N) C[(row0 + 3) * N + col0 + 0] = acc30;
    if (row0 + 3 < M && col0 + 1 < N) C[(row0 + 3) * N + col0 + 1] = acc31;
    if (row0 + 3 < M && col0 + 2 < N) C[(row0 + 3) * N + col0 + 2] = acc32;
    if (row0 + 3 < M && col0 + 3 < N) C[(row0 + 3) * N + col0 + 3] = acc33;
}

// Specialized fixed-K=512 GPT kernels. They remove K bounds/index multiplies from the hot loop.
__kernel void matmul_register_block4_k512_f32(__global const float* A,
                                         __global const float* B,
                                         __global float* C,
                                         int M,
                                         int N,
                                         int K) {
    int lcol = get_local_id(0);
    int lrow = get_local_id(1);
    int lid = lrow * RB_LX + lcol;
    int block_col = get_group_id(0) * RB_N;
    int block_row = get_group_id(1) * RB_M;

    __local float As[RB_M][RB_K];
    __local float Bs[RB_K][RB_N];

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    for (int kt = 0; kt < 512; kt += RB_K) {
        for (int idx = lid; idx < RB_M * RB_K; idx += RB_LX * RB_LY) {
            int r = idx / RB_K;
            int kk = idx - r * RB_K;
            int gr = block_row + r;
            int gk = kt + kk;
            As[r][kk] = (gr < M) ? A[gr * 512 + gk] : 0.0f;
        }
        for (int idx = lid; idx < RB_K * RB_N; idx += RB_LX * RB_LY) {
            int kk = idx / RB_N;
            int c = idx - kk * RB_N;
            int gk = kt + kk;
            int gc = block_col + c;
            Bs[kk][c] = (gc < N) ? B[gk * N + gc] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int rr = lrow * RB_THREAD_M;
        int cc = lcol * RB_THREAD_N;
        for (int kk = 0; kk < RB_K; ++kk) {
            float a0 = As[rr + 0][kk];
            float a1 = As[rr + 1][kk];
            float a2 = As[rr + 2][kk];
            float a3 = As[rr + 3][kk];
            float b0 = Bs[kk][cc + 0];
            float b1 = Bs[kk][cc + 1];
            float b2 = Bs[kk][cc + 2];
            float b3 = Bs[kk][cc + 3];
            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int row0 = block_row + lrow * RB_THREAD_M;
    int col0 = block_col + lcol * RB_THREAD_N;
    if (row0 + 0 < M && col0 + 0 < N) C[(row0 + 0) * N + col0 + 0] = acc00;
    if (row0 + 0 < M && col0 + 1 < N) C[(row0 + 0) * N + col0 + 1] = acc01;
    if (row0 + 0 < M && col0 + 2 < N) C[(row0 + 0) * N + col0 + 2] = acc02;
    if (row0 + 0 < M && col0 + 3 < N) C[(row0 + 0) * N + col0 + 3] = acc03;
    if (row0 + 1 < M && col0 + 0 < N) C[(row0 + 1) * N + col0 + 0] = acc10;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = acc11;
    if (row0 + 1 < M && col0 + 2 < N) C[(row0 + 1) * N + col0 + 2] = acc12;
    if (row0 + 1 < M && col0 + 3 < N) C[(row0 + 1) * N + col0 + 3] = acc13;
    if (row0 + 2 < M && col0 + 0 < N) C[(row0 + 2) * N + col0 + 0] = acc20;
    if (row0 + 2 < M && col0 + 1 < N) C[(row0 + 2) * N + col0 + 1] = acc21;
    if (row0 + 2 < M && col0 + 2 < N) C[(row0 + 2) * N + col0 + 2] = acc22;
    if (row0 + 2 < M && col0 + 3 < N) C[(row0 + 2) * N + col0 + 3] = acc23;
    if (row0 + 3 < M && col0 + 0 < N) C[(row0 + 3) * N + col0 + 0] = acc30;
    if (row0 + 3 < M && col0 + 1 < N) C[(row0 + 3) * N + col0 + 1] = acc31;
    if (row0 + 3 < M && col0 + 2 < N) C[(row0 + 3) * N + col0 + 2] = acc32;
    if (row0 + 3 < M && col0 + 3 < N) C[(row0 + 3) * N + col0 + 3] = acc33;
}


__kernel void matmul_transa_rb4_k512_f32(__global const float* A,
                                    __global const float* B,
                                    __global float* C,
                                    int M,
                                    int N,
                                    int K) {
    int lcol = get_local_id(0);
    int lrow = get_local_id(1);
    int lid = lrow * RB_LX + lcol;
    int block_col = get_group_id(0) * RB_N;
    int block_row = get_group_id(1) * RB_M;

    __local float As[RB_M][RB_K];
    __local float Bs[RB_K][RB_N];

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    for (int kt = 0; kt < 512; kt += RB_K) {
        for (int idx = lid; idx < RB_M * RB_K; idx += RB_LX * RB_LY) {
            int kk = idx / RB_M;
            int r = idx - kk * RB_M;
            int gr = block_row + r;
            int gk = kt + kk;
            As[r][kk] = (gr < M) ? A[gk * M + gr] : 0.0f;
        }
        for (int idx = lid; idx < RB_K * RB_N; idx += RB_LX * RB_LY) {
            int kk = idx / RB_N;
            int c = idx - kk * RB_N;
            int gk = kt + kk;
            int gc = block_col + c;
            Bs[kk][c] = (gc < N) ? B[gk * N + gc] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int rr = lrow * RB_THREAD_M;
        int cc = lcol * RB_THREAD_N;
        for (int kk = 0; kk < RB_K; ++kk) {
            float a0 = As[rr + 0][kk];
            float a1 = As[rr + 1][kk];
            float a2 = As[rr + 2][kk];
            float a3 = As[rr + 3][kk];
            float b0 = Bs[kk][cc + 0];
            float b1 = Bs[kk][cc + 1];
            float b2 = Bs[kk][cc + 2];
            float b3 = Bs[kk][cc + 3];
            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int row0 = block_row + lrow * RB_THREAD_M;
    int col0 = block_col + lcol * RB_THREAD_N;
    if (row0 + 0 < M && col0 + 0 < N) C[(row0 + 0) * N + col0 + 0] = acc00;
    if (row0 + 0 < M && col0 + 1 < N) C[(row0 + 0) * N + col0 + 1] = acc01;
    if (row0 + 0 < M && col0 + 2 < N) C[(row0 + 0) * N + col0 + 2] = acc02;
    if (row0 + 0 < M && col0 + 3 < N) C[(row0 + 0) * N + col0 + 3] = acc03;
    if (row0 + 1 < M && col0 + 0 < N) C[(row0 + 1) * N + col0 + 0] = acc10;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = acc11;
    if (row0 + 1 < M && col0 + 2 < N) C[(row0 + 1) * N + col0 + 2] = acc12;
    if (row0 + 1 < M && col0 + 3 < N) C[(row0 + 1) * N + col0 + 3] = acc13;
    if (row0 + 2 < M && col0 + 0 < N) C[(row0 + 2) * N + col0 + 0] = acc20;
    if (row0 + 2 < M && col0 + 1 < N) C[(row0 + 2) * N + col0 + 1] = acc21;
    if (row0 + 2 < M && col0 + 2 < N) C[(row0 + 2) * N + col0 + 2] = acc22;
    if (row0 + 2 < M && col0 + 3 < N) C[(row0 + 2) * N + col0 + 3] = acc23;
    if (row0 + 3 < M && col0 + 0 < N) C[(row0 + 3) * N + col0 + 0] = acc30;
    if (row0 + 3 < M && col0 + 1 < N) C[(row0 + 3) * N + col0 + 1] = acc31;
    if (row0 + 3 < M && col0 + 2 < N) C[(row0 + 3) * N + col0 + 2] = acc32;
    if (row0 + 3 < M && col0 + 3 < N) C[(row0 + 3) * N + col0 + 3] = acc33;
}

__kernel void matmul_transb_rb4_k512_f32(__global const float* A,
                                    __global const float* B,
                                    __global float* C,
                                    int M,
                                    int N,
                                    int K) {
    int lcol = get_local_id(0);
    int lrow = get_local_id(1);
    int lid = lrow * RB_LX + lcol;
    int block_col = get_group_id(0) * RB_N;
    int block_row = get_group_id(1) * RB_M;

    __local float As[RB_M][RB_K];
    __local float Bs[RB_K][RB_N];

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    for (int kt = 0; kt < 512; kt += RB_K) {
        for (int idx = lid; idx < RB_M * RB_K; idx += RB_LX * RB_LY) {
            int r = idx / RB_K;
            int kk = idx - r * RB_K;
            int gr = block_row + r;
            int gk = kt + kk;
            As[r][kk] = (gr < M) ? A[gr * 512 + gk] : 0.0f;
        }
        for (int idx = lid; idx < RB_K * RB_N; idx += RB_LX * RB_LY) {
            int c = idx / RB_K;
            int kk = idx - c * RB_K;
            int gk = kt + kk;
            int gc = block_col + c;
            Bs[kk][c] = (gc < N) ? B[gc * 512 + gk] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int rr = lrow * RB_THREAD_M;
        int cc = lcol * RB_THREAD_N;
        for (int kk = 0; kk < RB_K; ++kk) {
            float a0 = As[rr + 0][kk];
            float a1 = As[rr + 1][kk];
            float a2 = As[rr + 2][kk];
            float a3 = As[rr + 3][kk];
            float b0 = Bs[kk][cc + 0];
            float b1 = Bs[kk][cc + 1];
            float b2 = Bs[kk][cc + 2];
            float b3 = Bs[kk][cc + 3];
            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int row0 = block_row + lrow * RB_THREAD_M;
    int col0 = block_col + lcol * RB_THREAD_N;
    if (row0 + 0 < M && col0 + 0 < N) C[(row0 + 0) * N + col0 + 0] = acc00;
    if (row0 + 0 < M && col0 + 1 < N) C[(row0 + 0) * N + col0 + 1] = acc01;
    if (row0 + 0 < M && col0 + 2 < N) C[(row0 + 0) * N + col0 + 2] = acc02;
    if (row0 + 0 < M && col0 + 3 < N) C[(row0 + 0) * N + col0 + 3] = acc03;
    if (row0 + 1 < M && col0 + 0 < N) C[(row0 + 1) * N + col0 + 0] = acc10;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = acc11;
    if (row0 + 1 < M && col0 + 2 < N) C[(row0 + 1) * N + col0 + 2] = acc12;
    if (row0 + 1 < M && col0 + 3 < N) C[(row0 + 1) * N + col0 + 3] = acc13;
    if (row0 + 2 < M && col0 + 0 < N) C[(row0 + 2) * N + col0 + 0] = acc20;
    if (row0 + 2 < M && col0 + 1 < N) C[(row0 + 2) * N + col0 + 1] = acc21;
    if (row0 + 2 < M && col0 + 2 < N) C[(row0 + 2) * N + col0 + 2] = acc22;
    if (row0 + 2 < M && col0 + 3 < N) C[(row0 + 2) * N + col0 + 3] = acc23;
    if (row0 + 3 < M && col0 + 0 < N) C[(row0 + 3) * N + col0 + 0] = acc30;
    if (row0 + 3 < M && col0 + 1 < N) C[(row0 + 3) * N + col0 + 1] = acc31;
    if (row0 + 3 < M && col0 + 2 < N) C[(row0 + 3) * N + col0 + 2] = acc32;
    if (row0 + 3 < M && col0 + 3 < N) C[(row0 + 3) * N + col0 + 3] = acc33;
}


__kernel void matmul_q8_0_f32(__global const char* A,
                              __global const char* B,
                              __global float* C,
                              int M,
                              int N,
                              int K,
                              float scale_a,
                              float scale_b) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;

    int acc = 0;
    for (int k = 0; k < K; ++k) {
        acc += ((int)A[row * K + k]) * ((int)B[k * N + col]);
    }
    C[row * N + col] = ((float)acc) * scale_a * scale_b;
}

__kernel void matmul_q8_0_dot4_f32(__global const char* A,
                                   __global const char* B,
                                   __global float* C,
                                   int M,
                                   int N,
                                   int K,
                                   float scale_a,
                                   float scale_b) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;

    int acc = 0;
    int k = 0;
    for (; k + 3 < K; k += 4) {
        int a0 = (int)A[row * K + k + 0];
        int a1 = (int)A[row * K + k + 1];
        int a2 = (int)A[row * K + k + 2];
        int a3 = (int)A[row * K + k + 3];
        int b0 = (int)B[(k + 0) * N + col];
        int b1 = (int)B[(k + 1) * N + col];
        int b2 = (int)B[(k + 2) * N + col];
        int b3 = (int)B[(k + 3) * N + col];
        acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    }
    for (; k < K; ++k) {
        acc += ((int)A[row * K + k]) * ((int)B[k * N + col]);
    }
    C[row * N + col] = ((float)acc) * scale_a * scale_b;
}

__kernel void matmul_q8_0_rb4_f32(__global const char* A,
                                  __global const char* B,
                                  __global float* C,
                                  int M,
                                  int N,
                                  int K,
                                  float scale_a,
                                  float scale_b) {
    int lcol = get_local_id(0);
    int lrow = get_local_id(1);
    int lid = lrow * RB_LX + lcol;
    int block_col = get_group_id(0) * RB_N;
    int block_row = get_group_id(1) * RB_M;

    __local char As[RB_M][RB_K];
    __local char Bs[RB_K][RB_N];

    int acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0;
    int acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0;
    int acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;
    int acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0;

    for (int kt = 0; kt < K; kt += RB_K) {
        for (int idx = lid; idx < RB_M * RB_K; idx += RB_LX * RB_LY) {
            int r = idx / RB_K;
            int kk = idx - r * RB_K;
            int gr = block_row + r;
            int gk = kt + kk;
            As[r][kk] = (gr < M && gk < K) ? A[gr * K + gk] : (char)0;
        }
        for (int idx = lid; idx < RB_K * RB_N; idx += RB_LX * RB_LY) {
            int kk = idx / RB_N;
            int c = idx - kk * RB_N;
            int gk = kt + kk;
            int gc = block_col + c;
            Bs[kk][c] = (gk < K && gc < N) ? B[gk * N + gc] : (char)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int rr = lrow * RB_THREAD_M;
        int cc = lcol * RB_THREAD_N;
        for (int kk = 0; kk < RB_K; ++kk) {
            int a0 = (int)As[rr + 0][kk];
            int a1 = (int)As[rr + 1][kk];
            int a2 = (int)As[rr + 2][kk];
            int a3 = (int)As[rr + 3][kk];
            int b0 = (int)Bs[kk][cc + 0];
            int b1 = (int)Bs[kk][cc + 1];
            int b2 = (int)Bs[kk][cc + 2];
            int b3 = (int)Bs[kk][cc + 3];
            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float scale = scale_a * scale_b;
    int row0 = block_row + lrow * RB_THREAD_M;
    int col0 = block_col + lcol * RB_THREAD_N;
    if (row0 + 0 < M && col0 + 0 < N) C[(row0 + 0) * N + col0 + 0] = ((float)acc00) * scale;
    if (row0 + 0 < M && col0 + 1 < N) C[(row0 + 0) * N + col0 + 1] = ((float)acc01) * scale;
    if (row0 + 0 < M && col0 + 2 < N) C[(row0 + 0) * N + col0 + 2] = ((float)acc02) * scale;
    if (row0 + 0 < M && col0 + 3 < N) C[(row0 + 0) * N + col0 + 3] = ((float)acc03) * scale;
    if (row0 + 1 < M && col0 + 0 < N) C[(row0 + 1) * N + col0 + 0] = ((float)acc10) * scale;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = ((float)acc11) * scale;
    if (row0 + 1 < M && col0 + 2 < N) C[(row0 + 1) * N + col0 + 2] = ((float)acc12) * scale;
    if (row0 + 1 < M && col0 + 3 < N) C[(row0 + 1) * N + col0 + 3] = ((float)acc13) * scale;
    if (row0 + 2 < M && col0 + 0 < N) C[(row0 + 2) * N + col0 + 0] = ((float)acc20) * scale;
    if (row0 + 2 < M && col0 + 1 < N) C[(row0 + 2) * N + col0 + 1] = ((float)acc21) * scale;
    if (row0 + 2 < M && col0 + 2 < N) C[(row0 + 2) * N + col0 + 2] = ((float)acc22) * scale;
    if (row0 + 2 < M && col0 + 3 < N) C[(row0 + 2) * N + col0 + 3] = ((float)acc23) * scale;
    if (row0 + 3 < M && col0 + 0 < N) C[(row0 + 3) * N + col0 + 0] = ((float)acc30) * scale;
    if (row0 + 3 < M && col0 + 1 < N) C[(row0 + 3) * N + col0 + 1] = ((float)acc31) * scale;
    if (row0 + 3 < M && col0 + 2 < N) C[(row0 + 3) * N + col0 + 2] = ((float)acc32) * scale;
    if (row0 + 3 < M && col0 + 3 < N) C[(row0 + 3) * N + col0 + 3] = ((float)acc33) * scale;
}

inline int q4_0_load(__global const uchar* x, int idx) {
    uchar packed = x[idx >> 1];
    uchar code = (idx & 1) ? ((packed >> 4) & 15) : (packed & 15);
    return ((int)code) - 8;
}

inline int q4_0_col_load(__global const uchar* x, int col, int K, int k) {
    int idx = col * K + k;
    uchar packed = x[idx >> 1];
    uchar code = (idx & 1) ? ((packed >> 4) & 15) : (packed & 15);
    return ((int)code) - 8;
}

inline int q4_0_tile8_cols(int N, int col0) {
    int remaining = N - col0;
    return remaining < 8 ? remaining : 8;
}

inline int q4_0_tile8_elem_offset(int N,
                                  int blocks_per_col,
                                  int block_b,
                                  int col,
                                  int k) {
    const int tile = col >> 3;
    const int col0 = tile << 3;
    const int tcol = col & 7;
    const int tile_cols = q4_0_tile8_cols(N, col0);
    const int block = k / block_b;
    const int inner = k - block * block_b;
    return tile * blocks_per_col * block_b * 8 + (block * block_b + inner) * tile_cols + tcol;
}

inline int q4_0_tile8_load(__global const uchar* x,
                           int N,
                           int blocks_per_col,
                           int block_b,
                           int col,
                           int k) {
    int idx = q4_0_tile8_elem_offset(N, blocks_per_col, block_b, col, k);
    uchar packed = x[idx >> 1];
    uchar code = (idx & 1) ? ((packed >> 4) & 15) : (packed & 15);
    return ((int)code) - 8;
}

inline float q4_0_tile8_scale(__global const float* scales,
                              int N,
                              int blocks_per_col,
                              int block_b,
                              int col,
                              int k) {
    const int tile = col >> 3;
    const int col0 = tile << 3;
    const int tcol = col & 7;
    const int tile_cols = q4_0_tile8_cols(N, col0);
    const int block = k / block_b;
    return scales[tile * blocks_per_col * 8 + block * tile_cols + tcol];
}

inline void q4_0_tile8_accum8(__global const uchar* B,
                              __global const float* scales_b,
                              int N,
                              int blocks_per_col,
                              int block_b,
                              int col0,
                              int k,
                              float av,
                              __private float* acc) {
    const int group = col0 >> 3;
    const int tile_cols = q4_0_tile8_cols(N, col0);
    const int block = k / block_b;
    const int inner = k - block * block_b;
    const int elem_base = group * blocks_per_col * block_b * 8 + (block * block_b + inner) * tile_cols;
    const int scale_base = group * blocks_per_col * 8 + block * tile_cols;
    if (tile_cols == 8) {
        const int byte_base = elem_base >> 1;
        const uchar p0 = B[byte_base + 0];
        const uchar p1 = B[byte_base + 1];
        const uchar p2 = B[byte_base + 2];
        const uchar p3 = B[byte_base + 3];
        acc[0] += av * ((float)((int)(p0 & 15) - 8)) * scales_b[scale_base + 0];
        acc[1] += av * ((float)((int)((p0 >> 4) & 15) - 8)) * scales_b[scale_base + 1];
        acc[2] += av * ((float)((int)(p1 & 15) - 8)) * scales_b[scale_base + 2];
        acc[3] += av * ((float)((int)((p1 >> 4) & 15) - 8)) * scales_b[scale_base + 3];
        acc[4] += av * ((float)((int)(p2 & 15) - 8)) * scales_b[scale_base + 4];
        acc[5] += av * ((float)((int)((p2 >> 4) & 15) - 8)) * scales_b[scale_base + 5];
        acc[6] += av * ((float)((int)(p3 & 15) - 8)) * scales_b[scale_base + 6];
        acc[7] += av * ((float)((int)((p3 >> 4) & 15) - 8)) * scales_b[scale_base + 7];
    } else {
        for (int j = 0; j < 8; ++j) {
            if (j < tile_cols) {
                acc[j] += av * ((float)q4_0_tile8_load(B, N, blocks_per_col, block_b, col0 + j, k)) *
                          scales_b[scale_base + j];
            }
        }
    }
}

__kernel void matmul_q4_0_f32(__global const uchar* A,
                              __global const uchar* B,
                              __global float* C,
                              int M,
                              int N,
                              int K,
                              float scale_a,
                              float scale_b) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;

    int acc = 0;
    for (int k = 0; k < K; ++k) {
        int av = q4_0_load(A, row * K + k);
        int bv = q4_0_load(B, k * N + col);
        acc += av * bv;
    }
    C[row * N + col] = ((float)acc) * scale_a * scale_b;
}

__kernel void matmul_q4_0_rb4_f32(__global const uchar* A,
                                  __global const uchar* B,
                                  __global float* C,
                                  int M,
                                  int N,
                                  int K,
                                  float scale_a,
                                  float scale_b) {
    int lcol = get_local_id(0);
    int lrow = get_local_id(1);
    int block_col = get_group_id(0) * RB_N;
    int block_row = get_group_id(1) * RB_M;
    int row0 = block_row + lrow * RB_THREAD_M;
    int col0 = block_col + lcol * RB_THREAD_N;
    int acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0;
    int acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0;
    int acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;
    int acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0;
    for (int k = 0; k < K; ++k) {
        int a0 = (row0 + 0 < M) ? q4_0_load(A, (row0 + 0) * K + k) : 0;
        int a1 = (row0 + 1 < M) ? q4_0_load(A, (row0 + 1) * K + k) : 0;
        int a2 = (row0 + 2 < M) ? q4_0_load(A, (row0 + 2) * K + k) : 0;
        int a3 = (row0 + 3 < M) ? q4_0_load(A, (row0 + 3) * K + k) : 0;
        int b0 = (col0 + 0 < N) ? q4_0_load(B, k * N + col0 + 0) : 0;
        int b1 = (col0 + 1 < N) ? q4_0_load(B, k * N + col0 + 1) : 0;
        int b2 = (col0 + 2 < N) ? q4_0_load(B, k * N + col0 + 2) : 0;
        int b3 = (col0 + 3 < N) ? q4_0_load(B, k * N + col0 + 3) : 0;
        acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
        acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
        acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
        acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
    }
    float scale = scale_a * scale_b;
    if (row0 + 0 < M && col0 + 0 < N) C[(row0 + 0) * N + col0 + 0] = ((float)acc00) * scale;
    if (row0 + 0 < M && col0 + 1 < N) C[(row0 + 0) * N + col0 + 1] = ((float)acc01) * scale;
    if (row0 + 0 < M && col0 + 2 < N) C[(row0 + 0) * N + col0 + 2] = ((float)acc02) * scale;
    if (row0 + 0 < M && col0 + 3 < N) C[(row0 + 0) * N + col0 + 3] = ((float)acc03) * scale;
    if (row0 + 1 < M && col0 + 0 < N) C[(row0 + 1) * N + col0 + 0] = ((float)acc10) * scale;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = ((float)acc11) * scale;
    if (row0 + 1 < M && col0 + 2 < N) C[(row0 + 1) * N + col0 + 2] = ((float)acc12) * scale;
    if (row0 + 1 < M && col0 + 3 < N) C[(row0 + 1) * N + col0 + 3] = ((float)acc13) * scale;
    if (row0 + 2 < M && col0 + 0 < N) C[(row0 + 2) * N + col0 + 0] = ((float)acc20) * scale;
    if (row0 + 2 < M && col0 + 1 < N) C[(row0 + 2) * N + col0 + 1] = ((float)acc21) * scale;
    if (row0 + 2 < M && col0 + 2 < N) C[(row0 + 2) * N + col0 + 2] = ((float)acc22) * scale;
    if (row0 + 2 < M && col0 + 3 < N) C[(row0 + 2) * N + col0 + 3] = ((float)acc23) * scale;
    if (row0 + 3 < M && col0 + 0 < N) C[(row0 + 3) * N + col0 + 0] = ((float)acc30) * scale;
    if (row0 + 3 < M && col0 + 1 < N) C[(row0 + 3) * N + col0 + 1] = ((float)acc31) * scale;
    if (row0 + 3 < M && col0 + 2 < N) C[(row0 + 3) * N + col0 + 2] = ((float)acc32) * scale;
    if (row0 + 3 < M && col0 + 3 < N) C[(row0 + 3) * N + col0 + 3] = ((float)acc33) * scale;
}

__kernel void matmul_q4_0_dot4_rb4_f32(__global const uchar* A,
                                       __global const uchar* B,
                                       __global float* C,
                                       int M,
                                       int N,
                                       int K,
                                       float scale_a,
                                       float scale_b) {
    int lcol = get_local_id(0);
    int lrow = get_local_id(1);
    int block_col = get_group_id(0) * RB_N;
    int block_row = get_group_id(1) * RB_M;
    int row0 = block_row + lrow * RB_THREAD_M;
    int col0 = block_col + lcol * RB_THREAD_N;
    int acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0;
    int acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0;
    int acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;
    int acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0;
    for (int k = 0; k < K; k += 4) {
        for (int kk = 0; kk < 4; ++kk) {
            int src_k = k + kk;
            int a0 = (row0 + 0 < M) ? q4_0_load(A, (row0 + 0) * K + src_k) : 0;
            int a1 = (row0 + 1 < M) ? q4_0_load(A, (row0 + 1) * K + src_k) : 0;
            int a2 = (row0 + 2 < M) ? q4_0_load(A, (row0 + 2) * K + src_k) : 0;
            int a3 = (row0 + 3 < M) ? q4_0_load(A, (row0 + 3) * K + src_k) : 0;
            int b0 = (col0 + 0 < N) ? q4_0_load(B, src_k * N + col0 + 0) : 0;
            int b1 = (col0 + 1 < N) ? q4_0_load(B, src_k * N + col0 + 1) : 0;
            int b2 = (col0 + 2 < N) ? q4_0_load(B, src_k * N + col0 + 2) : 0;
            int b3 = (col0 + 3 < N) ? q4_0_load(B, src_k * N + col0 + 3) : 0;
            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }
    }
    float scale = scale_a * scale_b;
    if (row0 + 0 < M && col0 + 0 < N) C[(row0 + 0) * N + col0 + 0] = ((float)acc00) * scale;
    if (row0 + 0 < M && col0 + 1 < N) C[(row0 + 0) * N + col0 + 1] = ((float)acc01) * scale;
    if (row0 + 0 < M && col0 + 2 < N) C[(row0 + 0) * N + col0 + 2] = ((float)acc02) * scale;
    if (row0 + 0 < M && col0 + 3 < N) C[(row0 + 0) * N + col0 + 3] = ((float)acc03) * scale;
    if (row0 + 1 < M && col0 + 0 < N) C[(row0 + 1) * N + col0 + 0] = ((float)acc10) * scale;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = ((float)acc11) * scale;
    if (row0 + 1 < M && col0 + 2 < N) C[(row0 + 1) * N + col0 + 2] = ((float)acc12) * scale;
    if (row0 + 1 < M && col0 + 3 < N) C[(row0 + 1) * N + col0 + 3] = ((float)acc13) * scale;
    if (row0 + 2 < M && col0 + 0 < N) C[(row0 + 2) * N + col0 + 0] = ((float)acc20) * scale;
    if (row0 + 2 < M && col0 + 1 < N) C[(row0 + 2) * N + col0 + 1] = ((float)acc21) * scale;
    if (row0 + 2 < M && col0 + 2 < N) C[(row0 + 2) * N + col0 + 2] = ((float)acc22) * scale;
    if (row0 + 2 < M && col0 + 3 < N) C[(row0 + 2) * N + col0 + 3] = ((float)acc23) * scale;
    if (row0 + 3 < M && col0 + 0 < N) C[(row0 + 3) * N + col0 + 0] = ((float)acc30) * scale;
    if (row0 + 3 < M && col0 + 1 < N) C[(row0 + 3) * N + col0 + 1] = ((float)acc31) * scale;
    if (row0 + 3 < M && col0 + 2 < N) C[(row0 + 3) * N + col0 + 2] = ((float)acc32) * scale;
    if (row0 + 3 < M && col0 + 3 < N) C[(row0 + 3) * N + col0 + 3] = ((float)acc33) * scale;
}

__kernel void matmul_q8_q4_0_f32(__global const char* A,
                                 __global const uchar* B,
                                 __global float* C,
                                 int M,
                                 int N,
                                 int K,
                                 float scale_a,
                                 float scale_b) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;

    int acc = 0;
    for (int k = 0; k < K; ++k) {
        acc += ((int)A[row * K + k]) * q4_0_load(B, k * N + col);
    }
    C[row * N + col] = ((float)acc) * scale_a * scale_b;
}

__kernel void matmul_q4_q8_0_f32(__global const uchar* A,
                                 __global const char* B,
                                 __global float* C,
                                 int M,
                                 int N,
                                 int K,
                                 float scale_a,
                                 float scale_b) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;

    int acc = 0;
    for (int k = 0; k < K; ++k) {
        acc += q4_0_load(A, row * K + k) * ((int)B[k * N + col]);
    }
    C[row * N + col] = ((float)acc) * scale_a * scale_b;
}

__kernel void matmul_q8_q4_rb4_f32(__global const char* A,
                                   __global const uchar* B,
                                   __global float* C,
                                   int M,
                                   int N,
                                   int K,
                                   float scale_a,
                                   float scale_b) {
    int lcol = get_local_id(0), lrow = get_local_id(1);
    int row0 = get_group_id(1) * RB_M + lrow * RB_THREAD_M;
    int col0 = get_group_id(0) * RB_N + lcol * RB_THREAD_N;
    int acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0;
    int acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0;
    int acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;
    int acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0;
    for (int k = 0; k < K; ++k) {
        int a0 = (row0 + 0 < M) ? (int)A[(row0 + 0) * K + k] : 0;
        int a1 = (row0 + 1 < M) ? (int)A[(row0 + 1) * K + k] : 0;
        int a2 = (row0 + 2 < M) ? (int)A[(row0 + 2) * K + k] : 0;
        int a3 = (row0 + 3 < M) ? (int)A[(row0 + 3) * K + k] : 0;
        int b0 = (col0 + 0 < N) ? q4_0_load(B, k * N + col0 + 0) : 0;
        int b1 = (col0 + 1 < N) ? q4_0_load(B, k * N + col0 + 1) : 0;
        int b2 = (col0 + 2 < N) ? q4_0_load(B, k * N + col0 + 2) : 0;
        int b3 = (col0 + 3 < N) ? q4_0_load(B, k * N + col0 + 3) : 0;
        acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
        acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
        acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
        acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
    }
    float scale = scale_a * scale_b;
    if (row0 + 0 < M && col0 + 0 < N) C[(row0 + 0) * N + col0 + 0] = ((float)acc00) * scale;
    if (row0 + 0 < M && col0 + 1 < N) C[(row0 + 0) * N + col0 + 1] = ((float)acc01) * scale;
    if (row0 + 0 < M && col0 + 2 < N) C[(row0 + 0) * N + col0 + 2] = ((float)acc02) * scale;
    if (row0 + 0 < M && col0 + 3 < N) C[(row0 + 0) * N + col0 + 3] = ((float)acc03) * scale;
    if (row0 + 1 < M && col0 + 0 < N) C[(row0 + 1) * N + col0 + 0] = ((float)acc10) * scale;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = ((float)acc11) * scale;
    if (row0 + 1 < M && col0 + 2 < N) C[(row0 + 1) * N + col0 + 2] = ((float)acc12) * scale;
    if (row0 + 1 < M && col0 + 3 < N) C[(row0 + 1) * N + col0 + 3] = ((float)acc13) * scale;
    if (row0 + 2 < M && col0 + 0 < N) C[(row0 + 2) * N + col0 + 0] = ((float)acc20) * scale;
    if (row0 + 2 < M && col0 + 1 < N) C[(row0 + 2) * N + col0 + 1] = ((float)acc21) * scale;
    if (row0 + 2 < M && col0 + 2 < N) C[(row0 + 2) * N + col0 + 2] = ((float)acc22) * scale;
    if (row0 + 2 < M && col0 + 3 < N) C[(row0 + 2) * N + col0 + 3] = ((float)acc23) * scale;
    if (row0 + 3 < M && col0 + 0 < N) C[(row0 + 3) * N + col0 + 0] = ((float)acc30) * scale;
    if (row0 + 3 < M && col0 + 1 < N) C[(row0 + 3) * N + col0 + 1] = ((float)acc31) * scale;
    if (row0 + 3 < M && col0 + 2 < N) C[(row0 + 3) * N + col0 + 2] = ((float)acc32) * scale;
    if (row0 + 3 < M && col0 + 3 < N) C[(row0 + 3) * N + col0 + 3] = ((float)acc33) * scale;
}

__kernel void matmul_q4_q8_rb4_f32(__global const uchar* A,
                                   __global const char* B,
                                   __global float* C,
                                   int M,
                                   int N,
                                   int K,
                                   float scale_a,
                                   float scale_b) {
    int lcol = get_local_id(0), lrow = get_local_id(1);
    int row0 = get_group_id(1) * RB_M + lrow * RB_THREAD_M;
    int col0 = get_group_id(0) * RB_N + lcol * RB_THREAD_N;
    int acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0;
    int acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0;
    int acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;
    int acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0;
    for (int k = 0; k < K; ++k) {
        int a0 = (row0 + 0 < M) ? q4_0_load(A, (row0 + 0) * K + k) : 0;
        int a1 = (row0 + 1 < M) ? q4_0_load(A, (row0 + 1) * K + k) : 0;
        int a2 = (row0 + 2 < M) ? q4_0_load(A, (row0 + 2) * K + k) : 0;
        int a3 = (row0 + 3 < M) ? q4_0_load(A, (row0 + 3) * K + k) : 0;
        int b0 = (col0 + 0 < N) ? (int)B[k * N + col0 + 0] : 0;
        int b1 = (col0 + 1 < N) ? (int)B[k * N + col0 + 1] : 0;
        int b2 = (col0 + 2 < N) ? (int)B[k * N + col0 + 2] : 0;
        int b3 = (col0 + 3 < N) ? (int)B[k * N + col0 + 3] : 0;
        acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
        acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
        acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
        acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
    }
    float scale = scale_a * scale_b;
    if (row0 + 0 < M && col0 + 0 < N) C[(row0 + 0) * N + col0 + 0] = ((float)acc00) * scale;
    if (row0 + 0 < M && col0 + 1 < N) C[(row0 + 0) * N + col0 + 1] = ((float)acc01) * scale;
    if (row0 + 0 < M && col0 + 2 < N) C[(row0 + 0) * N + col0 + 2] = ((float)acc02) * scale;
    if (row0 + 0 < M && col0 + 3 < N) C[(row0 + 0) * N + col0 + 3] = ((float)acc03) * scale;
    if (row0 + 1 < M && col0 + 0 < N) C[(row0 + 1) * N + col0 + 0] = ((float)acc10) * scale;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = ((float)acc11) * scale;
    if (row0 + 1 < M && col0 + 2 < N) C[(row0 + 1) * N + col0 + 2] = ((float)acc12) * scale;
    if (row0 + 1 < M && col0 + 3 < N) C[(row0 + 1) * N + col0 + 3] = ((float)acc13) * scale;
    if (row0 + 2 < M && col0 + 0 < N) C[(row0 + 2) * N + col0 + 0] = ((float)acc20) * scale;
    if (row0 + 2 < M && col0 + 1 < N) C[(row0 + 2) * N + col0 + 1] = ((float)acc21) * scale;
    if (row0 + 2 < M && col0 + 2 < N) C[(row0 + 2) * N + col0 + 2] = ((float)acc22) * scale;
    if (row0 + 2 < M && col0 + 3 < N) C[(row0 + 2) * N + col0 + 3] = ((float)acc23) * scale;
    if (row0 + 3 < M && col0 + 0 < N) C[(row0 + 3) * N + col0 + 0] = ((float)acc30) * scale;
    if (row0 + 3 < M && col0 + 1 < N) C[(row0 + 3) * N + col0 + 1] = ((float)acc31) * scale;
    if (row0 + 3 < M && col0 + 2 < N) C[(row0 + 3) * N + col0 + 2] = ((float)acc32) * scale;
    if (row0 + 3 < M && col0 + 3 < N) C[(row0 + 3) * N + col0 + 3] = ((float)acc33) * scale;
}

inline float quant_matmul_scale(__global const float* scales,
                                float scalar_scale,
                                int mode,
                                int row,
                                int col,
                                int cols,
                                int block_size) {
    if (mode == 1) return scales[row];
    if (mode == 2) return scales[col];
    if (mode == 3) return scales[(row * cols + col) / block_size];
    return scalar_scale;
}

inline ushort gguf_load_u16_le(__global const uchar* x, int off) {
    return (ushort)(((ushort)x[off]) | (((ushort)x[off + 1]) << 8));
}

inline float gguf_f16_to_f32(ushort h) {
    uint sign = ((uint)h & 0x8000u) << 16;
    uint exp = ((uint)h >> 10) & 0x1fu;
    uint mant = ((uint)h) & 0x03ffu;
    if (exp == 0u) {
        if (mant == 0u) return as_float(sign);
        float value = ((float)mant) * 5.9604644775390625e-8f; // 2^-24
        return sign ? -value : value;
    }
    if (exp == 31u) return as_float(sign | 0x7f800000u | (mant << 13));
    uint bits = sign | ((exp + 112u) << 23) | (mant << 13);
    return as_float(bits);
}

inline void gguf_k_scale_min(int j, __global const uchar* q, int* d, int* m) {
    if (j < 4) {
        *d = ((int)q[j]) & 63;
        *m = ((int)q[j + 4]) & 63;
    } else {
        *d = (((int)q[j + 4]) & 15) | ((((int)q[j - 4]) >> 6) << 4);
        *m = (((int)q[j + 4]) >> 4) | ((((int)q[j]) >> 6) << 4);
    }
}

inline float gguf_q4_k_value(__global const uchar* B, int idx) {
    int block = idx >> 8; // / 256
    int loc = idx & 255;
    int sub = loc >> 5; // 32-element K sub-block
    int l = loc & 31;
    int group = sub >> 1;
    int base = block * 144;
    __global const uchar* scales = B + base + 4;
    __global const uchar* qs = B + base + 16 + group * 32;
    int sc = 0;
    int mn = 0;
    gguf_k_scale_min(sub, scales, &sc, &mn);
    int code = (sub & 1) ? (((int)qs[l]) >> 4) : (((int)qs[l]) & 15);
    float d = gguf_f16_to_f32(gguf_load_u16_le(B, base));
    float dmin = gguf_f16_to_f32(gguf_load_u16_le(B, base + 2));
    return d * ((float)sc) * ((float)code) - dmin * ((float)mn);
}

inline float gguf_q5_k_value(__global const uchar* B, int idx) {
    int block = idx >> 8; // / 256
    int loc = idx & 255;
    int sub = loc >> 5;
    int l = loc & 31;
    int group = sub >> 1;
    int base = block * 176;
    __global const uchar* scales = B + base + 4;
    __global const uchar* qh = B + base + 16;
    __global const uchar* ql = B + base + 48 + group * 32;
    int sc = 0;
    int mn = 0;
    gguf_k_scale_min(sub, scales, &sc, &mn);
    int high_bit = 1 << sub;
    int code = (sub & 1) ? (((int)ql[l]) >> 4) : (((int)ql[l]) & 15);
    if ((((int)qh[l]) & high_bit) != 0) code += 16;
    float d = gguf_f16_to_f32(gguf_load_u16_le(B, base));
    float dmin = gguf_f16_to_f32(gguf_load_u16_le(B, base + 2));
    return d * ((float)sc) * ((float)code) - dmin * ((float)mn);
}

inline int gguf_i8_from_u8(uchar v) {
    int x = (int)v;
    return x >= 128 ? x - 256 : x;
}

inline float gguf_q6_k_value(__global const uchar* B, int idx) {
    int block = idx >> 8; // / 256
    int loc = idx & 255;
    int half_idx = loc >> 7; // two 128-value halves per block
    int l128 = loc & 127;
    int base = block * 210;
    __global const uchar* ql = B + base + half_idx * 64;
    __global const uchar* qh = B + base + 128 + half_idx * 32;
    __global const uchar* sc = B + base + 192 + half_idx * 8;
    int l = 0;
    int code = 0;
    int scale_idx = 0;
    if (l128 < 32) {
        l = l128;
        code = ((int)(ql[l] & 0x0fu)) | ((((int)(qh[l] >> 0)) & 0x03) << 4);
        scale_idx = (l >> 4) + 0;
    } else if (l128 < 64) {
        l = l128 - 32;
        code = ((int)(ql[l + 32] & 0x0fu)) | ((((int)(qh[l] >> 2)) & 0x03) << 4);
        scale_idx = (l >> 4) + 2;
    } else if (l128 < 96) {
        l = l128 - 64;
        code = ((int)((ql[l] >> 4) & 0x0fu)) | ((((int)(qh[l] >> 4)) & 0x03) << 4);
        scale_idx = (l >> 4) + 4;
    } else {
        l = l128 - 96;
        code = ((int)((ql[l + 32] >> 4) & 0x0fu)) | ((((int)(qh[l] >> 6)) & 0x03) << 4);
        scale_idx = (l >> 4) + 6;
    }
    code -= 32;
    float d = gguf_f16_to_f32(gguf_load_u16_le(B, base + 208));
    return d * ((float)gguf_i8_from_u8(sc[scale_idx])) * ((float)code);
}

__kernel void matmul_q8_q4_k_f32(__global const char* A,
                                 __global const uchar* B,
                                 __global float* C,
                                 int M,
                                 int N,
                                 int K,
                                 float scale_a,
                                 __global const float* scales_a,
                                 int mode_a,
                                 int block_a) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float sa = quant_matmul_scale(scales_a, scale_a, mode_a, row, k, K, block_a);
        acc += ((float)((int)A[row * K + k])) * sa * gguf_q4_k_value(B, k * N + col);
    }
    C[row * N + col] = acc;
}

__kernel void matmul_q8_q5_k_f32(__global const char* A,
                                 __global const uchar* B,
                                 __global float* C,
                                 int M,
                                 int N,
                                 int K,
                                 float scale_a,
                                 __global const float* scales_a,
                                 int mode_a,
                                 int block_a) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float sa = quant_matmul_scale(scales_a, scale_a, mode_a, row, k, K, block_a);
        acc += ((float)((int)A[row * K + k])) * sa * gguf_q5_k_value(B, k * N + col);
    }
    C[row * N + col] = acc;
}

__kernel void matmul_q8_q6_k_f32(__global const char* A,
                                 __global const uchar* B,
                                 __global float* C,
                                 int M,
                                 int N,
                                 int K,
                                 float scale_a,
                                 __global const float* scales_a,
                                 int mode_a,
                                 int block_a) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float sa = quant_matmul_scale(scales_a, scale_a, mode_a, row, k, K, block_a);
        acc += ((float)((int)A[row * K + k])) * sa * gguf_q6_k_value(B, k * N + col);
    }
    C[row * N + col] = acc;
}

__kernel void matmul_f32_q4_k_m1_f32(__global const float* A,
                                     __global const uchar* B,
                                     __global float* C,
                                     int N,
                                     int K) {
    int col = get_global_id(0);
    if (col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[k] * gguf_q4_k_value(B, k * N + col);
    }
    C[col] = acc;
}

__kernel void matmul_f32_q5_k_m1_f32(__global const float* A,
                                     __global const uchar* B,
                                     __global float* C,
                                     int N,
                                     int K) {
    int col = get_global_id(0);
    if (col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[k] * gguf_q5_k_value(B, k * N + col);
    }
    C[col] = acc;
}

__kernel void matmul_f32_q6_k_m1_f32(__global const float* A,
                                     __global const uchar* B,
                                     __global float* C,
                                     int N,
                                     int K) {
    int col = get_global_id(0);
    if (col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[k] * gguf_q6_k_value(B, k * N + col);
    }
    C[col] = acc;
}

__kernel void matmul_f32_q4_k_m1_wg_f32(__global const float* A,
                                        __global const uchar* B,
                                        __global float* C,
                                        int N,
                                        int K,
                                        __local float* scratch) {
    int col = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    if (col >= N) return;
    float acc = 0.0f;
    for (int k = lid; k < K; k += local_size) {
        acc += A[k] * gguf_q4_k_value(B, k * N + col);
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) C[col] = scratch[0];
}

__kernel void matmul_f32_q5_k_m1_wg_f32(__global const float* A,
                                        __global const uchar* B,
                                        __global float* C,
                                        int N,
                                        int K,
                                        __local float* scratch) {
    int col = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    if (col >= N) return;
    float acc = 0.0f;
    for (int k = lid; k < K; k += local_size) {
        acc += A[k] * gguf_q5_k_value(B, k * N + col);
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) C[col] = scratch[0];
}

__kernel void matmul_f32_q6_k_m1_wg_f32(__global const float* A,
                                        __global const uchar* B,
                                        __global float* C,
                                        int N,
                                        int K,
                                        __local float* scratch) {
    int col = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    if (col >= N) return;
    float acc = 0.0f;
    for (int k = lid; k < K; k += local_size) {
        acc += A[k] * gguf_q6_k_value(B, k * N + col);
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) C[col] = scratch[0];
}

__kernel void matmul_f32_q4_k_m1_wg4_f32(__global const float* A,
                                         __global const uchar* B,
                                         __global float* C,
                                         int N,
                                         int K,
                                         __local float* scratch) {
    int group = get_group_id(0);
    int lid = get_local_id(0);
    int slot = lid >> 5;       // 4 output columns per workgroup, 32 lanes each
    int lane = lid & 31;
    int col = group * 4 + slot;
    float acc = 0.0f;
    if (col < N) {
        for (int k = lane; k < K; k += 32) {
            acc += A[k] * gguf_q4_k_value(B, k * N + col);
        }
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    int base = slot << 5;
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[base + lane] += scratch[base + lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && col < N) C[col] = scratch[base];
}

__kernel void matmul_f32_q5_k_m1_wg4_f32(__global const float* A,
                                         __global const uchar* B,
                                         __global float* C,
                                         int N,
                                         int K,
                                         __local float* scratch) {
    int group = get_group_id(0);
    int lid = get_local_id(0);
    int slot = lid >> 5;
    int lane = lid & 31;
    int col = group * 4 + slot;
    float acc = 0.0f;
    if (col < N) {
        for (int k = lane; k < K; k += 32) {
            acc += A[k] * gguf_q5_k_value(B, k * N + col);
        }
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    int base = slot << 5;
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[base + lane] += scratch[base + lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && col < N) C[col] = scratch[base];
}

__kernel void matmul_f32_q6_k_m1_wg4_f32(__global const float* A,
                                         __global const uchar* B,
                                         __global float* C,
                                         int N,
                                         int K,
                                         __local float* scratch) {
    int group = get_group_id(0);
    int lid = get_local_id(0);
    int slot = lid >> 5;
    int lane = lid & 31;
    int col = group * 4 + slot;
    float acc = 0.0f;
    if (col < N) {
        for (int k = lane; k < K; k += 32) {
            acc += A[k] * gguf_q6_k_value(B, k * N + col);
        }
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    int base = slot << 5;
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[base + lane] += scratch[base + lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && col < N) C[col] = scratch[base];
}

__kernel void matmul_f32_q4_k_m1_2out_f32(__global const float* A,
                                          __global const uchar* B0,
                                          __global const uchar* B1,
                                          __global float* C0,
                                          __global float* C1,
                                          int N0,
                                          int N1,
                                          int K) {
    int gid = get_global_id(0);
    int total = N0 + N1;
    if (gid >= total) return;
    __global const uchar* B = gid < N0 ? B0 : B1;
    __global float* C = gid < N0 ? C0 : C1;
    int N = gid < N0 ? N0 : N1;
    int col = gid < N0 ? gid : gid - N0;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[k] * gguf_q4_k_value(B, k * N + col);
    }
    C[col] = acc;
}

__kernel void matmul_f32_q5_k_m1_2out_f32(__global const float* A,
                                          __global const uchar* B0,
                                          __global const uchar* B1,
                                          __global float* C0,
                                          __global float* C1,
                                          int N0,
                                          int N1,
                                          int K) {
    int gid = get_global_id(0);
    int total = N0 + N1;
    if (gid >= total) return;
    __global const uchar* B = gid < N0 ? B0 : B1;
    __global float* C = gid < N0 ? C0 : C1;
    int N = gid < N0 ? N0 : N1;
    int col = gid < N0 ? gid : gid - N0;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[k] * gguf_q5_k_value(B, k * N + col);
    }
    C[col] = acc;
}

__kernel void matmul_f32_q6_k_m1_2out_f32(__global const float* A,
                                          __global const uchar* B0,
                                          __global const uchar* B1,
                                          __global float* C0,
                                          __global float* C1,
                                          int N0,
                                          int N1,
                                          int K) {
    int gid = get_global_id(0);
    int total = N0 + N1;
    if (gid >= total) return;
    __global const uchar* B = gid < N0 ? B0 : B1;
    __global float* C = gid < N0 ? C0 : C1;
    int N = gid < N0 ? N0 : N1;
    int col = gid < N0 ? gid : gid - N0;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[k] * gguf_q6_k_value(B, k * N + col);
    }
    C[col] = acc;
}

__kernel void matmul_f32_q4_k_m1_2out_wg_f32(__global const float* A,
                                             __global const uchar* B0,
                                             __global const uchar* B1,
                                             __global float* C0,
                                             __global float* C1,
                                             int N0,
                                             int N1,
                                             int K,
                                             __local float* scratch) {
    int gid = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    int total = N0 + N1;
    if (gid >= total) return;
    __global const uchar* B = gid < N0 ? B0 : B1;
    __global float* C = gid < N0 ? C0 : C1;
    int N = gid < N0 ? N0 : N1;
    int col = gid < N0 ? gid : gid - N0;
    float acc = 0.0f;
    for (int k = lid; k < K; k += local_size) {
        acc += A[k] * gguf_q4_k_value(B, k * N + col);
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) C[col] = scratch[0];
}

__kernel void matmul_f32_q5_k_m1_2out_wg_f32(__global const float* A,
                                             __global const uchar* B0,
                                             __global const uchar* B1,
                                             __global float* C0,
                                             __global float* C1,
                                             int N0,
                                             int N1,
                                             int K,
                                             __local float* scratch) {
    int gid = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    int total = N0 + N1;
    if (gid >= total) return;
    __global const uchar* B = gid < N0 ? B0 : B1;
    __global float* C = gid < N0 ? C0 : C1;
    int N = gid < N0 ? N0 : N1;
    int col = gid < N0 ? gid : gid - N0;
    float acc = 0.0f;
    for (int k = lid; k < K; k += local_size) {
        acc += A[k] * gguf_q5_k_value(B, k * N + col);
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) C[col] = scratch[0];
}

__kernel void matmul_f32_q6_k_m1_2out_wg_f32(__global const float* A,
                                             __global const uchar* B0,
                                             __global const uchar* B1,
                                             __global float* C0,
                                             __global float* C1,
                                             int N0,
                                             int N1,
                                             int K,
                                             __local float* scratch) {
    int gid = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    int total = N0 + N1;
    if (gid >= total) return;
    __global const uchar* B = gid < N0 ? B0 : B1;
    __global float* C = gid < N0 ? C0 : C1;
    int N = gid < N0 ? N0 : N1;
    int col = gid < N0 ? gid : gid - N0;
    float acc = 0.0f;
    for (int k = lid; k < K; k += local_size) {
        acc += A[k] * gguf_q6_k_value(B, k * N + col);
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) C[col] = scratch[0];
}

__kernel void matmul_f32_q4_k_m1_2out_wg4_f32(__global const float* A,
                                              __global const uchar* B0,
                                              __global const uchar* B1,
                                              __global float* C0,
                                              __global float* C1,
                                              int N0,
                                              int N1,
                                              int K,
                                              __local float* scratch) {
    int group = get_group_id(0);
    int lid = get_local_id(0);
    int slot = lid >> 5;
    int lane = lid & 31;
    int gid = group * 4 + slot;
    int total = N0 + N1;
    __global const uchar* B = gid < N0 ? B0 : B1;
    __global float* C = gid < N0 ? C0 : C1;
    int N = gid < N0 ? N0 : N1;
    int col = gid < N0 ? gid : gid - N0;
    float acc = 0.0f;
    if (gid < total) {
        for (int k = lane; k < K; k += 32) {
            acc += A[k] * gguf_q4_k_value(B, k * N + col);
        }
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    int base = slot << 5;
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[base + lane] += scratch[base + lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && gid < total) C[col] = scratch[base];
}

__kernel void matmul_f32_q4_k_m1_3out_f32(__global const float* A,
                                          __global const uchar* B0,
                                          __global const uchar* B1,
                                          __global const uchar* B2,
                                          __global float* C0,
                                          __global float* C1,
                                          __global float* C2,
                                          int N0,
                                          int N1,
                                          int N2,
                                          int K) {
    int gid = get_global_id(0);
    int total01 = N0 + N1;
    int total = total01 + N2;
    if (gid >= total) return;
    __global const uchar* B = gid < N0 ? B0 : (gid < total01 ? B1 : B2);
    __global float* C = gid < N0 ? C0 : (gid < total01 ? C1 : C2);
    int N = gid < N0 ? N0 : (gid < total01 ? N1 : N2);
    int col = gid < N0 ? gid : (gid < total01 ? gid - N0 : gid - total01);
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[k] * gguf_q4_k_value(B, k * N + col);
    }
    C[col] = acc;
}

__kernel void matmul_f32_q5_k_m1_3out_f32(__global const float* A,
                                          __global const uchar* B0,
                                          __global const uchar* B1,
                                          __global const uchar* B2,
                                          __global float* C0,
                                          __global float* C1,
                                          __global float* C2,
                                          int N0,
                                          int N1,
                                          int N2,
                                          int K) {
    int gid = get_global_id(0);
    int total01 = N0 + N1;
    int total = total01 + N2;
    if (gid >= total) return;
    __global const uchar* B = gid < N0 ? B0 : (gid < total01 ? B1 : B2);
    __global float* C = gid < N0 ? C0 : (gid < total01 ? C1 : C2);
    int N = gid < N0 ? N0 : (gid < total01 ? N1 : N2);
    int col = gid < N0 ? gid : (gid < total01 ? gid - N0 : gid - total01);
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[k] * gguf_q5_k_value(B, k * N + col);
    }
    C[col] = acc;
}

__kernel void matmul_f32_q6_k_m1_3out_f32(__global const float* A,
                                          __global const uchar* B0,
                                          __global const uchar* B1,
                                          __global const uchar* B2,
                                          __global float* C0,
                                          __global float* C1,
                                          __global float* C2,
                                          int N0,
                                          int N1,
                                          int N2,
                                          int K) {
    int gid = get_global_id(0);
    int total01 = N0 + N1;
    int total = total01 + N2;
    if (gid >= total) return;
    __global const uchar* B = gid < N0 ? B0 : (gid < total01 ? B1 : B2);
    __global float* C = gid < N0 ? C0 : (gid < total01 ? C1 : C2);
    int N = gid < N0 ? N0 : (gid < total01 ? N1 : N2);
    int col = gid < N0 ? gid : (gid < total01 ? gid - N0 : gid - total01);
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[k] * gguf_q6_k_value(B, k * N + col);
    }
    C[col] = acc;
}

__kernel void matmul_f32_q4_k_m1_3out_wg_f32(__global const float* A,
                                             __global const uchar* B0,
                                             __global const uchar* B1,
                                             __global const uchar* B2,
                                             __global float* C0,
                                             __global float* C1,
                                             __global float* C2,
                                             int N0,
                                             int N1,
                                             int N2,
                                             int K,
                                             __local float* scratch) {
    int gid = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    int total01 = N0 + N1;
    int total = total01 + N2;
    if (gid >= total) return;
    __global const uchar* B = gid < N0 ? B0 : (gid < total01 ? B1 : B2);
    __global float* C = gid < N0 ? C0 : (gid < total01 ? C1 : C2);
    int N = gid < N0 ? N0 : (gid < total01 ? N1 : N2);
    int col = gid < N0 ? gid : (gid < total01 ? gid - N0 : gid - total01);
    float acc = 0.0f;
    for (int k = lid; k < K; k += local_size) {
        acc += A[k] * gguf_q4_k_value(B, k * N + col);
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) C[col] = scratch[0];
}

__kernel void matmul_f32_q5_k_m1_3out_wg_f32(__global const float* A,
                                             __global const uchar* B0,
                                             __global const uchar* B1,
                                             __global const uchar* B2,
                                             __global float* C0,
                                             __global float* C1,
                                             __global float* C2,
                                             int N0,
                                             int N1,
                                             int N2,
                                             int K,
                                             __local float* scratch) {
    int gid = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    int total01 = N0 + N1;
    int total = total01 + N2;
    if (gid >= total) return;
    __global const uchar* B = gid < N0 ? B0 : (gid < total01 ? B1 : B2);
    __global float* C = gid < N0 ? C0 : (gid < total01 ? C1 : C2);
    int N = gid < N0 ? N0 : (gid < total01 ? N1 : N2);
    int col = gid < N0 ? gid : (gid < total01 ? gid - N0 : gid - total01);
    float acc = 0.0f;
    for (int k = lid; k < K; k += local_size) {
        acc += A[k] * gguf_q5_k_value(B, k * N + col);
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) C[col] = scratch[0];
}

__kernel void matmul_f32_q6_k_m1_3out_wg_f32(__global const float* A,
                                             __global const uchar* B0,
                                             __global const uchar* B1,
                                             __global const uchar* B2,
                                             __global float* C0,
                                             __global float* C1,
                                             __global float* C2,
                                             int N0,
                                             int N1,
                                             int N2,
                                             int K,
                                             __local float* scratch) {
    int gid = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    int total01 = N0 + N1;
    int total = total01 + N2;
    if (gid >= total) return;
    __global const uchar* B = gid < N0 ? B0 : (gid < total01 ? B1 : B2);
    __global float* C = gid < N0 ? C0 : (gid < total01 ? C1 : C2);
    int N = gid < N0 ? N0 : (gid < total01 ? N1 : N2);
    int col = gid < N0 ? gid : (gid < total01 ? gid - N0 : gid - total01);
    float acc = 0.0f;
    for (int k = lid; k < K; k += local_size) {
        acc += A[k] * gguf_q6_k_value(B, k * N + col);
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) C[col] = scratch[0];
}

__kernel void matmul_f32_q4_k_m1_3out_wg4_f32(__global const float* A,
                                              __global const uchar* B0,
                                              __global const uchar* B1,
                                              __global const uchar* B2,
                                              __global float* C0,
                                              __global float* C1,
                                              __global float* C2,
                                              int N0,
                                              int N1,
                                              int N2,
                                              int K,
                                              __local float* scratch) {
    int group = get_group_id(0);
    int lid = get_local_id(0);
    int slot = lid >> 5;
    int lane = lid & 31;
    int gid = group * 4 + slot;
    int total01 = N0 + N1;
    int total = total01 + N2;
    __global const uchar* B = gid < N0 ? B0 : (gid < total01 ? B1 : B2);
    __global float* C = gid < N0 ? C0 : (gid < total01 ? C1 : C2);
    int N = gid < N0 ? N0 : (gid < total01 ? N1 : N2);
    int col = gid < N0 ? gid : (gid < total01 ? gid - N0 : gid - total01);
    float acc = 0.0f;
    if (gid < total) {
        for (int k = lane; k < K; k += 32) {
            acc += A[k] * gguf_q4_k_value(B, k * N + col);
        }
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    int base = slot << 5;
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[base + lane] += scratch[base + lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && gid < total) C[col] = scratch[base];
}

__kernel void matmul_q8_0_scaled_f32(__global const char* A,
                                     __global const char* B,
                                     __global float* C,
                                     int M,
                                     int N,
                                     int K,
                                     float scale_a,
                                     float scale_b,
                                     __global const float* scales_a,
                                     __global const float* scales_b,
                                     int mode_a,
                                     int mode_b,
                                     int block_a,
                                     int block_b) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float sa = quant_matmul_scale(scales_a, scale_a, mode_a, row, k, K, block_a);
        float sb = quant_matmul_scale(scales_b, scale_b, mode_b, k, col, N, block_b);
        acc += ((float)((int)A[row * K + k])) * sa * ((float)((int)B[k * N + col])) * sb;
    }
    C[row * N + col] = acc;
}

__kernel void matmul_q4_0_scaled_f32(__global const uchar* A,
                                     __global const uchar* B,
                                     __global float* C,
                                     int M,
                                     int N,
                                     int K,
                                     float scale_a,
                                     float scale_b,
                                     __global const float* scales_a,
                                     __global const float* scales_b,
                                     int mode_a,
                                     int mode_b,
                                     int block_a,
                                     int block_b) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float sa = quant_matmul_scale(scales_a, scale_a, mode_a, row, k, K, block_a);
        float sb = quant_matmul_scale(scales_b, scale_b, mode_b, k, col, N, block_b);
        acc += ((float)q4_0_load(A, row * K + k)) * sa * ((float)q4_0_load(B, k * N + col)) * sb;
    }
    C[row * N + col] = acc;
}

__kernel void matmul_q8_q4_scaled_f32(__global const char* A,
                                      __global const uchar* B,
                                      __global float* C,
                                      int M,
                                      int N,
                                      int K,
                                      float scale_a,
                                      float scale_b,
                                      __global const float* scales_a,
                                      __global const float* scales_b,
                                      int mode_a,
                                      int mode_b,
                                      int block_a,
                                      int block_b) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float sa = quant_matmul_scale(scales_a, scale_a, mode_a, row, k, K, block_a);
        float sb = quant_matmul_scale(scales_b, scale_b, mode_b, k, col, N, block_b);
        acc += ((float)((int)A[row * K + k])) * sa * ((float)q4_0_load(B, k * N + col)) * sb;
    }
    C[row * N + col] = acc;
}

__kernel void matmul_f32_q4_0_scaled_m1_f32(__global const float* A,
                                            __global const uchar* B,
                                            __global float* C,
                                            int N,
                                            int K,
                                            float scale_b,
                                            __global const float* scales_b,
                                            int mode_b,
                                            int block_b) {
    int col = get_global_id(0);
    if (col >= N) return;
    float acc = 0.0f;
    if (mode_b == 2) {
        float sb = scales_b[col];
        for (int k = 0; k < K; ++k) {
            acc += A[k] * ((float)q4_0_load(B, k * N + col)) * sb;
        }
    } else {
        for (int k = 0; k < K; ++k) {
            float sb = quant_matmul_scale(scales_b, scale_b, mode_b, k, col, N, block_b);
            acc += A[k] * ((float)q4_0_load(B, k * N + col)) * sb;
        }
    }
    C[col] = acc;
}

__kernel void matmul_f32_q4_0_col_m1_wg4_f32(__global const float* A,
                                             __global const uchar* B,
                                             __global const float* scales_b,
                                             __global float* C,
                                             int N,
                                             int K,
                                             int blocks_per_col,
                                             int block_b,
                                             __local float* scratch) {
    int group = get_group_id(0);
    int lid = get_local_id(0);
    int slot = lid >> 5;       // 4 output columns per workgroup, 32 lanes each
    int lane = lid & 31;
    int col = group * 4 + slot;
    float acc = 0.0f;
    if (col < N) {
        const int scale_base = col * blocks_per_col;
        for (int k = lane; k < K; k += 32) {
            const int q = q4_0_col_load(B, col, K, k);
            const float sb = scales_b[scale_base + (k / block_b)];
            acc += A[k] * ((float)q) * sb;
        }
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    int base = slot << 5;
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[base + lane] += scratch[base + lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && col < N) C[col] = scratch[base];
}

// AMD/RX580-oriented variant: one wave64 computes four output columns.
// Compared to wg4_f32 it loads A once per lane and accumulates four columns,
// trading a little more per-lane ILP for fewer lanes and better wave64 mapping.
__kernel void matmul_f32_q4_0_col_m1_wg64x4_f32(__global const float* A,
                                                __global const uchar* B,
                                                __global const float* scales_b,
                                                __global float* C,
                                                int N,
                                                int K,
                                                int blocks_per_col,
                                                int block_b,
                                                __local float* scratch) {
    int group = get_group_id(0);
    int lid = get_local_id(0);
    int col0 = group * 4;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const int c0 = col0;
    const int c1 = col0 + 1;
    const int c2 = col0 + 2;
    const int c3 = col0 + 3;
    for (int k = lid; k < K; k += 64) {
        const float av = A[k];
        const int block = k / block_b;
        if (c0 < N) acc0 += av * ((float)q4_0_col_load(B, c0, K, k)) * scales_b[c0 * blocks_per_col + block];
        if (c1 < N) acc1 += av * ((float)q4_0_col_load(B, c1, K, k)) * scales_b[c1 * blocks_per_col + block];
        if (c2 < N) acc2 += av * ((float)q4_0_col_load(B, c2, K, k)) * scales_b[c2 * blocks_per_col + block];
        if (c3 < N) acc3 += av * ((float)q4_0_col_load(B, c3, K, k)) * scales_b[c3 * blocks_per_col + block];
    }
    scratch[lid] = acc0;
    scratch[64 + lid] = acc1;
    scratch[128 + lid] = acc2;
    scratch[192 + lid] = acc3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
            scratch[64 + lid] += scratch[64 + lid + stride];
            scratch[128 + lid] += scratch[128 + lid + stride];
            scratch[192 + lid] += scratch[192 + lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        if (c0 < N) C[c0] = scratch[0];
        if (c1 < N) C[c1] = scratch[64];
        if (c2 < N) C[c2] = scratch[128];
        if (c3 < N) C[c3] = scratch[192];
    }
}

// Tile8 layout: for each output tile of 8 columns, quantized values are stored
// as [k-block][k-within-block][tile-col]. This makes the eight output columns
// contiguous for the same k, avoiding the column-stride jumps of Q4_0_COL.
__kernel void matmul_f32_q4_0_tile8_m1_wg64x8_f32(__global const float* A,
                                                  __global const uchar* B,
                                                  __global const float* scales_b,
                                                  __global float* C,
                                                  int N,
                                                  int K,
                                                  int blocks_per_col,
                                                  int block_b,
                                                  __local float* scratch) {
    const int group = get_group_id(0);
    const int lid = get_local_id(0);
    const int col0 = group * 8;
    const int tile_cols = q4_0_tile8_cols(N, col0);
    const int tile_elem_base = group * blocks_per_col * block_b * 8;
    const int scale_tile_base = group * blocks_per_col * 8;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    for (int k = lid; k < K; k += 64) {
        const float av = A[k];
        const int block = k / block_b;
        const int inner = k - block * block_b;
        const int elem_base = tile_elem_base + (block * block_b + inner) * tile_cols;
        const int scale_base = scale_tile_base + block * tile_cols;
        if (tile_cols == 8) {
            const int byte_base = elem_base >> 1;
            const uchar p0 = B[byte_base + 0];
            const uchar p1 = B[byte_base + 1];
            const uchar p2 = B[byte_base + 2];
            const uchar p3 = B[byte_base + 3];
            acc0 += av * ((float)((int)(p0 & 15) - 8)) * scales_b[scale_base + 0];
            acc1 += av * ((float)((int)((p0 >> 4) & 15) - 8)) * scales_b[scale_base + 1];
            acc2 += av * ((float)((int)(p1 & 15) - 8)) * scales_b[scale_base + 2];
            acc3 += av * ((float)((int)((p1 >> 4) & 15) - 8)) * scales_b[scale_base + 3];
            acc4 += av * ((float)((int)(p2 & 15) - 8)) * scales_b[scale_base + 4];
            acc5 += av * ((float)((int)((p2 >> 4) & 15) - 8)) * scales_b[scale_base + 5];
            acc6 += av * ((float)((int)(p3 & 15) - 8)) * scales_b[scale_base + 6];
            acc7 += av * ((float)((int)((p3 >> 4) & 15) - 8)) * scales_b[scale_base + 7];
        } else {
            if (tile_cols > 0) acc0 += av * ((float)q4_0_tile8_load(B, N, blocks_per_col, block_b, col0 + 0, k)) * scales_b[scale_base + 0];
            if (tile_cols > 1) acc1 += av * ((float)q4_0_tile8_load(B, N, blocks_per_col, block_b, col0 + 1, k)) * scales_b[scale_base + 1];
            if (tile_cols > 2) acc2 += av * ((float)q4_0_tile8_load(B, N, blocks_per_col, block_b, col0 + 2, k)) * scales_b[scale_base + 2];
            if (tile_cols > 3) acc3 += av * ((float)q4_0_tile8_load(B, N, blocks_per_col, block_b, col0 + 3, k)) * scales_b[scale_base + 3];
            if (tile_cols > 4) acc4 += av * ((float)q4_0_tile8_load(B, N, blocks_per_col, block_b, col0 + 4, k)) * scales_b[scale_base + 4];
            if (tile_cols > 5) acc5 += av * ((float)q4_0_tile8_load(B, N, blocks_per_col, block_b, col0 + 5, k)) * scales_b[scale_base + 5];
            if (tile_cols > 6) acc6 += av * ((float)q4_0_tile8_load(B, N, blocks_per_col, block_b, col0 + 6, k)) * scales_b[scale_base + 6];
        }
    }
    scratch[lid] = acc0;
    scratch[64 + lid] = acc1;
    scratch[128 + lid] = acc2;
    scratch[192 + lid] = acc3;
    scratch[256 + lid] = acc4;
    scratch[320 + lid] = acc5;
    scratch[384 + lid] = acc6;
    scratch[448 + lid] = acc7;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
            scratch[64 + lid] += scratch[64 + lid + stride];
            scratch[128 + lid] += scratch[128 + lid + stride];
            scratch[192 + lid] += scratch[192 + lid + stride];
            scratch[256 + lid] += scratch[256 + lid + stride];
            scratch[320 + lid] += scratch[320 + lid + stride];
            scratch[384 + lid] += scratch[384 + lid + stride];
            scratch[448 + lid] += scratch[448 + lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        if (tile_cols > 0) C[col0 + 0] = scratch[0];
        if (tile_cols > 1) C[col0 + 1] = scratch[64];
        if (tile_cols > 2) C[col0 + 2] = scratch[128];
        if (tile_cols > 3) C[col0 + 3] = scratch[192];
        if (tile_cols > 4) C[col0 + 4] = scratch[256];
        if (tile_cols > 5) C[col0 + 5] = scratch[320];
        if (tile_cols > 6) C[col0 + 6] = scratch[384];
        if (tile_cols > 7) C[col0 + 7] = scratch[448];
    }
}

__kernel void matmul_f32_q4_0_tile8_m1_wg64x16_f32(__global const float* A,
                                                   __global const uchar* B,
                                                   __global const float* scales_b,
                                                   __global float* C,
                                                   int N,
                                                   int K,
                                                   int blocks_per_col,
                                                   int block_b,
                                                   __local float* scratch) {
    const int group = get_group_id(0);
    const int lid = get_local_id(0);
    const int col0 = group * 16;
    float acc[16] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    for (int k = lid; k < K; k += 64) {
        const float av = A[k];
        if (col0 < N) q4_0_tile8_accum8(B, scales_b, N, blocks_per_col, block_b, col0, k, av, acc);
        if (col0 + 8 < N) q4_0_tile8_accum8(B, scales_b, N, blocks_per_col, block_b, col0 + 8, k, av, acc + 8);
    }
    for (int j = 0; j < 16; ++j) scratch[j * 64 + lid] = acc[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (int j = 0; j < 16; ++j) scratch[j * 64 + lid] += scratch[j * 64 + lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        for (int j = 0; j < 16; ++j) {
            const int col = col0 + j;
            if (col < N) C[col] = scratch[j * 64];
        }
    }
}

__kernel void matmul_silu_product_q4_0_col_m1_wg64x4_f32(__global const float* gate,
                                                         __global const float* up,
                                                         __global const uchar* B,
                                                         __global const float* scales_b,
                                                         __global float* C,
                                                         int N,
                                                         int K,
                                                         int blocks_per_col,
                                                         int block_b,
                                                         __local float* scratch) {
    int group = get_group_id(0);
    int lid = get_local_id(0);
    int col0 = group * 4;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const int c0 = col0;
    const int c1 = col0 + 1;
    const int c2 = col0 + 2;
    const int c3 = col0 + 3;
    for (int k = lid; k < K; k += 64) {
        const float g = gate[k];
        const float av = (g * native_recip(1.0f + native_exp(-g))) * up[k];
        const int block = k / block_b;
        if (c0 < N) acc0 += av * ((float)q4_0_col_load(B, c0, K, k)) * scales_b[c0 * blocks_per_col + block];
        if (c1 < N) acc1 += av * ((float)q4_0_col_load(B, c1, K, k)) * scales_b[c1 * blocks_per_col + block];
        if (c2 < N) acc2 += av * ((float)q4_0_col_load(B, c2, K, k)) * scales_b[c2 * blocks_per_col + block];
        if (c3 < N) acc3 += av * ((float)q4_0_col_load(B, c3, K, k)) * scales_b[c3 * blocks_per_col + block];
    }
    scratch[lid] = acc0;
    scratch[64 + lid] = acc1;
    scratch[128 + lid] = acc2;
    scratch[192 + lid] = acc3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
            scratch[64 + lid] += scratch[64 + lid + stride];
            scratch[128 + lid] += scratch[128 + lid + stride];
            scratch[192 + lid] += scratch[192 + lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        if (c0 < N) C[c0] = scratch[0];
        if (c1 < N) C[c1] = scratch[64];
        if (c2 < N) C[c2] = scratch[128];
        if (c3 < N) C[c3] = scratch[192];
    }
}


__kernel void matmul_silu_product_q4_0_tile8_m1_wg64x8_f32(__global const float* gate,
                                                           __global const float* up,
                                                           __global const uchar* B,
                                                           __global const float* scales_b,
                                                           __global float* C,
                                                           int N,
                                                           int K,
                                                           int blocks_per_col,
                                                           int block_b,
                                                           __local float* scratch) {
    const int group = get_group_id(0);
    const int lid = get_local_id(0);
    const int col0 = group * 8;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = lid; k < K; k += 64) {
        const float g = gate[k];
        const float av = (g * native_recip(1.0f + native_exp(-g))) * up[k];
        if (col0 + 7 < N) {
            q4_0_tile8_accum8(B, scales_b, N, blocks_per_col, block_b, col0, k, av, acc);
        } else {
            for (int j = 0; j < 8; ++j) {
                const int col = col0 + j;
                if (col < N) {
                    acc[j] += av * ((float)q4_0_tile8_load(B, N, blocks_per_col, block_b, col, k)) *
                              q4_0_tile8_scale(scales_b, N, blocks_per_col, block_b, col, k);
                }
            }
        }
    }
    for (int j = 0; j < 8; ++j) scratch[j * 64 + lid] = acc[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (int j = 0; j < 8; ++j) scratch[j * 64 + lid] += scratch[j * 64 + lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        for (int j = 0; j < 8; ++j) {
            const int col = col0 + j;
            if (col < N) C[col] = scratch[j * 64];
        }
    }
}

__kernel void matmul_f32_q4_0_col_m1_2out_wg4_f32(__global const float* A,
                                                  __global const uchar* B0,
                                                  __global const float* S0,
                                                  __global const uchar* B1,
                                                  __global const float* S1,
                                                  __global float* C0,
                                                  __global float* C1,
                                                  int N0,
                                                  int N1,
                                                  int K,
                                                  int blocks_per_col,
                                                  int block_b,
                                                  __local float* scratch) {
    int group = get_group_id(0);
    int lid = get_local_id(0);
    int slot = lid >> 5;
    int lane = lid & 31;
    int gid = group * 4 + slot;
    int total = N0 + N1;
    float acc = 0.0f;
    if (gid < total) {
        const int first = gid < N0;
        int col = first ? gid : gid - N0;
        __global const uchar* B = first ? B0 : B1;
        __global const float* S = first ? S0 : S1;
        const int scale_base = col * blocks_per_col;
        for (int k = lane; k < K; k += 32) {
            const int q = q4_0_col_load(B, col, K, k);
            const float sb = S[scale_base + (k / block_b)];
            acc += A[k] * ((float)q) * sb;
        }
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    int base = slot << 5;
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[base + lane] += scratch[base + lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && gid < total) {
        if (gid < N0) C0[gid] = scratch[base];
        else C1[gid - N0] = scratch[base];
    }
}

__kernel void matmul_f32_q4_0_col_m1_2out_wg64x4_f32(__global const float* A,
                                                     __global const uchar* B0,
                                                     __global const float* S0,
                                                     __global const uchar* B1,
                                                     __global const float* S1,
                                                     __global float* C0,
                                                     __global float* C1,
                                                     int N0,
                                                     int N1,
                                                     int K,
                                                     int blocks_per_col,
                                                     int block_b,
                                                     __local float* scratch) {
    int group = get_group_id(0);
    int lid = get_local_id(0);
    int gid0 = group * 4;
    int total = N0 + N1;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int k = lid; k < K; k += 64) {
        const float av = A[k];
        const int block = k / block_b;
        int gid = gid0;
        if (gid < total) {
            int first = gid < N0;
            int col = first ? gid : gid - N0;
            __global const uchar* B = first ? B0 : B1;
            __global const float* S = first ? S0 : S1;
            acc0 += av * ((float)q4_0_col_load(B, col, K, k)) * S[col * blocks_per_col + block];
        }
        gid = gid0 + 1;
        if (gid < total) {
            int first = gid < N0;
            int col = first ? gid : gid - N0;
            __global const uchar* B = first ? B0 : B1;
            __global const float* S = first ? S0 : S1;
            acc1 += av * ((float)q4_0_col_load(B, col, K, k)) * S[col * blocks_per_col + block];
        }
        gid = gid0 + 2;
        if (gid < total) {
            int first = gid < N0;
            int col = first ? gid : gid - N0;
            __global const uchar* B = first ? B0 : B1;
            __global const float* S = first ? S0 : S1;
            acc2 += av * ((float)q4_0_col_load(B, col, K, k)) * S[col * blocks_per_col + block];
        }
        gid = gid0 + 3;
        if (gid < total) {
            int first = gid < N0;
            int col = first ? gid : gid - N0;
            __global const uchar* B = first ? B0 : B1;
            __global const float* S = first ? S0 : S1;
            acc3 += av * ((float)q4_0_col_load(B, col, K, k)) * S[col * blocks_per_col + block];
        }
    }
    scratch[lid] = acc0;
    scratch[64 + lid] = acc1;
    scratch[128 + lid] = acc2;
    scratch[192 + lid] = acc3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
            scratch[64 + lid] += scratch[64 + lid + stride];
            scratch[128 + lid] += scratch[128 + lid + stride];
            scratch[192 + lid] += scratch[192 + lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        int gid = gid0;
        if (gid < N0) C0[gid] = scratch[0];
        else if (gid < total) C1[gid - N0] = scratch[0];
        gid = gid0 + 1;
        if (gid < N0) C0[gid] = scratch[64];
        else if (gid < total) C1[gid - N0] = scratch[64];
        gid = gid0 + 2;
        if (gid < N0) C0[gid] = scratch[128];
        else if (gid < total) C1[gid - N0] = scratch[128];
        gid = gid0 + 3;
        if (gid < N0) C0[gid] = scratch[192];
        else if (gid < total) C1[gid - N0] = scratch[192];
    }
}


__kernel void matmul_rmsnorm_f32_q4_0_tile8_m1_2out_wg64x8_f32(__global const float* X,
                                                                __global const float* norm_weight,
                                                                __global const uchar* B0,
                                                                __global const float* S0,
                                                                __global const uchar* B1,
                                                                __global const float* S1,
                                                                __global float* C0,
                                                                __global float* C1,
                                                                int N0,
                                                                int N1,
                                                                int K,
                                                                int blocks_per_col,
                                                                int block_b,
                                                                float eps,
                                                                __local float* scratch) {
    const int group = get_group_id(0);
    const int lid = get_local_id(0);
    const int gid0 = group * 8;
    const int total = N0 + N1;

    float ss = 0.0f;
    for (int k = lid; k < K; k += 64) {
        const float x = X[k];
        ss += x * x;
    }
    scratch[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = rsqrt(scratch[0] / (float)K + eps);
    barrier(CLK_LOCAL_MEM_FENCE);

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = lid; k < K; k += 64) {
        const float av = X[k] * inv * norm_weight[k];
        if (gid0 + 7 < N0) {
            q4_0_tile8_accum8(B0, S0, N0, blocks_per_col, block_b, gid0, k, av, acc);
        } else if (gid0 >= N0 && gid0 + 7 < total) {
            q4_0_tile8_accum8(B1, S1, N1, blocks_per_col, block_b, gid0 - N0, k, av, acc);
        } else {
            for (int j = 0; j < 8; ++j) {
                const int gid = gid0 + j;
                if (gid < total) {
                    const int first = gid < N0;
                    const int col = first ? gid : gid - N0;
                    __global const uchar* B = first ? B0 : B1;
                    __global const float* S = first ? S0 : S1;
                    const int N = first ? N0 : N1;
                    acc[j] += av * ((float)q4_0_tile8_load(B, N, blocks_per_col, block_b, col, k)) *
                              q4_0_tile8_scale(S, N, blocks_per_col, block_b, col, k);
                }
            }
        }
    }
    for (int j = 0; j < 8; ++j) scratch[j * 64 + lid] = acc[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (int j = 0; j < 8; ++j) scratch[j * 64 + lid] += scratch[j * 64 + lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        for (int j = 0; j < 8; ++j) {
            const int gid = gid0 + j;
            if (gid < N0) C0[gid] = scratch[j * 64];
            else if (gid < total) C1[gid - N0] = scratch[j * 64];
        }
    }
}

__kernel void matmul_f32_q4_0_tile8_m1_2out_wg64x8_f32(__global const float* A,
                                                       __global const uchar* B0,
                                                       __global const float* S0,
                                                       __global const uchar* B1,
                                                       __global const float* S1,
                                                       __global float* C0,
                                                       __global float* C1,
                                                       int N0,
                                                       int N1,
                                                       int K,
                                                       int blocks_per_col,
                                                       int block_b,
                                                       __local float* scratch) {
    const int group = get_group_id(0);
    const int lid = get_local_id(0);
    const int gid0 = group * 8;
    const int total = N0 + N1;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = lid; k < K; k += 64) {
        const float av = A[k];
        if (gid0 + 7 < N0) {
            q4_0_tile8_accum8(B0, S0, N0, blocks_per_col, block_b, gid0, k, av, acc);
        } else if (gid0 >= N0 && gid0 + 7 < total) {
            q4_0_tile8_accum8(B1, S1, N1, blocks_per_col, block_b, gid0 - N0, k, av, acc);
        } else {
            for (int j = 0; j < 8; ++j) {
                const int gid = gid0 + j;
                if (gid < total) {
                    const int first = gid < N0;
                    const int col = first ? gid : gid - N0;
                    __global const uchar* B = first ? B0 : B1;
                    __global const float* S = first ? S0 : S1;
                    const int N = first ? N0 : N1;
                    acc[j] += av * ((float)q4_0_tile8_load(B, N, blocks_per_col, block_b, col, k)) *
                              q4_0_tile8_scale(S, N, blocks_per_col, block_b, col, k);
                }
            }
        }
    }
    for (int j = 0; j < 8; ++j) scratch[j * 64 + lid] = acc[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (int j = 0; j < 8; ++j) scratch[j * 64 + lid] += scratch[j * 64 + lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        for (int j = 0; j < 8; ++j) {
            const int gid = gid0 + j;
            if (gid < N0) C0[gid] = scratch[j * 64];
            else if (gid < total) C1[gid - N0] = scratch[j * 64];
        }
    }
}

__kernel void matmul_f32_q4_0_col_m1_3out_wg4_f32(__global const float* A,
                                                  __global const uchar* B0,
                                                  __global const float* S0,
                                                  __global const uchar* B1,
                                                  __global const float* S1,
                                                  __global const uchar* B2,
                                                  __global const float* S2,
                                                  __global float* C0,
                                                  __global float* C1,
                                                  __global float* C2,
                                                  int N0,
                                                  int N1,
                                                  int N2,
                                                  int K,
                                                  int blocks_per_col,
                                                  int block_b,
                                                  __local float* scratch) {
    int group = get_group_id(0);
    int lid = get_local_id(0);
    int slot = lid >> 5;
    int lane = lid & 31;
    int gid = group * 4 + slot;
    int total01 = N0 + N1;
    int total = total01 + N2;
    float acc = 0.0f;
    if (gid < total) {
        int col = gid;
        __global const uchar* B = B0;
        __global const float* S = S0;
        if (gid >= total01) {
            col = gid - total01;
            B = B2;
            S = S2;
        } else if (gid >= N0) {
            col = gid - N0;
            B = B1;
            S = S1;
        }
        const int scale_base = col * blocks_per_col;
        for (int k = lane; k < K; k += 32) {
            const int q = q4_0_col_load(B, col, K, k);
            const float sb = S[scale_base + (k / block_b)];
            acc += A[k] * ((float)q) * sb;
        }
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    int base = slot << 5;
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[base + lane] += scratch[base + lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && gid < total) {
        if (gid < N0) C0[gid] = scratch[base];
        else if (gid < total01) C1[gid - N0] = scratch[base];
        else C2[gid - total01] = scratch[base];
    }
}

__kernel void matmul_f32_q4_0_col_m1_3out_wg64x4_f32(__global const float* A,
                                                     __global const uchar* B0,
                                                     __global const float* S0,
                                                     __global const uchar* B1,
                                                     __global const float* S1,
                                                     __global const uchar* B2,
                                                     __global const float* S2,
                                                     __global float* C0,
                                                     __global float* C1,
                                                     __global float* C2,
                                                     int N0,
                                                     int N1,
                                                     int N2,
                                                     int K,
                                                     int blocks_per_col,
                                                     int block_b,
                                                     __local float* scratch) {
    int group = get_group_id(0);
    int lid = get_local_id(0);
    int gid0 = group * 4;
    int total01 = N0 + N1;
    int total = total01 + N2;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int k = lid; k < K; k += 64) {
        const float av = A[k];
        const int block = k / block_b;
        int gid = gid0;
        if (gid < total) {
            int col = gid;
            __global const uchar* B = B0;
            __global const float* S = S0;
            if (gid >= total01) { col = gid - total01; B = B2; S = S2; }
            else if (gid >= N0) { col = gid - N0; B = B1; S = S1; }
            acc0 += av * ((float)q4_0_col_load(B, col, K, k)) * S[col * blocks_per_col + block];
        }
        gid = gid0 + 1;
        if (gid < total) {
            int col = gid;
            __global const uchar* B = B0;
            __global const float* S = S0;
            if (gid >= total01) { col = gid - total01; B = B2; S = S2; }
            else if (gid >= N0) { col = gid - N0; B = B1; S = S1; }
            acc1 += av * ((float)q4_0_col_load(B, col, K, k)) * S[col * blocks_per_col + block];
        }
        gid = gid0 + 2;
        if (gid < total) {
            int col = gid;
            __global const uchar* B = B0;
            __global const float* S = S0;
            if (gid >= total01) { col = gid - total01; B = B2; S = S2; }
            else if (gid >= N0) { col = gid - N0; B = B1; S = S1; }
            acc2 += av * ((float)q4_0_col_load(B, col, K, k)) * S[col * blocks_per_col + block];
        }
        gid = gid0 + 3;
        if (gid < total) {
            int col = gid;
            __global const uchar* B = B0;
            __global const float* S = S0;
            if (gid >= total01) { col = gid - total01; B = B2; S = S2; }
            else if (gid >= N0) { col = gid - N0; B = B1; S = S1; }
            acc3 += av * ((float)q4_0_col_load(B, col, K, k)) * S[col * blocks_per_col + block];
        }
    }
    scratch[lid] = acc0;
    scratch[64 + lid] = acc1;
    scratch[128 + lid] = acc2;
    scratch[192 + lid] = acc3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
            scratch[64 + lid] += scratch[64 + lid + stride];
            scratch[128 + lid] += scratch[128 + lid + stride];
            scratch[192 + lid] += scratch[192 + lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        int gid = gid0;
        if (gid < N0) C0[gid] = scratch[0];
        else if (gid < total01) C1[gid - N0] = scratch[0];
        else if (gid < total) C2[gid - total01] = scratch[0];
        gid = gid0 + 1;
        if (gid < N0) C0[gid] = scratch[64];
        else if (gid < total01) C1[gid - N0] = scratch[64];
        else if (gid < total) C2[gid - total01] = scratch[64];
        gid = gid0 + 2;
        if (gid < N0) C0[gid] = scratch[128];
        else if (gid < total01) C1[gid - N0] = scratch[128];
        else if (gid < total) C2[gid - total01] = scratch[128];
        gid = gid0 + 3;
        if (gid < N0) C0[gid] = scratch[192];
        else if (gid < total01) C1[gid - N0] = scratch[192];
        else if (gid < total) C2[gid - total01] = scratch[192];
    }
}

__kernel void matmul_f32_q4_0_tile8_m1_3out_wg64x8_f32(__global const float* A,
                                                       __global const uchar* B0,
                                                       __global const float* S0,
                                                       __global const uchar* B1,
                                                       __global const float* S1,
                                                       __global const uchar* B2,
                                                       __global const float* S2,
                                                       __global float* C0,
                                                       __global float* C1,
                                                       __global float* C2,
                                                       int N0,
                                                       int N1,
                                                       int N2,
                                                       int K,
                                                       int blocks_per_col,
                                                       int block_b,
                                                       __local float* scratch) {
    const int group = get_group_id(0);
    const int lid = get_local_id(0);
    const int gid0 = group * 8;
    const int total01 = N0 + N1;
    const int total = total01 + N2;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = lid; k < K; k += 64) {
        const float av = A[k];
        if (gid0 + 7 < N0) {
            q4_0_tile8_accum8(B0, S0, N0, blocks_per_col, block_b, gid0, k, av, acc);
        } else if (gid0 >= N0 && gid0 + 7 < total01) {
            q4_0_tile8_accum8(B1, S1, N1, blocks_per_col, block_b, gid0 - N0, k, av, acc);
        } else if (gid0 >= total01 && gid0 + 7 < total) {
            q4_0_tile8_accum8(B2, S2, N2, blocks_per_col, block_b, gid0 - total01, k, av, acc);
        } else {
            for (int j = 0; j < 8; ++j) {
                const int gid = gid0 + j;
                if (gid < total) {
                    int col = gid;
                    int N = N0;
                    __global const uchar* B = B0;
                    __global const float* S = S0;
                    if (gid >= total01) {
                        col = gid - total01;
                        N = N2;
                        B = B2;
                        S = S2;
                    } else if (gid >= N0) {
                        col = gid - N0;
                        N = N1;
                        B = B1;
                        S = S1;
                    }
                    acc[j] += av * ((float)q4_0_tile8_load(B, N, blocks_per_col, block_b, col, k)) *
                              q4_0_tile8_scale(S, N, blocks_per_col, block_b, col, k);
                }
            }
        }
    }
    for (int j = 0; j < 8; ++j) scratch[j * 64 + lid] = acc[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (int j = 0; j < 8; ++j) scratch[j * 64 + lid] += scratch[j * 64 + lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        for (int j = 0; j < 8; ++j) {
            const int gid = gid0 + j;
            if (gid < N0) C0[gid] = scratch[j * 64];
            else if (gid < total01) C1[gid - N0] = scratch[j * 64];
            else if (gid < total) C2[gid - total01] = scratch[j * 64];
        }
    }
}

__kernel void matmul_q4_q8_scaled_f32(__global const uchar* A,
                                      __global const char* B,
                                      __global float* C,
                                      int M,
                                      int N,
                                      int K,
                                      float scale_a,
                                      float scale_b,
                                      __global const float* scales_a,
                                      __global const float* scales_b,
                                      int mode_a,
                                      int mode_b,
                                      int block_a,
                                      int block_b) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float sa = quant_matmul_scale(scales_a, scale_a, mode_a, row, k, K, block_a);
        float sb = quant_matmul_scale(scales_b, scale_b, mode_b, k, col, N, block_b);
        acc += ((float)q4_0_load(A, row * K + k)) * sa * ((float)((int)B[k * N + col])) * sb;
    }
    C[row * N + col] = acc;
}
