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
