#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void cast_f32_to_f16(__global const float* x,
                              __global half* out,
                              int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = (half)x[gid];
}

__kernel void cast_f16_to_f32(__global const half* x,
                              __global float* out,
                              int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = (float)x[gid];
}

__kernel void matmul_f16_accum_f32(__global const half* A,
                                   __global const half* B,
                                   __global float* C,
                                   int M,
                                   int N,
                                   int K) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += (float)A[row * K + k] * (float)B[k * N + col];
    }
    C[row * N + col] = acc;
}
